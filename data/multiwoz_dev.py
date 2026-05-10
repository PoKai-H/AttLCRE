import json
import os
import random
import re
import tiktoken
from difflib import SequenceMatcher
from collections import Counter

enc = tiktoken.get_encoding("cl100k_base")

BASE_DIR = "/Users/reeseliu/Desktop/Linear attention/multiwoz/data/MultiWOZ_2.2"
SPLITS = ["dev"]
OUTPUT_BASE = "multiwoz_hotel_restaurant_train"
os.makedirs(OUTPUT_BASE, exist_ok=True)

MIN_TURNS = 51
MAX_BG_ATTEMPTS = 300
USE_EMBEDDING_CHECK = False

ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}

sample_counter = 0
task_counter = Counter()

INTRO_BANK = [
    [
        {"speaker": "A", "text": "Hello, I need some help planning part of my trip."},
        {"speaker": "B", "text": "Of course, I can help with that."},
        {"speaker": "A", "text": "I still need to arrange a few things while I am in Cambridge."},
        {"speaker": "B", "text": "Sure, tell me what you need."},
    ],
    [
        {"speaker": "A", "text": "Hi, I need help organizing some travel details."},
        {"speaker": "B", "text": "Certainly, I can help with that."},
        {"speaker": "A", "text": "There are a couple of things I have not settled yet."},
        {"speaker": "B", "text": "No problem, go ahead."},
    ],
    [
        {"speaker": "A", "text": "Good afternoon, I need some assistance with my plans."},
        {"speaker": "B", "text": "I would be happy to help."},
        {"speaker": "A", "text": "I still have a few arrangements to make."},
        {"speaker": "B", "text": "All right, let me know the details."},
    ],
]

NOISE_BANK = [
    ("A", "I am also looking for information about a local museum."),
    ("B", "Certainly, I can help with that."),
    ("A", "What is the address, please?"),
    ("B", "It is located near the centre of town."),
    ("A", "Could you also tell me the phone number?"),
    ("B", "Yes, of course."),
    ("A", "I may also need a taxi later in the evening."),
    ("B", "Where would you like to go?"),
    ("A", "I will need to leave from the museum."),
    ("B", "What time would you like to be picked up?"),
    ("A", "Could you help me find a place to eat as well?"),
    ("B", "Sure, what kind of food would you prefer?"),
    ("A", "I would like something inexpensive."),
    ("B", "There are a few options available."),
    ("A", "Can you also give me the postcode?"),
    ("B", "Yes, I can provide that information."),
    ("A", "I would also like to know the entrance fee."),
    ("B", "The entrance fee is five pounds."),
    ("A", "Is it open on Monday?"),
    ("B", "Yes, it is open that day."),
    ("A", "Could you recommend a place in the centre?"),
    ("B", "There are several possibilities."),
    ("A", "I also need the contact number, please."),
    ("B", "I can provide the contact details."),
    ("A", "Would it be possible to arrange transportation for later?"),
    ("B", "Yes, I can help arrange that."),
    ("A", "I may want to visit an attraction before dinner."),
    ("B", "There are a few popular attractions nearby."),
    ("A", "Could you tell me more about one of them?"),
    ("B", "Certainly, what would you like to know?"),
]

HOTEL_QUERY_BANK = [
    "Which place should the user choose?",
    "Which hotel best matches the user's original request?",
    "Which accommodation satisfies the user's earlier constraints?",
    "Which option fits the requested stay?",
]

RESTAURANT_QUERY_BANK = [
    "Which restaurant should the user choose?",
    "Which restaurant best matches the user's earlier request?",
    "Which dining option satisfies the user's original preferences?",
    "Which place to eat fits the user's constraints?",
]

TRAIN_QUERY_BANK = [
    "Which train should the user take?",
    "Which train best matches the user's earlier request?",
    "Which option satisfies the user's original travel constraints?",
    "Which train fits the requested route and timing?",
]

HOTEL_DISTRACTOR_BANK = [
    [
        {"speaker": "A", "text": "Actually, staying in the centre might also be okay."},
        {"speaker": "B", "text": "Okay, noted."},
    ],
    [
        {"speaker": "A", "text": "I suppose another area could also work if needed."},
        {"speaker": "B", "text": "I understand."},
    ],
    [
        {"speaker": "A", "text": "I guess parking may not matter as much as I first thought."},
        {"speaker": "B", "text": "All right, I will keep that in mind."},
    ],
    [
        {"speaker": "A", "text": "Perhaps the number of stars is not so important after all."},
        {"speaker": "B", "text": "Okay, I understand."},
    ],
]

RESTAURANT_DISTRACTOR_BANK = [
    [
        {"speaker": "A", "text": "Actually, a different area might also be acceptable."},
        {"speaker": "B", "text": "All right, noted."},
    ],
    [
        {"speaker": "A", "text": "I suppose a slightly more expensive place could also work."},
        {"speaker": "B", "text": "Okay, I will keep that in mind."},
    ],
    [
        {"speaker": "A", "text": "Maybe the type of food does not matter quite as much."},
        {"speaker": "B", "text": "Understood."},
    ],
]

TRAIN_DISTRACTOR_BANK = [
    [
        {"speaker": "A", "text": "Actually, I might be okay with a later train."},
        {"speaker": "B", "text": "Okay, I will keep that in mind."},
    ],
    [
        {"speaker": "A", "text": "I suppose leaving from a different station could also work."},
        {"speaker": "B", "text": "All right, noted."},
    ],
    [
        {"speaker": "A", "text": "Maybe arriving a little later would still be acceptable."},
        {"speaker": "B", "text": "I understand."},
    ],
]


def get_turn_range(n):
    if n <= 10:
        return "1-10"
    elif n <= 15:
        return "10-15"
    elif n <= 20:
        return "15-20"
    elif n <= 25:
        return "20-25"
    else:
        return "25+"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def ngram_set(text: str, n: int = 4):
    toks = normalize_text(text).split()
    return set(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))


def ngram_overlap(a: str, b: str, n: int = 4) -> float:
    sa = ngram_set(a, n)
    sb = ngram_set(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def too_similar(candidate_text: str, existing_texts, lexical_th=0.92, ngram_th=0.60) -> bool:
    for prev in existing_texts:
        if lexical_similarity(candidate_text, prev) >= lexical_th:
            return True
        if ngram_overlap(candidate_text, prev, n=4) >= ngram_th:
            return True
    return False


def extract_raw_turns(dialog):
    lines = []
    turns = dialog.get("turns", [])

    for turn in turns:
        speaker = turn.get("speaker", "").strip().upper()
        utterance = turn.get("utterance", "").strip()

        if not utterance:
            continue

        if speaker == "USER":
            role = "A"
        elif speaker == "SYSTEM":
            role = "B"
        else:
            continue

        lines.append({"speaker": role, "text": utterance})

    return lines


def detect_main_task(raw_turns):
    text = " ".join(t["text"].lower() for t in raw_turns)

    hotel_kw = [
        "guesthouse", "hotel", "parking", "internet", "stars", "star",
        "stay", "area", "pricerange", "price range"
    ]
    restaurant_kw = [
        "restaurant", "food", "table", "eat", "pricerange",
        "cheap", "moderate", "expensive", "booktime"
    ]
    train_kw = [
        "train", "departure", "destination", "arrive", "arriveby",
        "leave", "leaveat", "station"
    ]

    scores = {
        "hotel": sum(k in text for k in hotel_kw),
        "restaurant": sum(k in text for k in restaurant_kw),
        "train": sum(k in text for k in train_kw),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


def find_first_match(text_low, options):
    for x in options:
        if re.search(rf"\b{re.escape(x)}\b", text_low):
            return x
    return None


def clean_place_name(name):
    if not name:
        return None
    name = name.strip(" .?,!")
    name = re.sub(r"\s+", " ", name)
    return name


def extract_hotel_constraints(raw_turns):
    text = " ".join(t["text"] for t in raw_turns)
    text_low = text.lower()

    area = find_first_match(text_low, ["north", "south", "east", "west", "centre", "center"])
    stars = None
    parking = None
    internet = None
    hotel_type = None
    price = find_first_match(text_low, ["cheap", "moderate", "expensive"])

    m = re.search(r"\b([1-5])\s*star\b", text_low)
    if m:
        stars = m.group(1)

    if "free parking" in text_low or "parking" in text_low:
        parking = "yes"
    if "internet" in text_low or "wifi" in text_low:
        internet = "yes"

    if "guesthouse" in text_low:
        hotel_type = "guesthouse"
    elif "hotel" in text_low:
        hotel_type = "hotel"

    slots = {
        "area": area,
        "stars": stars,
        "parking": parking,
        "internet": internet,
        "type": hotel_type,
        "pricerange": price,
    }

    if sum(v is not None for v in slots.values()) < 2:
        return None
    return slots


def extract_restaurant_constraints(raw_turns):
    text = " ".join(t["text"] for t in raw_turns)
    text_low = text.lower()

    area = find_first_match(text_low, ["north", "south", "east", "west", "centre", "center"])
    price = find_first_match(text_low, ["cheap", "moderate", "expensive"])
    food = find_first_match(
        text_low,
        ["chinese", "indian", "italian", "british", "french", "thai",
         "asian", "european", "vegetarian", "pizza", "steak"]
    )

    time_match = re.search(r"\b(?:at|for)\s+(\d{1,2}:\d{2})\b", text_low)
    booktime = time_match.group(1) if time_match else None

    slots = {
        "area": area,
        "pricerange": price,
        "food": food,
        "booktime": booktime,
    }

    if sum(v is not None for v in slots.values()) < 2:
        return None
    return slots


def extract_train_constraints(raw_turns):
    text = " ".join(t["text"] for t in raw_turns)
    text_low = text.lower()

    departure = None
    destination = None
    leaveat = None
    arriveby = None
    day = find_first_match(
        text_low,
        ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    )

    # simple route extraction
    m = re.search(r"\bfrom\s+([a-z ]+?)\s+to\s+([a-z ]+?)(?:[.,]| leaving| arriving| on | by | after | before |$)", text_low)
    if m:
        departure = clean_place_name(m.group(1))
        destination = clean_place_name(m.group(2))
    else:
        m1 = re.search(r"\bfrom\s+([a-z ]+?)(?:[.,]| to | leaving| arriving| on | by | after | before |$)", text_low)
        m2 = re.search(r"\bto\s+([a-z ]+?)(?:[.,]| from | leaving| arriving| on | by | after | before |$)", text_low)
        if m1:
            departure = clean_place_name(m1.group(1))
        if m2:
            destination = clean_place_name(m2.group(1))

    m = re.search(r"\bleave(?: at| after)?\s+(\d{1,2}:\d{2})\b", text_low)
    if m:
        leaveat = m.group(1)

    m = re.search(r"\barrive(?: by)?\s+(\d{1,2}:\d{2})\b", text_low)
    if m:
        arriveby = m.group(1)

    # extra patterns
    m = re.search(r"\bafter\s+(\d{1,2}:\d{2})\b", text_low)
    if m and leaveat is None:
        leaveat = m.group(1)

    m = re.search(r"\bby\s+(\d{1,2}:\d{2})\b", text_low)
    if m and arriveby is None and "arrive" in text_low:
        arriveby = m.group(1)

    slots = {
        "departure": departure,
        "destination": destination,
        "leaveat": leaveat,
        "arriveby": arriveby,
        "day": day,
    }

    if sum(v is not None for v in slots.values()) < 2:
        return None
    return slots


def build_hotel_signal(slots):
    templates = []

    parts_a = []
    if slots.get("pricerange"):
        parts_a.append(slots["pricerange"])
    if slots.get("stars"):
        parts_a.append(f"{slots['stars']} star")
    if slots.get("type"):
        parts_a.append(slots["type"])
    else:
        parts_a.append("hotel")
    if slots.get("area"):
        area_word = "centre" if slots["area"] == "center" else slots["area"]
        parts_a.append(f"in the {area_word}")
    if slots.get("parking") == "yes":
        parts_a.append("with free parking")
    if slots.get("internet") == "yes":
        parts_a.append("with internet")
    if len(parts_a) >= 2:
        templates.append("I would like a " + " ".join(parts_a) + ".")

    parts_b = ["I need a place to stay"]
    if slots.get("area"):
        area_word = "centre" if slots["area"] == "center" else slots["area"]
        parts_b.append(f"in the {area_word}")
    if slots.get("type"):
        parts_b.append(f"that is a {slots['type']}")
    if slots.get("stars"):
        parts_b.append(f"with {slots['stars']} stars")
    if slots.get("pricerange"):
        parts_b.append(f"in the {slots['pricerange']} price range")
    if slots.get("parking") == "yes":
        parts_b.append("and free parking")
    if slots.get("internet") == "yes":
        parts_b.append("and internet access")
    templates.append(" ".join(parts_b) + ".")

    parts_c = []
    if slots.get("type"):
        parts_c.append(f"The user wants a {slots['type']}")
    else:
        parts_c.append("The user wants a hotel")
    if slots.get("area"):
        area_word = "centre" if slots["area"] == "center" else slots["area"]
        parts_c.append(f"in the {area_word}")
    if slots.get("stars"):
        parts_c.append(f"with {slots['stars']} stars")
    if slots.get("pricerange"):
        parts_c.append(f"that is {slots['pricerange']}")
    if slots.get("parking") == "yes":
        parts_c.append("with parking")
    if slots.get("internet") == "yes":
        parts_c.append("with internet")
    templates.append(" ".join(parts_c) + ".")

    return random.choice(templates)


def build_restaurant_signal(slots):
    templates = []

    parts_a = []
    if slots.get("pricerange"):
        parts_a.append(slots["pricerange"])
    if slots.get("food"):
        parts_a.append(slots["food"])
    parts_a.append("restaurant")
    if slots.get("area"):
        area_word = "centre" if slots["area"] == "center" else slots["area"]
        parts_a.append(f"in the {area_word}")
    if slots.get("booktime"):
        parts_a.append(f"for {slots['booktime']}")
    templates.append("I would like a " + " ".join(parts_a) + ".")

    parts_b = ["I need somewhere to eat"]
    if slots.get("area"):
        area_word = "centre" if slots["area"] == "center" else slots["area"]
        parts_b.append(f"in the {area_word}")
    if slots.get("food"):
        parts_b.append(f"serving {slots['food']} food")
    if slots.get("pricerange"):
        parts_b.append(f"that is {slots['pricerange']}")
    if slots.get("booktime"):
        parts_b.append(f"at {slots['booktime']}")
    templates.append(" ".join(parts_b) + ".")

    parts_c = ["The user is looking for a restaurant"]
    if slots.get("food"):
        parts_c.append(f"with {slots['food']} food")
    if slots.get("pricerange"):
        parts_c.append(f"in the {slots['pricerange']} price range")
    if slots.get("area"):
        area_word = "centre" if slots["area"] == "center" else slots["area"]
        parts_c.append(f"in the {area_word}")
    if slots.get("booktime"):
        parts_c.append(f"for {slots['booktime']}")
    templates.append(" ".join(parts_c) + ".")

    return random.choice(templates)


def build_train_signal(slots):
    templates = []

    parts_a = ["I need a train"]
    if slots.get("departure"):
        parts_a.append(f"from {slots['departure']}")
    if slots.get("destination"):
        parts_a.append(f"to {slots['destination']}")
    if slots.get("day"):
        parts_a.append(f"on {slots['day']}")
    if slots.get("leaveat"):
        parts_a.append(f"leaving after {slots['leaveat']}")
    if slots.get("arriveby"):
        parts_a.append(f"arriving by {slots['arriveby']}")
    templates.append(" ".join(parts_a) + ".")

    parts_b = ["I am looking for a train"]
    if slots.get("day"):
        parts_b.append(f"on {slots['day']}")
    if slots.get("departure") and slots.get("destination"):
        parts_b.append(f"from {slots['departure']} to {slots['destination']}")
    if slots.get("arriveby"):
        parts_b.append(f"that arrives by {slots['arriveby']}")
    if slots.get("leaveat"):
        parts_b.append(f"and leaves after {slots['leaveat']}")
    templates.append(" ".join(parts_b) + ".")

    parts_c = ["The user needs rail travel"]
    if slots.get("departure"):
        parts_c.append(f"from {slots['departure']}")
    if slots.get("destination"):
        parts_c.append(f"to {slots['destination']}")
    if slots.get("leaveat"):
        parts_c.append(f"after {slots['leaveat']}")
    if slots.get("arriveby"):
        parts_c.append(f"with arrival by {slots['arriveby']}")
    templates.append(" ".join(parts_c) + ".")

    return random.choice(templates)


def extract_signal(raw_turns, main_task):
    if main_task == "hotel":
        slots = extract_hotel_constraints(raw_turns)
        if slots is None:
            return None, None
        return build_hotel_signal(slots), slots

    if main_task == "restaurant":
        slots = extract_restaurant_constraints(raw_turns)
        if slots is None:
            return None, None
        return build_restaurant_signal(slots), slots

    if main_task == "train":
        slots = extract_train_constraints(raw_turns)
        if slots is None:
            return None, None
        return build_train_signal(slots), slots

    return None, None


def sample_intro_turns():
    return [dict(x) for x in random.choice(INTRO_BANK)]


def sample_noise_turn(existing_texts):
    tries = 0
    while tries < 50:
        tries += 1
        speaker, text = random.choice(NOISE_BANK)
        if not too_similar(text, existing_texts):
            return {"speaker": speaker, "text": text}
    speaker, text = random.choice(NOISE_BANK)
    return {"speaker": speaker, "text": text}


def build_distractor_block(main_task, slots):
    if main_task == "hotel":
        if slots.get("area") in {"west", "east", "north", "south", "centre", "center"}:
            alt_area = "centre" if slots["area"] != "centre" else "west"
            block = [
                {"speaker": "A", "text": f"Actually, staying in the {alt_area} might also be okay."},
                {"speaker": "B", "text": "Okay, noted."},
            ]
            return block, {"type": "conflicting_preference", "value": f"area softened to {alt_area}"}

        if slots.get("parking") == "yes":
            block = [
                {"speaker": "A", "text": "I guess parking may not matter as much as I first thought."},
                {"speaker": "B", "text": "All right, I will keep that in mind."},
            ]
            return block, {"type": "conflicting_preference", "value": "parking may not matter"}

        return random.choice(HOTEL_DISTRACTOR_BANK), {
            "type": "conflicting_preference",
            "value": "softened hotel preference"
        }

    if main_task == "restaurant":
        if slots.get("area"):
            alt_area = "centre" if slots["area"] != "centre" else "west"
            block = [
                {"speaker": "A", "text": f"Actually, somewhere in the {alt_area} might also be acceptable."},
                {"speaker": "B", "text": "All right, noted."},
            ]
            return block, {"type": "conflicting_preference", "value": f"area softened to {alt_area}"}

        if slots.get("pricerange") == "cheap":
            block = [
                {"speaker": "A", "text": "I suppose a slightly more expensive place could also work."},
                {"speaker": "B", "text": "Okay, I will keep that in mind."},
            ]
            return block, {"type": "conflicting_preference", "value": "price softened"}

        return random.choice(RESTAURANT_DISTRACTOR_BANK), {
            "type": "conflicting_preference",
            "value": "softened restaurant preference"
        }

    if main_task == "train":
        if slots.get("leaveat"):
            block = [
                {"speaker": "A", "text": "Actually, I might be okay with a later train."},
                {"speaker": "B", "text": "Okay, I will keep that in mind."},
            ]
            return block, {"type": "conflicting_preference", "value": "later departure might be okay"}

        if slots.get("arriveby"):
            block = [
                {"speaker": "A", "text": "Maybe arriving a little later would still be acceptable."},
                {"speaker": "B", "text": "I understand."},
            ]
            return block, {"type": "conflicting_preference", "value": "later arrival might be okay"}

        return random.choice(TRAIN_DISTRACTOR_BANK), {
            "type": "conflicting_preference",
            "value": "softened train preference"
        }

    return [
        {"speaker": "A", "text": "Actually, another option might also work."},
        {"speaker": "B", "text": "Okay, noted."},
    ], {"type": "conflicting_preference", "value": "generic softened preference"}


def build_noise_and_distractor_block(main_task, slots, target_turns=45):
    turns = []
    existing = []

    noise_prefix_target = random.randint(33, 37)
    while len(turns) < noise_prefix_target:
        t = sample_noise_turn(existing)
        t["role"] = "noise"
        turns.append(t)
        existing.append(t["text"])

    distractor_turns, distractor_obj = build_distractor_block(main_task, slots)
    for t in distractor_turns:
        if not too_similar(t["text"], existing):
            tt = dict(t)
            tt["role"] = "distractor"
            turns.append(tt)
            existing.append(tt["text"])

    while len(turns) < target_turns:
        t = sample_noise_turn(existing)
        t["role"] = "noise"
        turns.append(t)
        existing.append(t["text"])

    return turns[:target_turns], distractor_obj


def flip_area(area):
    choices = ["north", "south", "east", "west", "centre"]
    if area in choices:
        alts = [x for x in choices if x != area]
        return random.choice(alts)
    return "centre"


def flip_price(price):
    choices = ["cheap", "moderate", "expensive"]
    if price in choices:
        alts = [x for x in choices if x != price]
        return random.choice(alts)
    return "moderate"


def flip_stars(stars):
    choices = ["1", "2", "3", "4", "5"]
    if stars in choices:
        alts = [x for x in choices if x != stars]
        return random.choice(alts)
    return "3"


def shift_time_str(time_str, delta_minutes=30):
    try:
        h, m = map(int, time_str.split(":"))
        total = h * 60 + m + delta_minutes
        total = max(0, min(total, 23 * 60 + 59))
        return f"{total // 60:02d}:{total % 60:02d}"
    except Exception:
        return time_str


def hotel_option_text(slots):
    parts = []
    if slots.get("pricerange"):
        parts.append(slots["pricerange"])
    if slots.get("stars"):
        parts.append(f"{slots['stars']} star")
    if slots.get("type"):
        parts.append(slots["type"])
    else:
        parts.append("hotel")
    if slots.get("area"):
        parts.append(f"in the {slots['area']}")
    if slots.get("parking") == "yes":
        parts.append("with free parking")
    elif slots.get("parking") == "no":
        parts.append("without parking")
    if slots.get("internet") == "yes":
        parts.append("with internet")
    return "The " + " ".join(parts) + "."


def restaurant_option_text(slots):
    parts = []
    if slots.get("pricerange"):
        parts.append(slots["pricerange"])
    if slots.get("food"):
        parts.append(slots["food"])
    parts.append("restaurant")
    if slots.get("area"):
        parts.append(f"in the {slots['area']}")
    if slots.get("booktime"):
        parts.append(f"for {slots['booktime']}")
    return "The " + " ".join(parts) + "."


def train_option_text(slots):
    parts = ["The train"]
    if slots.get("departure"):
        parts.append(f"from {slots['departure']}")
    if slots.get("destination"):
        parts.append(f"to {slots['destination']}")
    if slots.get("day"):
        parts.append(f"on {slots['day']}")
    if slots.get("leaveat"):
        parts.append(f"leaving after {slots['leaveat']}")
    if slots.get("arriveby"):
        parts.append(f"arriving by {slots['arriveby']}")
    return " ".join(parts) + "."


def build_hotel_candidates(slots):
    correct_slots = dict(slots)

    wrong_area_slots = dict(slots)
    if wrong_area_slots.get("area"):
        wrong_area_slots["area"] = flip_area(wrong_area_slots["area"])

    wrong_stars_slots = dict(slots)
    if wrong_stars_slots.get("stars"):
        wrong_stars_slots["stars"] = flip_stars(wrong_stars_slots["stars"])
    elif wrong_stars_slots.get("parking") == "yes":
        wrong_stars_slots["parking"] = "no"

    wrong_price_slots = dict(slots)
    if wrong_price_slots.get("pricerange"):
        wrong_price_slots["pricerange"] = flip_price(wrong_price_slots["pricerange"])
    elif wrong_price_slots.get("area"):
        wrong_price_slots["area"] = flip_area(wrong_price_slots["area"])

    candidates = [
        hotel_option_text(correct_slots),
        hotel_option_text(wrong_area_slots),
        hotel_option_text(wrong_stars_slots),
        hotel_option_text(wrong_price_slots),
    ]
    return candidates, 0


def build_restaurant_candidates(slots):
    correct_slots = dict(slots)

    wrong_area_slots = dict(slots)
    if wrong_area_slots.get("area"):
        wrong_area_slots["area"] = flip_area(wrong_area_slots["area"])

    wrong_food_slots = dict(slots)
    food_choices = ["chinese", "indian", "italian", "british", "thai", "vegetarian"]
    if wrong_food_slots.get("food") in food_choices:
        wrong_food_slots["food"] = random.choice([x for x in food_choices if x != wrong_food_slots["food"]])
    elif wrong_food_slots.get("pricerange"):
        wrong_food_slots["pricerange"] = flip_price(wrong_food_slots["pricerange"])

    wrong_price_slots = dict(slots)
    if wrong_price_slots.get("pricerange"):
        wrong_price_slots["pricerange"] = flip_price(wrong_price_slots["pricerange"])
    elif wrong_price_slots.get("area"):
        wrong_price_slots["area"] = flip_area(wrong_price_slots["area"])

    candidates = [
        restaurant_option_text(correct_slots),
        restaurant_option_text(wrong_area_slots),
        restaurant_option_text(wrong_food_slots),
        restaurant_option_text(wrong_price_slots),
    ]
    return candidates, 0


def build_train_candidates(slots):
    correct_slots = dict(slots)

    wrong_time_slots = dict(slots)
    if wrong_time_slots.get("leaveat"):
        wrong_time_slots["leaveat"] = shift_time_str(wrong_time_slots["leaveat"], 60)
    elif wrong_time_slots.get("arriveby"):
        wrong_time_slots["arriveby"] = shift_time_str(wrong_time_slots["arriveby"], 60)

    wrong_route_slots = dict(slots)
    if wrong_route_slots.get("destination"):
        wrong_route_slots["destination"] = "london" if wrong_route_slots["destination"] != "london" else "cambridge"
    elif wrong_route_slots.get("departure"):
        wrong_route_slots["departure"] = "ely" if wrong_route_slots["departure"] != "ely" else "cambridge"

    wrong_day_slots = dict(slots)
    day_choices = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if wrong_day_slots.get("day") in day_choices:
        wrong_day_slots["day"] = random.choice([x for x in day_choices if x != wrong_day_slots["day"]])
    elif wrong_day_slots.get("leaveat"):
        wrong_day_slots["leaveat"] = shift_time_str(wrong_day_slots["leaveat"], -60)

    candidates = [
        train_option_text(correct_slots),
        train_option_text(wrong_time_slots),
        train_option_text(wrong_route_slots),
        train_option_text(wrong_day_slots),
    ]
    return candidates, 0


def deduplicate_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_query_and_candidates(main_task, slots):
    if main_task == "hotel":
        query = random.choice(HOTEL_QUERY_BANK)
        candidates, correct_idx = build_hotel_candidates(slots)
    elif main_task == "restaurant":
        query = random.choice(RESTAURANT_QUERY_BANK)
        candidates, correct_idx = build_restaurant_candidates(slots)
    elif main_task == "train":
        query = random.choice(TRAIN_QUERY_BANK)
        candidates, correct_idx = build_train_candidates(slots)
    else:
        return None, None, None

    candidates = deduplicate_keep_order(candidates)

    # if collisions happened, add a generic negative
    generic_negatives = {
        "hotel": [
            "A train to the station.",
            "A cheap restaurant in the centre.",
            "A taxi to the museum.",
        ],
        "restaurant": [
            "A guesthouse with parking.",
            "A train leaving later in the evening.",
            "A taxi to the station.",
        ],
        "train": [
            "A cheap restaurant in the centre.",
            "A guesthouse with free parking.",
            "A museum near the centre.",
        ],
    }

    while len(candidates) < 4:
        neg = random.choice(generic_negatives[main_task])
        if neg not in candidates:
            candidates.append(neg)

    correct_text = candidates[correct_idx]
    random.shuffle(candidates)
    correct_index = candidates.index(correct_text)

    return query, candidates, correct_index


def convert_dialogue(dialog):
    global sample_counter

    raw_turns = extract_raw_turns(dialog)
    num_raw_turns = len(raw_turns)

    if num_raw_turns < 4:
        return None, None

    main_task = detect_main_task(raw_turns)
    if main_task not in ALLOWED_DOMAINS:
        return None, None

    signal_text, slots = extract_signal(raw_turns, main_task)
    if signal_text is None or slots is None:
        return None, None

    sample_id = f"ex_{sample_counter:06d}"
    sample_counter += 1

    intro_turns = sample_intro_turns()
    for t in intro_turns:
        t["role"] = "intro"

    signal_turn = {"speaker": "A", "text": signal_text, "role": "signal"}

    noise_distractor_turns, distractor_obj = build_noise_and_distractor_block(
        main_task, slots, target_turns=45
    )

    query_text, candidates, correct_index = build_query_and_candidates(main_task, slots)
    if query_text is None:
        return None, None

    query_turn = {"speaker": "A", "text": query_text, "role": "query"}

    final_turns = intro_turns + [signal_turn] + noise_distractor_turns + [query_turn]

    signal_position = 4
    distractor_position = None
    for i, t in enumerate(final_turns):
        if t["role"] == "distractor":
            distractor_position = i
            break

    signal_obj = {
        "type": f"{main_task}_constraint",
        "value": signal_text,
        "position": signal_position,
        "slots": slots,
    }

    distractor_obj["position"] = distractor_position

    metadata = {
        "num_turns": len(final_turns),
        "signal_distance": len(final_turns) - 1 - signal_position,
        "num_noise_turns": sum(1 for t in final_turns if t["role"] == "noise"),
        "has_distractor": distractor_position is not None,
        "difficulty": "medium",
        "raw_num_turns": num_raw_turns,
        "main_task": main_task,
    }

    dialogue_text_for_token = "\n".join(f"[{t['speaker']}] {t['text']}" for t in final_turns)
    metadata["num_tokens"] = len(enc.encode(dialogue_text_for_token))
    metadata["turn_range"] = get_turn_range(len(final_turns))

    result = {
        "sample_id": sample_id,
        "dialogue": final_turns,
        "signal": signal_obj,
        "distractor": distractor_obj,
        "query": query_text,
        "candidates": candidates,
        "correct_index": correct_index,
        "metadata": metadata
    }

    return result, metadata


for split in SPLITS:
    print(f"\n===== Processing {split.upper()} =====")

    DATA_DIR = os.path.join(BASE_DIR, split)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, split)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_count = 0
    split_counter = Counter()

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(DATA_DIR, filename)
        output_path = os.path.join(
            OUTPUT_DIR, filename.replace(".json", ".jsonl")
        )

        print(f"Processing {split}/{filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            dialogues_iter = data.values()
        else:
            dialogues_iter = data

        count = 0
        sample_printed = False

        with open(output_path, "w", encoding="utf-8") as out:
            for dialog in dialogues_iter:
                result, meta = convert_dialogue(dialog)
                if result is None:
                    continue

                out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

                count += 1
                total_count += 1
                split_counter[meta["main_task"]] += 1
                task_counter[meta["main_task"]] += 1

                if not sample_printed:
                    print(f"\nSample ID: {result['sample_id']}")
                    print(f"Turns: {meta['num_turns']}")
                    print(f"Raw turns: {meta['raw_num_turns']}")
                    print(f"Main task: {meta['main_task']}")
                    print(f"Turn range: {meta['turn_range']}")
                    print(f"Tokens: {meta['num_tokens']}")
                    print("Signal:", result["signal"])
                    print("Distractor:", result["distractor"])
                    print("Query:", result["query"])
                    print("Candidates:", result["candidates"])
                    print("Correct index:", result["correct_index"])
                    print("==============\n")
                    sample_printed = True

        print(f"Saved {count} → {output_path}")

    print(f"TOTAL {split}: {total_count}")
    print(f"{split} task distribution: {dict(split_counter)}")

print("\n===== OVERALL TASK DISTRIBUTION =====")
print(dict(task_counter))