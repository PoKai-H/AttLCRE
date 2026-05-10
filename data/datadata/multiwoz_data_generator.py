import json
import os
import random
import re
import uuid
from copy import deepcopy

BASE_DIR = "/Users/reeseliu/Desktop/Linear attention/multiwoz/data/MultiWOZ_2.2"
SPLITS = ["train", "dev", "test"]
OUTPUT_BASE = "multiwoz_generator"
ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}
random.seed(42)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# DOMAIN_SLOT_SCHEMA = {
#     "restaurant": {
#         "categorical": ["pricerange", "area", "bookday", "bookpeople"],
#         "non_categorical": ["food", "name", "booktime", "address", "phone", "postcode", "ref"],
#     },
#     "hotel": {
#         "categorical": ["pricerange", "parking", "internet", "stars", "area", "type", "bookpeople", "bookday", "bookstay"],
#         "non_categorical": ["name", "address", "phone", "postcode", "ref"],
#     },
#     "train": {
#         "categorical": ["destination", "departure", "day", "bookpeople"],
#         "non_categorical": ["arriveby", "leaveat", "trainid", "ref", "price", "duration"],
#     },
# }

WRONG_VALUE_BANK = {
    "restaurant": {
        "pricerange": ["cheap", "moderate", "expensive"],
        "area": ["centre", "north", "south", "east", "west"],
        "food": ["italian", "chinese", "indian", "british", "french"],
        "bookday": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        "bookpeople": ["1", "2", "3", "4", "5", "6"],
    },
    "hotel": {
        "pricerange": ["cheap", "moderate", "expensive"],
        "parking": ["yes", "no"],
        "internet": ["yes", "no"],
        "stars": ["1", "2", "3", "4", "5"],
        "area": ["centre", "north", "south", "east", "west"],
        "type": ["hotel", "guesthouse"],
        "bookpeople": ["1", "2", "3", "4", "5", "6"],
        "bookday": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        "bookstay": ["1", "2", "3", "4", "5", "6"],
    },
    "train": {
        "destination": ["cambridge", "london", "ely", "norwich", "peterborough"],
        "departure": ["cambridge", "london", "ely", "norwich", "peterborough"],
        "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        "bookpeople": ["1", "2", "3", "4", "5", "6"],
        "arriveby": ["09:00", "12:00", "15:00", "18:00", "20:00"],
        "leaveat": ["08:00", "10:00", "13:00", "16:00", "19:00"],
    },
}

QUERYABLE_SLOTS = {
    "restaurant": {"pricerange", "area", "food", "bookday", "bookpeople", "booktime"},
    "hotel": {"pricerange", "parking", "internet", "stars", "area", "type", "bookpeople", "bookday", "bookstay"},
    "train": {"destination", "departure", "day", "bookpeople", "arriveby", "leaveat"},
}

QUERY_BANK = {
    "restaurant": [
        "Which option matches what I mentioned earlier?",
        "Which restaurant best fits my needs?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which restaurant is best?",
    ],
    "hotel": [
        "Which hotel should I choose?",
        "Which hotel best fits my needs?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which hotel is best?",
    ],
    "train": [
        "Which train should I choose?",
        "Which train best fits my needs?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which train is best?",
    ],
}

def get_wrong_values_for_slot(domain: str, slot: str, true_value: str) -> list[str]:
    candidates = WRONG_VALUE_BANK.get(domain, {}).get(slot, [])
    true_value = normalize_value(true_value)

    wrong_values = []
    for v in candidates:
        nv = normalize_value(v)
        if nv is not None and nv != true_value:
            wrong_values.append(nv)

    return wrong_values

def sample_noise_from_dialog(dialog, k=3):
    turns = dialog.get("turns", [])
    blocks = []

    for turn in turns:
        if not isinstance(turn, dict):
            continue

        text = str(turn.get("utterance", "")).strip()
        if not text:
            continue

        speaker_raw = str(turn.get("speaker", "")).upper()
        speaker = "A" if speaker_raw == "USER" else "B"

        frames = turn.get("frames", [])
        has_state = False

        if isinstance(frames, list):
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                state = fr.get("state", {})
                if isinstance(state, dict) and state.get("slot_values"):
                    has_state = True
                    break

        if has_state:
            continue

        blocks.append([
            {
                "speaker": speaker,
                "text": text,
                "role": "noise"
            }
        ])

    random.shuffle(blocks)
    return blocks[:k]

#determines the domain of a dialogue.
def get_domain_from_dialog(dialog):
    services = dialog.get("services", [])
    found = [str(s).lower() for s in services if str(s).lower() in ALLOWED_DOMAINS]

    if found:
        return random.choice(found) #If valid domains are found, randomly return one.

    turns = dialog.get("turns", [])
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        frames = turn.get("frames", [])
        for fr in frames:
            service = str(fr.get("service", "")).lower()
            if service in ALLOWED_DOMAINS:
                return service

    # dialogue_id = str(dialog.get("dialogue_id", "")).lower()
    # for d in ALLOWED_DOMAINS:
    #     if d in dialogue_id:
    #         return d

    return None

#normalizes a slot value
def normalize_value(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"", "not mentioned", "none"}:
        return None
    return v

def extract_gold_state(dialog, domain):
    gold_state = {}
    allowed_slots = QUERYABLE_SLOTS[domain]

    turns = dialog.get("turns", [])
    for turn in turns:
        if not isinstance(turn, dict):
            continue

        frames = turn.get("frames", [])
        if not isinstance(frames, list):
            continue

        for fr in frames:
            service = str(fr.get("service", "")).lower()
            if service != domain:
                continue

            state = fr.get("state", {})
            if not isinstance(state, dict):
                continue

            slot_values = state.get("slot_values", {})
            if not isinstance(slot_values, dict):
                continue

            for slot, values in slot_values.items():
                short_slot = slot.split("-")[-1].lower()

                if short_slot not in allowed_slots:
                    continue

                if isinstance(values, list) and values:
                    value = normalize_value(values[0])
                else:
                    value = normalize_value(values)

                if value is not None:
                    gold_state[short_slot] = value

    return gold_state

def arrange_blocks_with_random_distance(
    signal_blocks,
    noise_blocks,
    distractor_blocks,
    query_context_turn,
    query_turn,
    min_distance=3,
    max_distance=10
):
    dialogue_blocks = []

    chosen_signal = random.choice(signal_blocks)
    remaining_signals = [b for b in signal_blocks if b != chosen_signal]

    target_distance = random.randint(min_distance, max_distance)

    filler_pool = noise_blocks + distractor_blocks + remaining_signals
    random.shuffle(filler_pool)

    actual_fillers = filler_pool[:target_distance]

    prefix_pool = filler_pool[target_distance:]
    random.shuffle(prefix_pool)
    prefix = prefix_pool[:random.randint(0, 3)]

    dialogue_blocks.extend(prefix)
    dialogue_blocks.append(chosen_signal)
    dialogue_blocks.extend(actual_fillers)
    dialogue_blocks.append([query_context_turn])
    dialogue_blocks.append([query_turn])

    return dialogue_blocks

def generate_sample(
    dialog: dict,
    sample_id: str,
    num_distractors: int = 3,
    difficulty: str | None = None,
    debug: bool = False,
):
    domain = get_domain_from_dialog(dialog)
    if domain not in ALLOWED_DOMAINS:
        if debug:
            print(f"[skip:{sample_id}] invalid domain -> {domain}")
        return None

    gold_state = extract_gold_state(dialog, domain)
    if not gold_state:
        if debug:
            print(f"[skip:{sample_id}] empty gold_state, domain={domain}")
            print("dialogue_id =", dialog.get("dialogue_id"))
            print("top keys =", list(dialog.keys()))
            turns = dialog.get("turns", [])
            if turns:
                print("frames sample =", turns[0].get("frames", []))
        return None

    full_values = deepcopy(gold_state)

    candidate_slots = list(gold_state.keys())
    if not candidate_slots:
        return None

    if difficulty == "easy":
        n_true = min(len(candidate_slots), random.choice([1, 2]))
    elif difficulty == "medium":
        n_true = min(len(candidate_slots), random.choice([2, 2, 3]))
    else:  # hard
        n_true = min(len(candidate_slots), random.choice([2, 3, 3]))

    chosen_slots = random.sample(candidate_slots, k=n_true)
    true_slots = {slot: gold_state[slot] for slot in chosen_slots}

    wrong_slots = {}
    for slot, value in true_slots.items():
        wrong_values = get_wrong_values_for_slot(domain, slot, value)
        if wrong_values:
            wrong_slots[slot] = wrong_values

    if not wrong_slots:
        if debug:
            print(f"[skip:{sample_id}] no wrong_slots")
        return None

    if difficulty is None:
        if num_distractors <= 1:
            difficulty = "easy"
        elif num_distractors <= 3:
            difficulty = "medium"
        else:
            difficulty = "hard"

    if difficulty == "easy":
        target_num_distractors = min(num_distractors, 1)
        noise_blocks = sample_noise_from_dialog(dialog, k=1)
        candidate_max_changed_slots = max(1, min(2, len(true_slots)))
    elif difficulty == "medium":
        target_num_distractors = max(2, num_distractors)
        noise_blocks = sample_noise_from_dialog(dialog, k=3)
        candidate_max_changed_slots = max(1, min(2, len(true_slots)))
    else:  # hard
        target_num_distractors = max(4, num_distractors)
        noise_blocks = sample_noise_from_dialog(dialog, k=6)
        candidate_max_changed_slots = min(3, len(true_slots))

    signal_blocks = generate_signal_pairs(domain, true_slots, difficulty)

    distractor_blocks = generate_distractor_pairs(
        domain=domain,
        full_values=full_values,
        wrong_slots=wrong_slots,
        true_slots=true_slots,
        num_distractors=target_num_distractors,
        difficulty=difficulty,
    )

    query_text = generate_query(domain)

    query_context_turn = {
        "speaker": "B",
        "text": "Let me compare the options based on what you mentioned earlier.",
        "role": "query_context",
    }

    query_turn = {
        "speaker": "A",
        "text": query_text,
        "role": "query",
    }

    if difficulty == "easy":
        dialogue_blocks = arrange_blocks_with_random_distance(
            signal_blocks,
            noise_blocks,
            distractor_blocks,
            query_context_turn,
            query_turn,
            min_distance=1,
            max_distance=3
        )

    elif difficulty == "medium":
        dialogue_blocks = arrange_blocks_with_random_distance(
            signal_blocks,
            noise_blocks,
            distractor_blocks,
            query_context_turn,
            query_turn,
            min_distance=3,
            max_distance=6
        )

    else:  # hard
        dialogue_blocks = arrange_blocks_with_random_distance(
            signal_blocks,
            noise_blocks,
            distractor_blocks,
            query_context_turn,
            query_turn,
            min_distance=6,
            max_distance=12
        )

    final_dialogue = []
    for block in dialogue_blocks:
        final_dialogue.extend(block)

    candidate_texts, correct_index = build_candidates(
        domain=domain,
        full_values=full_values,
        true_slots=true_slots,
        wrong_slots=wrong_slots,
        num_candidates=4,
        max_changed_slots=candidate_max_changed_slots,
        difficulty=difficulty,
    )
    if not candidate_texts or correct_index < 0:
        if debug:
            print(f"[skip:{sample_id}] failed to build unique candidates")
        return None

    signal_positions = [i for i, turn in enumerate(final_dialogue) if turn.get("role") == "signal"]
    query_positions = [i for i, turn in enumerate(final_dialogue) if turn.get("role") == "query"]

    if signal_positions and query_positions:
        signal_query_distance = query_positions[0] - max(signal_positions)
    else:
        signal_query_distance = None

    sample = {
        "sample_id": sample_id,
        "dialogue_id": dialog.get("dialogue_id"),
        "domain": domain,
        "difficulty": difficulty,
        "has_distractor": len(distractor_blocks) > 0,
        "num_distractors": len(distractor_blocks),
        "signal_query_distance": signal_query_distance,
        "true_slots": true_slots,
        "candidates": candidate_texts,
        "correct_index": correct_index,
        "dialogue": final_dialogue,
    }

    return sample

def generate_query(domain: str) -> str:
    if domain in QUERY_BANK:
        return random.choice(QUERY_BANK[domain])

    return random.choice([
        "Which option should I choose?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which option is best?",
    ])

#turn each important constraint in true_slots into a dialogue block.
def generate_signal_pairs(
    domain: str,
    true_slots: dict[str, str],
    difficulty: str = "medium",
) -> list[list[dict]]:

    signals = []

    for slot, value in true_slots.items():
        signals.append(render_signal_pair(domain, slot, value, difficulty))

    return signals

def render_signal_pair(
    domain: str,
    slot: str,
    value: str,
    difficulty: str = "medium",
) -> list[dict]:
    level = sample_reply_level(difficulty)
    return [
        {
            "speaker": "A",
            "text": render_slot_signal(domain, slot, value),
            "role": "signal",
            "slot": slot,
            "value": value,
        },
        {
            "speaker": "B",
            "text": render_reply(domain, slot, value, level),
            "role": "signal",
            "slot": slot,
            "value": value,
        },
    ]

#return the full list of signal blocks
def render_slot_signal(domain: str, slot: str, value: str) -> str:

    templates = {
        "restaurant": {
            "pricerange": [
                f"I want a {value} restaurant.",
                f"I am looking for something {value}.",
            ],
            "area": [
                f"I want a restaurant in the {value}.",
                f"The {value} area would be best for me.",
            ],
            "food": [
                f"I would like {value} food.",
                f"I am looking for a {value} restaurant.",
            ],
            "bookday": [
                f"I need the reservation on {value}.",
                f"{value.capitalize()} would be the day I want.",
            ],
            "bookpeople": [
                f"The booking should be for {value} people.",
                f"I need a table for {value}.",
            ],
            "booktime": [
                f"I want the reservation at {value}.",
                f"{value} would be the ideal booking time.",
            ],
        },
        "hotel": {
            "pricerange": [
                f"I need a {value} hotel.",
                f"I am looking for a {value} place to stay.",
            ],
            "area": [
                f"I want a hotel in the {value}.",
                f"The {value} area would be best.",
            ],
            "parking": [
                "The hotel must have parking." if value == "yes" else "Parking is not necessary.",
            ],
            "internet": [
                "I need free Wi-Fi." if value == "yes" else "Internet is not essential.",
            ],
            "stars": [
                f"I want a {value}-star hotel.",
                f"A {value}-star place would be ideal.",
            ],
            "type": [
                f"I would prefer a {value}.",
                f"I am looking for a {value}.",
            ],
            "bookpeople": [
                f"It is for {value} people.",
                f"The room should be for {value} guests.",
            ],
            "bookday": [
                f"I want to check in on {value}.",
                f"{value.capitalize()} is my preferred check-in day.",
            ],
            "bookstay": [
                f"I will stay for {value} nights.",
                f"I need the hotel for {value} nights.",
            ],
        },
        "train": {
            "departure": [
                f"I need to leave from {value}.",
                f"My departure station should be {value}.",
            ],
            "destination": [
                f"I need to go to {value}.",
                f"My destination is {value}.",
            ],
            "day": [
                f"I need the train on {value}.",
                f"{value.capitalize()} is the day I want to travel.",
            ],
            "bookpeople": [
                f"I need tickets for {value} people.",
                f"The booking should be for {value} passengers.",
            ],
            "arriveby": [
                f"I need to arrive by {value}.",
                f"The train should arrive before {value}.",
            ],
            "leaveat": [
                f"I need to leave at {value}.",
                f"I want a train departing at {value}.",
            ],
        },
    }

    slot_templates = templates.get(domain, {}).get(slot)
    if slot_templates:
        return random.choice(slot_templates)

    return f"My preference is {slot} = {value}."

def render_reply(domain: str, slot: str, value: str, level: int) -> str:

    """
    level 0: generic reply
    level 1: weak confirmation
    level 2: strong confirmation with slot-specific wording
    """

    generic = [
        "Okay.",
        "Got it.",
        "Understood.",
        "Sure.",
    ]

    weak = [
        "Okay, I will keep that in mind.",
        "Understood, I will take that into account.",
        "Sure, I will consider that.",
    ]

    strong_templates = {
        "restaurant": {
            "pricerange": [
                f"Okay, I will look for {value} restaurants.",
                f"Got it, I will keep the price range {value}.",
            ],
            "area": [
                f"Okay, I will focus on the {value} area.",
                f"Got it, I will search in the {value}.",
            ],
            "food": [
                f"Okay, I will look for {value} restaurants.",
                f"Got it, I will keep {value} food in mind.",
            ],
            "bookday": [
                f"Okay, I will use {value} for the booking day.",
            ],
            "bookpeople": [
                f"Got it, I will search for a table for {value}.",
            ],
            "booktime": [
                f"Okay, I will search for availability at {value}.",
            ],
        },
        "hotel": {
            "pricerange": [
                f"Okay, I will look for {value} hotels.",
            ],
            "area": [
                f"Got it, I will search in the {value} area.",
            ],
            "parking": [
                "Okay, I will only consider hotels with parking." if value == "yes"
                else "Okay, parking is not required.",
            ],
            "internet": [
                "Got it, I will look for hotels with internet." if value == "yes"
                else "Okay, internet is optional.",
            ],
            "stars": [
                f"Okay, I will look for {value}-star hotels.",
            ],
            "type": [
                f"Got it, I will focus on {value} options.",
            ],
            "bookpeople": [
                f"Okay, I will search for rooms for {value} guests.",
            ],
            "bookday": [
                f"Got it, I will use {value} as the check-in day.",
            ],
            "bookstay": [
                f"Okay, I will search for a {value}-night stay.",
            ],
        },
        "train": {
            "departure": [
                f"Okay, I will use {value} as the departure point.",
            ],
            "destination": [
                f"Got it, I will look for trains to {value}.",
            ],
            "day": [
                f"Okay, I will search for trains on {value}.",
            ],
            "bookpeople": [
                f"Got it, I will search for {value} tickets.",
            ],
            "arriveby": [
                f"Okay, I will only consider trains arriving by {value}.",
            ],
            "leaveat": [
                f"Got it, I will search for departures at {value}.",
            ],
        },
    }

    if level == 0:
        return random.choice(generic)
    elif level == 1:
        return random.choice(weak)
    else:
        slot_templates = strong_templates.get(domain, {}).get(slot)
        if slot_templates:
            return random.choice(slot_templates)
        return random.choice(weak)

def build_distractor_pool(
    domain: str,
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
) -> list[dict]:
    pool = []

    # hard negatives: relevant slot + wrong value
    for slot, wrong_values in wrong_slots.items():
        for value in wrong_values:
            pool.append({
                "distractor_type": "hard_negative",
                "slot": slot,
                "value": value,
            })

    # irrelevant: non-query slot + true value
    for slot, value in full_values.items():
        if slot not in true_slots:
            pool.append({
                "distractor_type": "irrelevant",
                "slot": slot,
                "value": value,
            })

    return pool

def sample_reply_level(difficulty: str = "medium") -> int:
    if difficulty == "easy":
        return random.choice([1, 2, 2])
    elif difficulty == "medium":
        return random.choice([0, 1, 1])
    else:  # hard
        return random.choice([0, 0, 1])

#Generate distractor dialogue pairs
def generate_distractor_pairs(
    domain: str,
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
    num_distractors: int,
    difficulty: str = "medium",
) -> list[list[dict]]:
    pool = build_distractor_pool(domain, full_values, wrong_slots, true_slots)

    random.shuffle(pool)
    selected = pool[:min(num_distractors, len(pool))]

    blocks = []
    for item in selected:
        level = sample_reply_level(difficulty)
        slot = item["slot"]
        value = item["value"]
        dtype = item["distractor_type"]

        blocks.append([
            {
                "speaker": "A",
                "text": render_slot_signal(domain, slot, value),
                "role": "distractor",
                "slot": slot,
                "value": value,
                "distractor_type": dtype,
            },
            {
                "speaker": "B",
                "text": render_reply(domain, slot, value, level),
                "role": "distractor",
                "slot": slot,
                "value": value,
                "distractor_type": dtype,
            },
        ])

    return blocks

def render_candidate(domain: str, option: dict[str, str]) -> str:
    intro_templates = [
        "The best match is",
        "A suitable option is",
        "You should choose",
        "The most suitable choice is",
    ]

    intro = random.choice(intro_templates)

    if domain == "restaurant":
        parts = []
        if "pricerange" in option:
            parts.append(f"a {option['pricerange']} restaurant")
        if "food" in option:
            parts.append(f"serving {option['food']} food")
        if "area" in option:
            parts.append(f"in the {option['area']}")
        if "bookday" in option:
            parts.append(f"for {option['bookday']}")
        if "booktime" in option:
            parts.append(f"at {option['booktime']}")
        if "bookpeople" in option:
            parts.append(f"for {option['bookpeople']} people")

        desc = ", ".join(parts)
        return f"{intro} {desc}."

    elif domain == "hotel":
        parts = []
        if "pricerange" in option:
            parts.append(f"a {option['pricerange']} hotel")
        if "stars" in option:
            parts.append(f"with {option['stars']} stars")
        if "area" in option:
            parts.append(f"in the {option['area']}")
        if "parking" in option:
            parts.append(f"parking: {option['parking']}")
        if "internet" in option:
            parts.append(f"internet: {option['internet']}")
        if "type" in option:
            parts.append(f"type: {option['type']}")
        if "bookday" in option:
            parts.append(f"check-in on {option['bookday']}")
        if "bookstay" in option:
            parts.append(f"for {option['bookstay']} nights")
        if "bookpeople" in option:
            parts.append(f"for {option['bookpeople']} guests")

        desc = ", ".join(parts)
        return f"{intro} {desc}."

    elif domain == "train":
        parts = []
        if "departure" in option:
            parts.append(f"departing from {option['departure']}")
        if "destination" in option:
            parts.append(f"going to {option['destination']}")
        if "day" in option:
            parts.append(f"on {option['day']}")
        if "leaveat" in option:
            parts.append(f"leaving at {option['leaveat']}")
        if "arriveby" in option:
            parts.append(f"arriving by {option['arriveby']}")
        if "bookpeople" in option:
            parts.append(f"for {option['bookpeople']} passengers")

        desc = ", ".join(parts)
        return f"{intro} a train {desc}."

    else:
        return f"{intro} an option with constraints: {option}."

def build_candidates(
    domain: str,
    full_values: dict[str, str],
    true_slots: dict[str, str],
    wrong_slots: dict[str, list[str]],
    num_candidates: int = 4,
    max_changed_slots: int | None = None,
    difficulty: str = "medium",
) -> tuple[list[str], int]:
    if max_changed_slots is None:
        max_changed_slots = len(true_slots)

    true_option = deepcopy(full_values)

    query_slot_names = [
        slot for slot in true_slots
        if slot in wrong_slots and wrong_slots[slot]
    ]

    if not query_slot_names:
        return [], -1

    if difficulty == "easy":
        min_changed = 1
        max_changed = min(max_changed_slots, len(query_slot_names))
    elif difficulty == "medium":
        min_changed = 1
        max_changed = min(max_changed_slots, len(query_slot_names))
    else:  # hard
        min_changed = 2 if len(query_slot_names) >= 2 else 1
        max_changed = min(max_changed_slots, len(query_slot_names))

    if max_changed < min_changed:
        min_changed = max_changed

    candidate_dicts = [deepcopy(true_option)]
    seen_texts = {render_candidate(domain, true_option)}
    max_attempts = 200
    attempts = 0

    while len(candidate_dicts) < num_candidates and attempts < max_attempts:
        attempts += 1
        cand = deepcopy(true_option)

        num_changed = random.randint(min_changed, max_changed)
        slots_to_change = random.sample(query_slot_names, k=num_changed)

        for slot in slots_to_change:
            new_value = random.choice(wrong_slots[slot])
            cand[slot] = new_value

        text = render_candidate(domain, cand)

        if text in seen_texts:
            continue

        seen_texts.add(text)
        candidate_dicts.append(cand)

    if len(candidate_dicts) < num_candidates:
        return [], -1

    candidate_texts = [render_candidate(domain, c) for c in candidate_dicts]

    indexed = list(enumerate(candidate_texts))
    random.shuffle(indexed)

    shuffled_candidate_texts = [x[1] for x in indexed]
    correct_index = next(i for i, (orig_idx, _) in enumerate(indexed) if orig_idx == 0)

    return shuffled_candidate_texts, correct_index

def load_dialogues_from_split(split_dir):
    all_dialogues = []

    if not os.path.exists(split_dir):
        print(f"[Warning] Split directory not found: {split_dir}")
        return all_dialogues

    for filename in os.listdir(split_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(split_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Skip] Failed to read {path}: {e}")
            continue

        if isinstance(data, dict):
            items = list(data.values())
        elif isinstance(data, list):
            items = data
        else:
            continue

        for x in items:
            if isinstance(x, dict):
                all_dialogues.append(x)

    return all_dialogues

def main():
    for split in SPLITS:
        split_dir = os.path.join(BASE_DIR, split)
        dialogues = load_dialogues_from_split(split_dir)

        print("split_dir =", split_dir)
        print("num dialogues loaded =", len(dialogues))

        data = []
        skip_invalid_domain = 0
        skip_empty_gold_state = 0
        skip_other = 0

        for i, dialog in enumerate(dialogues):
            difficulty = random.choice(["easy", "medium", "hard"])

            if difficulty == "easy":
                num_distractors = 1
            elif difficulty == "medium":
                num_distractors = 3
            else:
                num_distractors = 6

            sample = generate_sample(
                dialog=dialog,
                sample_id=f"ex_{i:06d}",
                num_distractors=num_distractors,
                difficulty=difficulty,
                debug=False
            )

            if sample is not None:
                data.append(sample)
            else:
                skip_other += 1

        split_output_dir = os.path.join(OUTPUT_BASE, split)
        os.makedirs(split_output_dir, exist_ok=True)

        output_path = os.path.join(split_output_dir, f"{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"saved {len(data)} samples to {output_path}")
        print(f"skip_invalid_domain = {skip_invalid_domain}")
        print(f"skip_empty_gold_state = {skip_empty_gold_state}")
        print(f"skip_other = {skip_other}")

if __name__ == "__main__":
    main()