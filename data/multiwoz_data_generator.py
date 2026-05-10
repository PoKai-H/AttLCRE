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


DOMAIN_SLOT_SCHEMA = {
    "restaurant": {
        "categorical": ["pricerange", "area", "bookday", "bookpeople"],
        "non_categorical": ["food", "name", "booktime", "address", "phone", "postcode", "ref"],
    },
    "hotel": {
        "categorical": ["pricerange", "parking", "internet", "stars", "area", "type", "bookpeople", "bookday", "bookstay"],
        "non_categorical": ["name", "address", "phone", "postcode", "ref"],
    },
    "train": {
        "categorical": ["destination", "departure", "day", "bookpeople"],
        "non_categorical": ["arriveby", "leaveat", "trainid", "ref", "price", "duration"],
    },
}

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

def get_domain_from_dialog(dialog):

    found = set()

    services = dialog.get("services", [])
    for s in services:
        s = str(s).lower()
        if s in ALLOWED_DOMAINS:
            found.add(s)

    domains = dialog.get("domains", [])
    for d in domains:
        d = str(d).lower()
        if d in ALLOWED_DOMAINS:
            found.add(d)

    frames = dialog.get("frames", [])
    for fr in frames:
        service = str(fr.get("service", "")).lower()
        if service in ALLOWED_DOMAINS:
            found.add(service)

    dialogue_id = str(dialog.get("dialogue_id", "")).lower()
    for d in ALLOWED_DOMAINS:
        if d in dialogue_id:
            found.add(d)

    if not found:
        return None

    return random.choice(list(found))

def normalize_value(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"", "not mentioned", "none"}:
        return None
    return v

def extract_gold_state(dialog, domain):
    gold_state = {}

    allowed_slots = set(DOMAIN_SLOT_SCHEMA[domain]["categorical"]) | set(
        DOMAIN_SLOT_SCHEMA[domain]["non_categorical"])

    frames = dialog.get("frames", [])
    for fr in frames:
        service = str(fr.get("service", "")).lower()
        if service != domain:
            continue

        state = fr.get("state", {})
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

def generate_sample(dialog: dict, num_distractors=3, sample_id=None):
    if sample_id is None:
        sample_id = f"ex_{uuid.uuid4().hex[:6]}"

    domain = get_domain_from_dialog(dialog)
    if domain not in ALLOWED_DOMAINS:
        return None

    gold_state = extract_gold_state(dialog, domain)
    if not gold_state:
        return None

    candidate_slots = list(gold_state.keys())
    if len(candidate_slots) < 2:
        return None

    k = min(len(candidate_slots), random.choice([2, 3]))
    selected_slot_names = random.sample(candidate_slots, k=k)

    full_values = deepcopy(gold_state)

    true_slots = {
        slot: full_values[slot]
        for slot in selected_slot_names
    }

    wrong_slots = {}
    for slot in selected_slot_names:
        bank = WRONG_VALUE_BANK.get(domain, {}).get(slot, [])
        gold_value = str(true_slots[slot]).lower()

        filtered = [v for v in bank if str(v).lower() != gold_value]
        if filtered:
            wrong_slots[slot] = filtered

    if not wrong_slots:
        return None

    query = generate_query(domain)
    signal_blocks = generate_signal_pairs(domain, true_slots)

    distractor_blocks = generate_distractor_pairs(
        domain=domain,
        full_values=full_values,
        wrong_slots=wrong_slots,
        true_slots=true_slots,
        num_distractors=num_distractors,
    )

    candidates, correct_index = build_candidates(
        domain=domain,
        full_values=full_values,
        true_slots=true_slots,
        wrong_slots=wrong_slots,
        num_candidates=4,
    )

    all_blocks = signal_blocks + distractor_blocks
    random.shuffle(all_blocks)

    dialogue = []
    for block in all_blocks:
        dialogue.extend(block)

    dialogue.append({
        "speaker": "A",
        "text": query,
        "role": "query",
    })

    return {
        "sample_id": sample_id,
        "domain": domain,
        "full_values": full_values,
        "true_slots": true_slots,
        "wrong_slots": wrong_slots,
        "signal_blocks": signal_blocks,
        "distractor_blocks": distractor_blocks,
        "dialogue": dialogue,
        "query": query,
        "candidates": candidates,
        "answer_idx": correct_index,
    }



QUERY_BANK = {
    "restaurant": [
        "Which restaurant should I choose?",
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

def generate_query(domain: str) -> str:
    if domain in QUERY_BANK:
        return random.choice(QUERY_BANK[domain])

    return random.choice([
        "Which option should I choose?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which option is best?",
    ])

def generate_signal_pairs(domain: str, true_slots: dict[str, str]) -> list[list[dict]]:

    signals = []

    for slot, value in true_slots.items():
        signals.append(render_signal_pair(domain, slot, value))

    return signals

def render_signal_pair(domain: str, slot: str, value: str) -> list[dict]:
    level = sample_reply_level()
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

def sample_reply_level() -> int:
    """
    0 = generic
    1 = weak confirmation
    2 = strong confirmation
    """
    return random.choice([0, 1, 2])

def generate_distractor_pairs(
    domain: str,
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
    num_distractors: int,
) -> list[list[dict]]:
    pool = build_distractor_pool(domain, full_values, wrong_slots, true_slots)

    random.shuffle(pool)
    selected = pool[:min(num_distractors, len(pool))]

    blocks = []
    for item in selected:
        level = sample_reply_level()
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
    if domain == "restaurant":
        parts = []
        if "pricerange" in option:
            parts.append(f"{option['pricerange']} pricing")
        if "food" in option:
            parts.append(f"{option['food']} food")
        if "area" in option:
            parts.append(f"in the {option['area']}")
        if "bookday" in option:
            parts.append(f"for {option['bookday']}")
        if "booktime" in option:
            parts.append(f"at {option['booktime']}")

        desc = ", ".join(parts)
        return f"You should choose the restaurant with {desc}."

    elif domain == "hotel":
        parts = []
        if "pricerange" in option:
            parts.append(f"{option['pricerange']} pricing")
        if "stars" in option:
            parts.append(f"{option['stars']}-star")
        if "area" in option:
            parts.append(f"in the {option['area']}")
        if "parking" in option:
            parts.append(f"parking {option['parking']}")
        if "internet" in option:
            parts.append(f"internet {option['internet']}")
        if "type" in option:
            parts.append(f"type {option['type']}")

        desc = ", ".join(parts)
        return f"You should choose the hotel with {desc}."

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

        desc = ", ".join(parts)
        return f"You should choose the train {desc}."

    else:
        return f"You should choose the option with constraints: {option}."

def build_candidates(
    domain: str,
    full_values: dict[str, str],
    true_slots: dict[str, str],
    wrong_slots: dict[str, list[str]],
    num_candidates: int = 4,
    max_changed_slots: int | None = None,
) -> tuple[list[str], int]:
    if max_changed_slots is None:
        max_changed_slots = len(true_slots)

    true_option = deepcopy(full_values)
    candidates = [deepcopy(true_option)]

    query_slot_names = [slot for slot in true_slots if slot in wrong_slots]
    if not query_slot_names:
        return [render_candidate(domain, true_option)], 0

    while len(candidates) < num_candidates:
        cand = deepcopy(true_option)

        num_changed = random.randint(1, min(max_changed_slots, len(query_slot_names)))
        slots_to_change = random.sample(query_slot_names, k=num_changed)

        for slot in slots_to_change:
            cand[slot] = random.choice(wrong_slots[slot])

        if cand not in candidates:
            candidates.append(cand)

    random.shuffle(candidates)

    candidate_texts = [render_candidate(domain, c) for c in candidates]
    correct_index = candidates.index(true_option)

    return candidate_texts, correct_index

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
    split = "train"
    split_dir = os.path.join(BASE_DIR, split)
    dialogues = load_dialogues_from_split(split_dir)

    data = []
    for i, dialog in enumerate(dialogues[:100]): 
        sample = generate_sample(dialog, num_distractors=3, sample_id=f"ex_{i:04d}")
        if sample is not None:
            data.append(sample)

    output_path = os.path.join(OUTPUT_BASE, f"{split}_debug.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"saved {len(data)} samples to {output_path}")

if __name__ == "__main__":
    main()