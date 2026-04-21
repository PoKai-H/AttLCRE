import random
import uuid
import json
from copy import deepcopy


NOISE_BANK = [
    "Can you give me the phone number?",
    "What are the opening hours?",
    "Do I need to make a reservation?",
    "I might go with some friends.",
    "It is for a small gathering.",
    "Can you help me with booking?",
    "I am planning this for tonight.",
]


def generate_sample(data_path: str, num_distractors=3, sample_id=None):
    if sample_id is None:
        sample_id = f"ex_{uuid.uuid4().hex[:6]}"

    all_slots = {
        "area": ["centre", "north", "south", "east", "west"],
        "price": ["cheap", "moderate", "expensive"],
        "food": ["Italian", "Chinese", "Japenese", "American"],
        "parking": ["Yes", "No"],
        "diet": ["vegan", "vegetarian", "omnivore"]
    }
    # list to change dict_keys to sequence that random.sample can use
    # k = 2 or 3
    selected_slots = random.sample(list(all_slots.keys()),k=random.choice([2, 3])  
    )
    # we will use full signal to make the output fix length but only query on selected slots
    full_values = {
        slot: random.choice(all_slots[slot])
        for slot in all_slots
    }
    
    true_slots = dict()
    for slot in selected_slots:
        true_slots[slot] = full_values[slot]
    
    wrong_slots = dict()
    for slot in selected_slots:
        candidates = all_slots[slot].copy()
        candidates.remove(true_slots[slot])
        wrong_slots[slot] = candidates

    # signal blocks
    # expected format:
    # [
    #   [turn1, turn2],
    #   [turn3, turn4],
    # ]
    signal_blocks = generate_signal_pair(true_slots)

    # distractor blocks
    # expected format:
    # [
    #   [turn1, turn2],
    #   [turn3, turn4],
    # ]
    distractor_blocks = generate_distractor_pairs(
        full_values=full_values,
        wrong_slots=wrong_slots,
        true_slots=true_slots,
        num_distractors=num_distractors,
    )


    # TODO: now use simple placeholder, use multiwoz data as noise
    noise_blocks = generate_noise_blocks(num_noise=5)
    
    # 8. shuffle blocks, not turns
    blocks = signal_blocks + distractor_blocks + noise_blocks
    random.shuffle(blocks)
    

    # 9. flatten blocks into dialogue
    dialogue = [turn for block in blocks for turn in block]

    # 10. query turn stays last
    query = random.choice([
        "Which option should I choose?",
        "Which restaurant best fits my needs?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which option is best?",
    ])
    dialogue.append({
        "speaker": "A",
        "text": query,
        "role": "query",
    })

    # 11. candidates
    candidate_texts, correct_index = build_candidates(
        full_values=full_values,
        true_slots=true_slots,
        wrong_slots=wrong_slots,
    )

    # 12. return structured sample
    sample = {
        "sample_id": sample_id,
        "dialogue": dialogue,
        "query": query,
        "candidates": candidate_texts,
        "correct_index": correct_index,
        "full_values": full_values,
        "true_slots": true_slots,
        "wrong_slots": wrong_slots,
        "metadata": {
            "difficulty": "hard",
            "num_query_slots": len(selected_slots),
            "num_noise_blocks": len(noise_blocks),
            "num_distractors": len(distractor_blocks),
            "has_distractor": len(distractor_blocks) > 0,
            "num_turns": len(dialogue),
        },
    }

    return sample
    # print(true_slots)
    # print("signal: ",generate_signal_pair(true_slots))
    # print("distractor:", generate_distractor_pairs(full_values, wrong_slots, true_slots, num_distractors=3))
    # print("candidates",build_candidates(full_values, true_slots, wrong_slots))

   
    
def generate_query() -> str:
    return random.choice([
        "Which option should I choose?",
        "Which restaurant best fits my needs?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which option is best?",
    ])


def generate_signal_pair(true_slot: dict[str: str]):
    signals = []

    for slot, value in true_slot.items():
        signals.append(render_signal_pair(slot, value))

    return signals

# === unuse function
# def generate_reply(target_slot: dict[str: str]):
#     replies = []

#     for slot, value in target_slot.items():
#         level = sample_reply_level()
#         replies.append(render_reply(slot, value, level))

#     return replies

def render_signal_pair(slot: str, value: str) -> list[dict]:
    level = sample_reply_level()
    return [
        {
            "speaker": "A",
            "text": render_slot_signal(slot, value),
            "role": "signal",
            "slot": slot,
            "value": value,
        },
        {
            "speaker": "B",
            "text": render_reply(slot, value, level),
            "role": "signal",
            "slot": slot,
            "value": value,
        },
    ]


def render_slot_signal(slot: str, value:str):
    if slot == "parking":
        if value == "Yes":
            candidates = [
                "I need parking.",
                "A place with parking is required.",
                "Parking is important for me.",
            ]
        else:
            candidates = [
                "Parking is not necessary.",
                "I do not need parking.",
                "Parking does not matter to me.",
            ]
        return random.choice(candidates)
    
    if slot == "diet":
        templates = {
            "vegan": [
                "I am vegan.",
                "It needs to be vegan.",
                "I only want vegan food.",
            ],
            "vegetarian": [
                "I am vegetarian.",
                "I want vegetarian food.",
                "A vegetarian place would be best.",
            ],
            "omnivore": [
                "I eat meat.",
                "A place that serves meat is fine.",
                "I am looking for a place with meat dishes.",
            ],
        }
        return random.choice(templates[value])


    templates = {
        "area": [
            "It should be in the {value}.",
            "I would prefer the {value} area.",
            "The {value} part of town works best for me.",
            "I am looking for something in the {value}.",
        ],
        "price": [
            "I want something {value}.",
            "A {value} place would be best.",
            "My preference is for a {value} option.",
            "It should be {value}.",
        ],
        "food": [
            "{value} food would be ideal.",
            "I would prefer {value} food.",
            "I am looking for a {value} restaurant.",
            "{value} would be my first choice.",
        ],
    }

    template = random.choice(templates[slot])
    return template.format(value=value)


def sample_reply_level():
    """
    we sample from a guassian distribution
    p(r < 0) ~= 0.5 
    p(0 < r < 1) ~= 0.34
    p(r > 1) ~= 0.16

    level 0 implies to no info
    level 1 implies to weak confirmation
    level 2 implies to strong confirmation
    """
    r = random.gauss(0, 1)
    if r < 0:
        return 0
    elif r < 1:
        return 1
    else:
        return 2

def render_reply(slot: str, value: str, level: int) -> str:
    """
    level 0 implies to no info
    level 1 implies to weak confirmation
    level 2 implies to strong confirmation
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
        "area": [
            f"Okay, I will focus on the {value} area.",
            f"Got it, I will search in the {value}.",
        ],
        "price": [
            f"Okay, I will look for {value} options.",
            f"Got it, I will keep the price {value}.",
        ],
        "food": [
            f"Okay, I will look for {value} restaurants.",
            f"Got it, I will keep {value} food in mind.",
        ],
        "parking": [
            "Okay, I will only consider places with parking." if value == "Yes"
            else "Okay, parking is not necessary.",
        ],
        "diet": [
            f"Okay, I will look for {value} options.",
            f"Got it, I will keep {value} preferences in mind.",
        ],
    }

    if level == 0:
        return random.choice(generic)
    elif level == 1:
        return random.choice(weak)
    else:
        return random.choice(strong_templates[slot])

def sample_distractor_type() -> str:
    p = random.random()
    if p < 0.4:
        return "hard_negative"
    elif p < 0.8:
        return "irrelevant"
    else:
        return "mixed" # to add some randomness



def render_distractor(
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
) -> list[dict]:
    dtype = sample_distractor_type()
    irrelevant_slots = [slot for slot in full_values if slot not in true_slots]

    if dtype == "irrelevant" and not irrelevant_slots:
        dtype = "hard_negative"

    if dtype == "hard_negative":
        slot = random.choice(list(true_slots.keys()))
        value = random.choice(wrong_slots[slot])

    elif dtype == "irrelevant":
        slot = random.choice(irrelevant_slots)
        value = full_values[slot]

    else:  # mixed
        if irrelevant_slots and random.random() >= 0.5:
            slot = random.choice(irrelevant_slots)
            value = full_values[slot]
        else:
            slot = random.choice(list(true_slots.keys()))
            value = random.choice(wrong_slots[slot])

    level = sample_reply_level()

    return [
        {
            "speaker": "A",
            "text": render_slot_signal(slot, value),
            "role": "distractor",
            "slot": slot,
            "value": value,
            "distractor_type": dtype,
        },
        {
            "speaker": "B",
            "text": render_reply(slot, value, level),
            "role": "distractor",
            "slot": slot,
            "value": value,
            "distractor_type": dtype,
        },
    ]

def render_candidate(option: dict[str, str]) -> str:
    return (
        f"You should choose a {option['price']} {option['food']} restaurant "
        f"in the {option['area']} with parking {option['parking']} "
        f"and diet {option['diet']}."
    )



def build_candidates(
    full_values: dict[str, str],
    true_slots: dict[str, str],
    wrong_slots: dict[str, list[str]],
    num_candidates: int = 4,
    max_changed_slots: int | None = None,
) -> tuple[list[str], int]:
    """
    Build candidate options for ranking.

    Args:
        full_values: full correct assignment for all slots
        true_slots: query-relevant slots with correct values
        wrong_slots: for each query slot, list of wrong values
        num_candidates: total number of candidates, including the correct one
        max_changed_slots: maximum number of query slots to corrupt in a negative.
                           If None, defaults to len(true_slots).

    Returns:
        candidate_texts, correct_index
    """
    if max_changed_slots is None:
        max_changed_slots = len(true_slots)

    true_option = deepcopy(full_values)
    candidates = [deepcopy(true_option)]

    query_slot_names = list(true_slots.keys())

    while len(candidates) < num_candidates:
        cand = deepcopy(true_option)

        # Randomly decide how many relevant slots to corrupt
        num_changed = random.randint(1, min(max_changed_slots, len(query_slot_names)))
        slots_to_change = random.sample(query_slot_names, k=num_changed)

        for slot in slots_to_change:
            cand[slot] = random.choice(wrong_slots[slot])

        # Avoid duplicates
        if cand not in candidates:
            candidates.append(cand)

    random.shuffle(candidates)

    candidate_texts = [render_candidate(c) for c in candidates]
    correct_index = candidates.index(true_option)

    return candidate_texts, correct_index

def build_distractor_pool(
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
) -> list[dict]:
    pool = []

    # hard negatives: selected slot + wrong value
    for slot, wrong_values in wrong_slots.items():
        for value in wrong_values:
            pool.append({
                "distractor_type": "hard_negative",
                "slot": slot,
                "value": value,
            })

    # irrelevant: non-query slot + its full value
    for slot, value in full_values.items():
        if slot not in true_slots:
            pool.append({
                "distractor_type": "irrelevant",
                "slot": slot,
                "value": value,
            })

    return pool


def generate_distractor_pairs(
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
    num_distractors: int,
) -> list[dict]:
    pool = build_distractor_pool(full_values, wrong_slots, true_slots)

    random.shuffle(pool)
    selected = pool[:min(num_distractors, len(pool))]

    turns = []
    for item in selected:
        level = sample_reply_level()
        slot = item["slot"]
        value = item["value"]
        dtype = item["distractor_type"]

        turns.append([
            {
                "speaker": "A",
                "text": render_slot_signal(slot, value),
                "role": "distractor",
                "slot": slot,
                "value": value,
                "distractor_type": dtype,
            },
            {
                "speaker": "B",
                "text": render_reply(slot, value, level),
                "role": "distractor",
                "slot": slot,
                "value": value,
                "distractor_type": dtype,
            },
        ])

    return turns


def generate_noise_blocks(num_noise=5):
    blocks = []
    for _ in range(num_noise):
        blocks.append([
            {
                "speaker": random.choice(["A", "B"]),
                "text": random.choice(NOISE_BANK),
                "role": "noise",
            }
        ])
    return blocks

def main():

    data = [generate_sample("") for _ in range(500)]

    with open("new_data2/test.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()