import random
import uuid
import json
from copy import deepcopy


def generate_sample(sample_id=None):
    if sample_id is None:
        sample_id = f"ex_{uuid.uuid4().hex[:6]}"

    all_slots = {
        "areas": ["centre", "north", "south", "east", "west"],
        "prices": ["cheap", "moderate", "expensive"],
        "food": ["Italian", "Chinese", "Japenese", "American"],
        "parking": ["Yes", "No"],
        "ranking": ["1", "2", "3", "4", "5"]
    }
    # list to change dict_keys to sequence that random.sample can use
    # k = 2 or 3
    selected_slots = random.sample(list(all_slots.keys()),k=random.choice([2, 3])  
)

    true_slots = dict()

    for slot in selected_slots:
        true_slots[slot] = random.choice(all_slots[slot])

    wrong_slot = dict()

    for slot in selected_slots:
        candidates = all_slots[slot].copy()
        candidates.remove(true_slots[slot])
        wrong_slot[slot] = candidates
    print(wrong_slot)

generate_sample()