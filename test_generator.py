import random
import uuid
import json
from copy import deepcopy


def generate_hard_sample(sample_id=None):
    if sample_id is None:
        sample_id = f"hard_{uuid.uuid4().hex[:6]}"

    areas = ["centre", "north", "south", "east", "west"]
    prices = ["cheap", "moderate", "expensive"]

    # --- True signal ---
    true_area = random.choice(areas)
    true_price = random.choice(prices)

    # --- Distractors ---
    wrong_area = random.choice([a for a in areas if a != true_area])
    wrong_price = random.choice([p for p in prices if p != true_price])

    dialogue = []

    # 1. Early signal
    dialogue.append({
        "speaker": "A",
        "text": f"I am looking for a {true_price} restaurant in the {true_area}.",
        "role": "signal"
    })

    dialogue.append({
        "speaker": "B",
        "text": f"Got it, {true_price} in the {true_area}.",
        "role": "signal"
    })

    # 2. Add lots of noise
    noise_templates = [
        "Can you also give me the phone number?",
        "I might travel next week.",
        "I like Italian food.",
        "Do they serve breakfast?",
        "I have a meeting later.",
        "It might rain tomorrow.",
        "I need parking as well.",
    ]

    for _ in range(random.randint(4, 7)):
        dialogue.append({
            "speaker": random.choice(["A", "B"]),
            "text": random.choice(noise_templates),
            "role": "noise"
        })

    # 3. Strong distractor
    dialogue.append({
        "speaker": "A",
        "text": f"Actually, a {wrong_price} place in the {wrong_area} could also work.",
        "role": "distractor"
    })

    dialogue.append({
        "speaker": "B",
        "text": f"Sure, {wrong_price} options in the {wrong_area} are available.",
        "role": "distractor"
    })

    # 4. More noise after distractor
    for _ in range(random.randint(2, 4)):
        dialogue.append({
            "speaker": random.choice(["A", "B"]),
            "text": random.choice(noise_templates),
            "role": "noise"
        })

    # 5. Query
    query = "What is the best choice?"

    dialogue.append({
        "speaker": "A",
        "text": query,
        "role": "query"
    })

    # --- Candidates ---
    candidates = [
        f"You should choose a {true_price} restaurant in the {true_area}.",   # correct
        f"You should choose a {wrong_price} restaurant in the {wrong_area}.", # distractor
        f"You should choose a {true_price} restaurant in the {wrong_area}.",  # partial wrong
        "There is not enough information."
    ]

    random.shuffle(candidates)
    correct_text = f"You should choose a {true_price} restaurant in the {true_area}."
    correct_index = candidates.index(correct_text)

    return {
        "sample_id": sample_id,
        "dialogue": dialogue,
        "query": query,
        "candidates": candidates,
        "correct_index": correct_index,
        "metadata": {
            "true_area": true_area,
            "true_price": true_price,
            "wrong_area": wrong_area,
            "wrong_price": wrong_price,
            "difficulty": "hard",
            "signal_position": 0,
            "distractor_position": len(dialogue) - 5,
            "num_turns": len(dialogue),
            "has_distractor": True,
        }
    }


def generate_dataset(n=5):
    return [generate_hard_sample() for _ in range(n)]


def make_local(sample, k=3):
    new_sample = deepcopy(sample)
    new_sample["dialogue"] = sample["dialogue"][-k:]

    if "metadata" not in new_sample:
        new_sample["metadata"] = {}

    new_sample["metadata"]["ablation"] = "local_only"
    new_sample["metadata"]["num_turns"] = len(new_sample["dialogue"])
    return new_sample


def remove_signal(sample):
    new_sample = deepcopy(sample)

    # Remove turns whose role is signal
    new_sample["dialogue"] = [
        turn for turn in sample["dialogue"]
        if turn.get("role") != "signal"
    ]

    if "metadata" not in new_sample:
        new_sample["metadata"] = {}

    new_sample["metadata"]["ablation"] = "remove_signal"
    new_sample["metadata"]["num_turns"] = len(new_sample["dialogue"])
    return new_sample


def candidate_only(sample):
    new_sample = deepcopy(sample)

    # Only keep the query turn as context
    new_sample["dialogue"] = [
        {
            "speaker": "A",
            "text": sample["query"],
            "role": "query"
        }
    ]

    if "metadata" not in new_sample:
        new_sample["metadata"] = {}

    new_sample["metadata"]["ablation"] = "candidate_only"
    new_sample["metadata"]["num_turns"] = 1
    return new_sample


def main():
    hard_data = generate_dataset(100)
    # local_data = [make_local(sample, k=3) for sample in hard_data]
    # rms_data = [remove_signal(sample) for sample in hard_data]
    # cand_only_data = [candidate_only(sample) for sample in hard_data]

    with open("new_data/val_hard.json", "w", encoding="utf-8") as f:
        json.dump(hard_data, f, indent=2, ensure_ascii=False)

    # with open("new_data/local_only.json", "w", encoding="utf-8") as f:
    #     json.dump(local_data, f, indent=2, ensure_ascii=False)

    # with open("new_data/remove_signal.json", "w", encoding="utf-8") as f:
    #     json.dump(rms_data, f, indent=2, ensure_ascii=False)

    # with open("new_data/candidate_only.json", "w", encoding="utf-8") as f:
    #     json.dump(cand_only_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()