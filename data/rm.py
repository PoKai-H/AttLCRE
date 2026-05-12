import json
import os
from copy import deepcopy

INPUT_PATH = "multiwoz_generator/test/test.json"
OUTPUT_DIR = "multiwoz_generator/t"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_signal(sample):
    new_sample = deepcopy(sample)

    new_sample["dialogue"] = [
        turn for turn in sample["dialogue"]
        if turn.get("role") != "signal"
    ]

    new_sample.setdefault("metadata", {})
    new_sample["metadata"]["ablation"] = "remove_signal"
    new_sample["metadata"]["num_turns"] = len(new_sample["dialogue"])

    return new_sample


def local_only(sample, k=4):
    new_sample = deepcopy(sample)

    dialogue = sample["dialogue"]
    new_sample["dialogue"] = dialogue[-k:]

    new_sample.setdefault("metadata", {})
    new_sample["metadata"]["ablation"] = f"local_only_k{k}"
    new_sample["metadata"]["local_k"] = k
    new_sample["metadata"]["num_turns"] = len(new_sample["dialogue"])

    return new_sample


def candidate_only(sample):
    new_sample = deepcopy(sample)

    new_sample["dialogue"] = []

    new_sample.setdefault("metadata", {})
    new_sample["metadata"]["ablation"] = "candidate_only"
    new_sample["metadata"]["num_turns"] = 0

    return new_sample


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rm_signal_data = [remove_signal(sample) for sample in data]
local_only_data = [local_only(sample, k=4) for sample in data]
candidate_only_data = [candidate_only(sample) for sample in data]

with open(os.path.join(OUTPUT_DIR, "rm_signal.json"), "w", encoding="utf-8") as f:
    json.dump(rm_signal_data, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "local_only_k4.json"), "w", encoding="utf-8") as f:
    json.dump(local_only_data, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "candidate_only.json"), "w", encoding="utf-8") as f:
    json.dump(candidate_only_data, f, indent=2, ensure_ascii=False)

print("Done!")
print(f"Saved to: {OUTPUT_DIR}")