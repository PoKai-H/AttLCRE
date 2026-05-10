import json
from copy import deepcopy

def remove_signal(sample):
    new_sample = deepcopy(sample)

    new_sample["dialogue"] = [
        turn for turn in sample["dialogue"]
        if turn.get("role") != "signal"
    ]

    if "metadata" not in new_sample:
        new_sample["metadata"] = {}

    new_sample["metadata"]["ablation"] = "remove_signal"
    new_sample["metadata"]["num_turns"] = len(new_sample["dialogue"])
    return new_sample

with open("data/datadata/multiwoz_generator/test/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = [remove_signal(sample) for sample in data]

with open("data/datadata/multiwoz_generator/rm_signal.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)
