import json

path = "data/datadata/multiwoz_generator/test/test.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

max_len = 0
max_sample = None

for sample in data:
    l = len(sample["dialogue"])
    if l > max_len:
        max_len = l
        max_sample = sample

print("Longest sample turns:", max_len)
print("Sample ID:", max_sample["sample_id"])