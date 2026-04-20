import json
from pathlib import Path

from src.render import build_rows


INPUT_PATH = Path("data/multiwoz_dev/dev/dialogues_001.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    decoder = json.JSONDecoder()
    content = path.read_text(encoding="utf-8")
    items: list[dict] = []
    index = 0

    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break

        item, next_index = decoder.raw_decode(content, index)
        items.append(item)
        index = next_index

    return items


def normalize_samples(samples: list[dict]) -> list[dict]:
    normalized: list[dict] = []

    for sample in samples:
        item = dict(sample)
        if "candidates" not in item and "candidate" in item:
            item["candidates"] = item["candidate"]
        normalized.append(item)

    return normalized


data = normalize_samples(load_jsonl(INPUT_PATH))
rows = build_rows(data)

print(rows[0])
