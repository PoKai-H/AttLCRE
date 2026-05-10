import json
import os
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

BASE_DIR = "/Users/reeseliu/Desktop/Linear attention/multiwoz/data/MultiWOZ_2.2"
SPLITS = ["train", "dev", "test"]
OUTPUT_BASE = "multiwoz_outputs"
os.makedirs(OUTPUT_BASE, exist_ok=True)

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

def convert_dialogue(dialog):
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

        lines.append(f"[{role}] {utterance}")

    num_valid_turns = len(lines)

    if num_valid_turns < 4:
        return None, None

    dialogue_text = "\n".join(lines)
    num_tokens = len(enc.encode(dialogue_text))
    turn_range = get_turn_range(num_valid_turns)

    result = {
        "dialogue": dialogue_text
    }

    meta = {
        "num_turns": num_valid_turns,
        "turn_range": turn_range,
        "num_tokens": num_tokens
    }

    return result, meta

for split in SPLITS:
    print(f"\n===== Processing {split.upper()} =====")

    DATA_DIR = os.path.join(BASE_DIR, split)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, split)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_count = 0

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

                out.write(json.dumps(result, ensure_ascii=False) + "\n")

                count += 1
                total_count += 1
                if not sample_printed:
                    print(f"\nTurns: {meta['num_turns']}")
                    print(f"Turn range: {meta['turn_range']}")
                    print(f"Tokens: {meta['num_tokens']}")
                    print("==============\n")
                    sample_printed = True

        print(f"Saved {count} → {output_path}")

    print(f"TOTAL {split}: {total_count}")