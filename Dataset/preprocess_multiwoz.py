import json
import os

BASE_DIR = "/Users/reeseliu/Desktop/Linear attention/multiwoz/data/MultiWOZ_2.2"
SPLITS = ["train", "dev", "test"]

OUTPUT_BASE = "multiwoz_outputs"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def convert_dialogue(dialog):
    lines = []

    for turn in dialog.get("turns", []):
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

    if len(lines) < 4:
        return None

    return "\n".join(lines)

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
                d = convert_dialogue(dialog)
                if d:
                    out.write(
                        json.dumps({"dialogue": d}, ensure_ascii=False) + "\n"
                    )
                    count += 1
                    total_count += 1

                    if not sample_printed:
                        print("\n=== SAMPLE ===")
                        print(d)
                        print("==============\n")
                        sample_printed = True

        print(f"Saved {count} → {output_path}")

    print(f"TOTAL {split}: {total_count}")