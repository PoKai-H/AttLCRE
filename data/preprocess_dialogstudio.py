import json
import os
import tiktoken

INPUT_DIR = "/Users/reeseliu/Desktop/Linear attention/DialogStudio/conversational-recommendation-dialogues"
OUTPUT_DIR = "/Users/reeseliu/Desktop/Linear attention/Dataset/DialogStudio/conversational-recommendation-dialogues"

os.makedirs(OUTPUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("cl100k_base")


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


def build_dialogue(dialog_content):
    logs = dialog_content.get("log", [])
    lines = []

    for turn in logs:
        user_text = turn.get("user utterance", "").strip()
        system_text = turn.get("system response", "").strip()

        if user_text:
            lines.append(f"[A] {user_text}")
        if system_text:
            lines.append(f"[B] {system_text}")

    if not lines:
        return None, None

    dialogue_text = "\n".join(lines)
    num_turns = len(lines)
    turn_range = get_turn_range(num_turns)
    num_tokens = len(enc.encode(dialogue_text))

    result = {
        "dialogue": dialogue_text
    }

    meta = {
        "num_turns": num_turns,
        "turn_range": turn_range,
        "num_tokens": num_tokens
    }

    return result, meta


def get_dialog_iterable(data):
    if isinstance(data, dict):
        return data.values(), len(data)
    elif isinstance(data, list):
        return data, len(data)
    else:
        return [], 0


def process_file(input_path, output_path):
    print(f"\nProcessing {os.path.basename(input_path)}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogs, num_threads = get_dialog_iterable(data)
    print(f"Threads: {num_threads}")

    count = 0
    sample_printed = False

    with open(output_path, "w", encoding="utf-8") as out:
        for dialog_content in dialogs:
            if not isinstance(dialog_content, dict):
                continue

            result, meta = build_dialogue(dialog_content)
            if result is None:
                continue

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

            if not sample_printed:
                print()
                print(f"\nTurns: {meta['num_turns']}")
                print(f"Turn range: {meta['turn_range']}")
                print(f"Tokens: {meta['num_tokens']}")
                print("==============\n")
                sample_printed = True

    print(f"Saved {count} → {output_path}")


def main():
    total_dialogues = 0

    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.lower().endswith(".json"):
                continue

            input_path = os.path.join(root, filename)

            rel_dir = os.path.relpath(root, INPUT_DIR)
            save_dir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(save_dir, exist_ok=True)

            output_path = os.path.join(
                save_dir,
                filename.replace(".json", ".jsonl")
            )

            process_file(input_path, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()