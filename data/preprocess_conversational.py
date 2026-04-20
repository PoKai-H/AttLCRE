import json
import os
import tiktoken
from collections import defaultdict

INPUT_DIR = "/Users/reeseliu/Desktop/Linear attention/conversational-datasets/reddit"
OUTPUT_DIR = "/Users/reeseliu/Desktop/Linear attention/conversational_outputs/reddit"

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


def clean_text(text):
    if not text:
        return ""
    text = text.strip()
    if text in ["[deleted]", "[removed]"]:
        return ""
    return text


def build_comment_maps(data):
    comments_by_id = {}
    children_map = defaultdict(list)

    for item in data:
        comment_id = item.get("id")
        parent_id = item.get("parent_id", "")
        body = clean_text(item.get("body", ""))

        if not comment_id or not body:
            continue

        comments_by_id[comment_id] = item

        if parent_id.startswith("t1_"):
            parent_comment_id = parent_id[3:]   # 去掉 t1_
            children_map[parent_comment_id].append(comment_id)

    return comments_by_id, children_map


def get_roots(comments_by_id):
    roots = []
    for comment_id, item in comments_by_id.items():
        parent_id = item.get("parent_id", "")
        if parent_id.startswith("t3_"):
            roots.append(comment_id)
    return roots


def dfs_paths(comment_id, children_map, path, all_paths):
    path.append(comment_id)

    if comment_id not in children_map or len(children_map[comment_id]) == 0:
        all_paths.append(path.copy())
    else:
        for child_id in children_map[comment_id]:
            dfs_paths(child_id, children_map, path, all_paths)

    path.pop()


def extract_all_paths(comments_by_id, children_map):
    roots = get_roots(comments_by_id)
    all_paths = []

    for root_id in roots:
        dfs_paths(root_id, children_map, [], all_paths)

    return all_paths


def build_dialogue_from_path(path, comments_by_id):
    lines = []

    for i, comment_id in enumerate(path):
        item = comments_by_id[comment_id]
        body = clean_text(item.get("body", ""))

        if not body:
            continue

        role = "A" if i % 2 == 0 else "B"
        lines.append(f"[{role}] {body}")

    if len(lines) < 2:
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


def process_file(input_path, output_path):
    print(f"\nProcessing {os.path.basename(input_path)}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Unsupported format: expected a list")
        return

    print(f"Raw comments: {len(data)}")

    comments_by_id, children_map = build_comment_maps(data)
    all_paths = extract_all_paths(comments_by_id, children_map)

    print(f"Conversation paths: {len(all_paths)}")

    count = 0
    sample_printed = False

    with open(output_path, "w", encoding="utf-8") as out:
        for path in all_paths:
            result, meta = build_dialogue_from_path(path, comments_by_id)

            if result is None:
                continue

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

            if not sample_printed:
                print(f"\nTurns: {meta['num_turns']}")
                print(f"Turn range: {meta['turn_range']}")
                print(f"Tokens: {meta['num_tokens']}")
                print("Sample dialogue:")
                print(result["dialogue"][:700])
                print("==============\n")
                sample_printed = True

    print(f"Saved {count} → {output_path}")


def main():
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.endswith(".json"):
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


if __name__ == "__main__":
    main()