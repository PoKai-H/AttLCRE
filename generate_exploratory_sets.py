from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from sample_generator import (
    build_distractor_pool,
    build_candidates,
    generate_distractor_pairs,
    generate_noise_blocks,
    generate_signal_pair,
    render_reply,
    render_slot_signal,
    sample_reply_level,
)


ALL_SLOTS = {
    "area": ["centre", "north", "south", "east", "west"],
    "price": ["cheap", "moderate", "expensive"],
    "food": ["Italian", "Chinese", "Japenese", "American"],
    "parking": ["Yes", "No"],
    "diet": ["vegan", "vegetarian", "omnivore"],
}


def sample_slot_values() -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    selected_slots = random.sample(list(ALL_SLOTS.keys()), k=random.choice([2, 3]))
    full_values = {slot: random.choice(values) for slot, values in ALL_SLOTS.items()}
    true_slots = {slot: full_values[slot] for slot in selected_slots}

    wrong_slots = {}
    for slot in selected_slots:
        wrong_values = ALL_SLOTS[slot].copy()
        wrong_values.remove(true_slots[slot])
        wrong_slots[slot] = wrong_values

    return full_values, true_slots, wrong_slots


def flatten(blocks: list[list[dict]]) -> list[dict]:
    return [turn for block in blocks for turn in block]


def generate_distractor_pairs_with_replacement(
    full_values: dict[str, str],
    wrong_slots: dict[str, list[str]],
    true_slots: dict[str, str],
    num_distractors: int,
) -> list[list[dict]]:
    pool = build_distractor_pool(
        full_values=full_values,
        wrong_slots=wrong_slots,
        true_slots=true_slots,
    )

    blocks = []
    for _ in range(num_distractors):
        item = random.choice(pool)
        slot = item["slot"]
        value = item["value"]
        dtype = item["distractor_type"]
        level = sample_reply_level()

        blocks.append([
            {
                "speaker": "A",
                "text": render_slot_signal(slot, value),
                "role": "distractor",
                "slot": slot,
                "value": value,
                "distractor_type": dtype,
            },
            {
                "speaker": "B",
                "text": render_reply(slot, value, level),
                "role": "distractor",
                "slot": slot,
                "value": value,
                "distractor_type": dtype,
            },
        ])

    return blocks


def signal_query_distance(dialogue: list[dict]) -> int:
    query_index = len(dialogue) - 1
    signal_indices = [
        idx
        for idx, turn in enumerate(dialogue)
        if turn.get("role") == "signal"
    ]
    if not signal_indices:
        return -1
    return min(query_index - idx for idx in signal_indices)


def split_blocks_by_suffix_turns(
    blocks: list[list[dict]],
    target_suffix_turns: int,
) -> tuple[list[list[dict]], list[list[dict]]]:
    shuffled = blocks.copy()
    random.shuffle(shuffled)

    suffix = []
    suffix_turns = 0
    while shuffled and suffix_turns < target_suffix_turns:
        block = shuffled.pop()
        suffix.append(block)
        suffix_turns += len(block)

    random.shuffle(shuffled)
    random.shuffle(suffix)
    return shuffled, suffix


def generate_variant_sample(
    variant: str,
    sample_id: str,
    num_distractors: int,
    num_noise: int,
    suffix_turn_range: tuple[int, int],
) -> dict:
    full_values, true_slots, wrong_slots = sample_slot_values()

    signal_blocks = generate_signal_pair(true_slots)
    if variant == "high_distractor":
        distractor_blocks = generate_distractor_pairs_with_replacement(
            full_values=full_values,
            wrong_slots=wrong_slots,
            true_slots=true_slots,
            num_distractors=num_distractors,
        )
    else:
        distractor_blocks = generate_distractor_pairs(
            full_values=full_values,
            wrong_slots=wrong_slots,
            true_slots=true_slots,
            num_distractors=num_distractors,
        )
    noise_blocks = generate_noise_blocks(num_noise=num_noise)

    if variant not in {"short_distance", "long_distance", "high_distractor"}:
        raise ValueError(f"Unsupported variant: {variant}")

    target_suffix_turns = random.randint(*suffix_turn_range)
    prefix_blocks, suffix_blocks = split_blocks_by_suffix_turns(
        blocks=distractor_blocks + noise_blocks,
        target_suffix_turns=target_suffix_turns,
    )

    blocks = prefix_blocks + signal_blocks + suffix_blocks
    difficulty = variant

    query = random.choice([
        "Which option should I choose?",
        "Which restaurant best fits my needs?",
        "Which one matches what I asked for earlier?",
        "Given what I said before, which option is best?",
    ])
    dialogue = flatten(blocks)
    dialogue.append({
        "speaker": "A",
        "text": query,
        "role": "query",
    })

    candidate_texts, correct_index = build_candidates(
        full_values=full_values,
        true_slots=true_slots,
        wrong_slots=wrong_slots,
    )

    distance = signal_query_distance(dialogue)

    return {
        "sample_id": sample_id,
        "dialogue": dialogue,
        "query": query,
        "candidates": candidate_texts,
        "correct_index": correct_index,
        "full_values": full_values,
        "true_slots": true_slots,
        "wrong_slots": wrong_slots,
        "metadata": {
            "difficulty": difficulty,
            "num_query_slots": len(true_slots),
            "num_noise_blocks": num_noise,
            "num_distractors": len(distractor_blocks),
            "has_distractor": len(distractor_blocks) > 0,
            "signal_query_distance": distance,
            "target_suffix_turns": target_suffix_turns,
            "suffix_turn_range": list(suffix_turn_range),
            "num_turns": len(dialogue),
        },
    }


def generate_dataset(
    variant: str,
    num_samples: int,
    num_distractors: int,
    num_noise: int,
    suffix_turn_range: tuple[int, int],
) -> list[dict]:
    return [
        generate_variant_sample(
            variant=variant,
            sample_id=f"{variant}_{uuid.uuid4().hex[:8]}",
            num_distractors=num_distractors,
            num_noise=num_noise,
            suffix_turn_range=suffix_turn_range,
        )
        for _ in range(num_samples)
    ]


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="new_data2")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    configs = {
        "short_distance": {
            "num_distractors": 3,
            "num_noise": 5,
            "suffix_turn_range": (0, 4),
        },
        "long_distance": {
            "num_distractors": 4,
            "num_noise": 30,
            "suffix_turn_range": (20, 34),
        },
        "high_distractor": {
            "num_distractors": 8,
            "num_noise": 24,
            "suffix_turn_range": (20, 38),
        },
    }

    for variant, config in configs.items():
        data = generate_dataset(
            variant=variant,
            num_samples=args.num_samples,
            num_distractors=config["num_distractors"],
            num_noise=config["num_noise"],
            suffix_turn_range=config["suffix_turn_range"],
        )
        output_path = output_dir / f"{variant}.json"
        write_json(output_path, data)
        distances = [s["metadata"]["signal_query_distance"] for s in data]
        avg_distance = sum(distances) / len(distances)
        print(
            f"Wrote {len(data)} samples to {output_path} | "
            f"signal distance min/avg/max: {min(distances)}/{avg_distance:.2f}/{max(distances)} | "
            f"distractors: {config['num_distractors']} | "
            f"noise blocks: {config['num_noise']}"
        )


if __name__ == "__main__":
    main()
