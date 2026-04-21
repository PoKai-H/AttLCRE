from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.render import build_rows
from src.dataset import RankingDataset, collate_fn
from src.models import build_model_and_tokenizer
from src.eval import (
    collect_candidate_scores,
    compute_sample_level_predictions,
    print_eval_summary,
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["bert", "longformer", "small_transformer"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs_test_only")
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # render-related args
    parser.add_argument("--include_roles", action="store_true")
    parser.add_argument("--include_speaker", action="store_true")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="full",
        choices=["full", "candidate_only", "local_only"],
    )
    parser.add_argument("--local_k", type=int, default=3)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading new test data...")
    test_samples = load_json(args.test_path)

    print("Rendering rows...")
    test_rows = build_rows(
        test_samples,
        include_speaker=args.include_speaker,
        include_roles=args.include_roles,
        render_mode=args.render_mode,
        local_k=args.local_k,
    )

    print("Building model and tokenizer...")
    model, tokenizer, max_length = build_model_and_tokenizer(args.model)
    model.to(device)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)

    test_dataset = RankingDataset(test_rows, tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("Running evaluation...")
    test_candidate_outputs = collect_candidate_scores(model, test_loader, device)
    test_predictions = compute_sample_level_predictions(test_candidate_outputs)

    print_eval_summary("New Test Set", test_predictions)

    save_json(output_dir / "test_predictions.json", test_predictions)
    print(f"Saved predictions to: {output_dir / 'test_predictions.json'}")


if __name__ == "__main__":
    main()