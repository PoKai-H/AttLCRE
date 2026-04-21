from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.render import build_rows
from src.dataset import RankingDataset, collate_fn
from src.models import build_model_and_tokenizer
from src.trainer import run_training
from src.eval import (
    collect_candidate_scores,
    compute_sample_level_predictions,
    print_eval_summary,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["bert", "longformer"])
    parser.add_argument("--train_path", type=str, default="new_data2/train.json")
    parser.add_argument("--val_path", type=str, default="new_data2/val.json")
    parser.add_argument("--test_path", type=str, default="new_data2/test.json")
    parser.add_argument("--output_dir", type=str, default="new_data2/outputs")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_roles", action="store_true")
    parser.add_argument("--include_speaker", action="store_true")
    parser.add_argument("--render_mode", type=str, default="full",
                    choices=["full", "candidate_only", "local_only"])
    parser.add_argument("--local_k", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_samples = load_json(args.train_path)
    val_samples = load_json(args.val_path)
    test_samples = load_json(args.test_path)

    print("Rendering rows...")
    train_rows = build_rows(
        train_samples,
        include_speaker=args.include_speaker,
        include_roles=args.include_roles,
        render_mode=args.render_mode,
        local_k=args.local_k,
    )
    val_rows = build_rows(
        val_samples,
        include_speaker=args.include_speaker,
        include_roles=args.include_roles,
        render_mode=args.render_mode,
        local_k=args.local_k,
    )
    test_rows = build_rows(
        test_samples,
        include_speaker=args.include_speaker,
        include_roles=args.include_roles,
        render_mode=args.render_mode,
        local_k=args.local_k,
    )

    model, tokenizer, max_length = build_model_and_tokenizer(args.model)
    model.to(device)

    train_dataset = RankingDataset(train_rows, tokenizer, max_length=max_length)
    val_dataset = RankingDataset(val_rows, tokenizer, max_length=max_length)
    test_dataset = RankingDataset(test_rows, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Model: {args.model}")
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)} | Test rows: {len(test_rows)}")

    model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        output_dir=str(output_dir),
    )

    val_candidate_outputs = collect_candidate_scores(model, val_loader, device)
    val_predictions = compute_sample_level_predictions(val_candidate_outputs)

    test_candidate_outputs = collect_candidate_scores(model, test_loader, device)
    test_predictions = compute_sample_level_predictions(test_candidate_outputs)

    print_eval_summary("Validation", val_predictions)
    print_eval_summary("Test", test_predictions)

    save_json(output_dir / "val_predictions.json", val_predictions)
    save_json(output_dir / "test_predictions.json", test_predictions)

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()