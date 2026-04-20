from __future__ import annotations

from collections import defaultdict
from typing import Any

import json
import torch


@torch.no_grad()
def collect_candidate_scores(model, dataloader, device: str) -> list[dict[str, Any]]:
    model.eval()
    outputs_list: list[dict[str, Any]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        logits = outputs.logits
        positive_scores = logits[:, 1].detach().cpu().tolist()

        for i, score in enumerate(positive_scores):
            outputs_list.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "candidate_index": batch["candidate_index"][i],
                    "gold_index": batch["gold_index"][i],
                    "score": score,
                    "candidate_text": batch["candidate_text"][i],
                    "metadata": batch["metadata"][i],
                }
            )

    return outputs_list


def compute_sample_level_predictions(
    candidate_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in candidate_outputs:
        grouped[row["sample_id"]].append(row)

    predictions: list[dict[str, Any]] = []

    for sample_id, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["candidate_index"])
        best_row = max(rows, key=lambda x: x["score"])

        pred_index = best_row["candidate_index"]
        gold_index = rows[0]["gold_index"]
        correct = int(pred_index == gold_index)

        predictions.append(
            {
                "sample_id": sample_id,
                "pred_index": pred_index,
                "gold_index": gold_index,
                "correct": correct,
                "predicted_candidate": best_row["candidate_text"],
                "scores_by_candidate": [
                    {
                        "candidate_index": r["candidate_index"],
                        "score": r["score"],
                        "candidate_text": r["candidate_text"],
                    }
                    for r in rows
                ],
                "metadata": rows[0]["metadata"],
            }
        )

    return predictions


def accuracy_from_predictions(predictions: list[dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    return sum(p["correct"] for p in predictions) / len(predictions)


def evaluate_by_metadata(
    predictions: list[dict[str, Any]],
    field_name: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)

    for pred in predictions:
        metadata = pred.get("metadata", {})
        value = metadata.get(field_name, "missing")
        grouped[str(value)].append(pred["correct"])

    result: dict[str, dict[str, float]] = {}
    for key, values in grouped.items():
        result[key] = {
            "count": float(len(values)),
            "accuracy": float(sum(values) / len(values)),
        }
    return result


def print_eval_summary(name: str, predictions: list[dict[str, Any]]) -> None:
    overall_acc = accuracy_from_predictions(predictions)
    print(f"\n===== {name} =====")
    print(f"Sample-level accuracy: {overall_acc:.4f}")

    print("\nBy difficulty:")
    print(json.dumps(evaluate_by_metadata(predictions, "difficulty"), indent=2))

    print("\nBy has_distractor:")
    print(json.dumps(evaluate_by_metadata(predictions, "has_distractor"), indent=2))