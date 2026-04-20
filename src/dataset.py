from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

class RankingDataset(Dataset):
    """
    Candidate-level dataset for binary ranking.
    Each item corresponds to (context, candidate, lebel)
    """

    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]

        encoded = self.tokenizer(
            row["context"],
            row["candidate"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # [CLS] context [SEP] candidate [SEP]
        # encoded = {
        #    "input_ids": tensor([101, ..., 102, ..., 102]), 
        #    "attention_mask": tensor([1, 1, 1, ...]),  -> which are padding
        #    "token_type_ids": tensor([0,0,...,1,1,...]) -> to distinguish context/candidate, 0 = context 1 = candidate
        # }

        # create input for model
        item = {
            "input_ids": encoded["input_ids"].squeeze(0), # (1, seq_len) -> (seq_len,)
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long), # 1 = correct 0 = wrong
            "sample_id": row["sample_id"],
            "candidate_index": row["candidate_index"],
            "gold_index": row["gold_index"],
            "metadata": row["metadata"],
            "candidate_text": row["candidate"],
        }

        # bert has token_type_ids but longformer dosent, pad with 0s
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        else:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])

        return item

# stacking samples into batch
def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "sample_id": [x["sample_id"] for x in batch],
        "candidate_index": [x["candidate_index"] for x in batch],
        "gold_index": [x["gold_index"] for x in batch],
        "metadata": [x["metadata"] for x in batch],
        "candidate_text": [x["candidate_text"] for x in batch],
    }