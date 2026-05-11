from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

BIAS_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "best",
    "choose",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "option",
    "should",
    "suitable",
    "the",
    "to",
    "you",
}


class RankingDataset(Dataset):
    """
    Candidate-level dataset for binary ranking.
    Each item corresponds to (context, candidate, lebel)
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        num_mem_tokens=0,
        use_memory=False,
        use_attention_bias=False,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_mem_tokens = num_mem_tokens
        self.use_memory = use_memory
        self.use_attention_bias = use_attention_bias

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]

        context = row["context"]
        candidate = row["candidate"]
        label = row["label"]
        
        if self.use_memory:
            mem_prefix = " ".join(["[MEM]"] * self.num_mem_tokens)
            context = f"{mem_prefix} {context}"
            
        encoded = self.tokenizer(
            context,
            candidate,
            truncation="only_first", 
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

        if self.use_attention_bias:
            item["attention_bias"] = self._build_attention_bias(
                input_ids=item["input_ids"],
                token_type_ids=item["token_type_ids"],
                candidate=candidate,
            )

        return item

    def _build_attention_bias(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        candidate: str,
    ) -> torch.Tensor:
        candidate_ids = self.tokenizer(
            candidate,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        candidate_ids = {
            token_id
            for token_id in candidate_ids
            if token_id not in self.tokenizer.all_special_ids
            and self.tokenizer.convert_ids_to_tokens(token_id).lower() not in BIAS_STOP_TOKENS
        }

        bias = torch.zeros_like(input_ids, dtype=torch.float)
        for idx, token_id in enumerate(input_ids.tolist()):
            is_context_token = token_type_ids[idx].item() == 0
            is_candidate_overlap = token_id in candidate_ids
            is_special_token = token_id in self.tokenizer.all_special_ids
            if is_context_token and is_candidate_overlap and not is_special_token:
                bias[idx] = 1.0
        return bias

# stacking samples into batch
def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
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
    if "attention_bias" in batch[0]:
        collated["attention_bias"] = torch.stack([x["attention_bias"] for x in batch])
    return collated
