from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    LongformerForSequenceClassification,
    LongformerTokenizer
)

from src.attnBiasBert import AttnBiasBert
from src.memBert import MemBert

class OutputWrapper:
    """
    Make custom model outputs look similar to Hugging Face outputs
    """

    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


def build_model_and_tokenizer(model_name: str):
    model_name = model_name.lower()

    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2
        )
        max_length = 512 # change from 256 to 512 for fair comparison with MemBert

    elif model_name == "longformer":
        tokenizer = LongformerTokenizer.from_predtrained("allenai/longformer-base-4096")
        model = LongformerForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels = 2
        )
        max_length = 1024

    elif model_name == "bert_mem":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        special_tokens = {"additional_special_tokens": ["[MEM]"]}

        tokenizer.add_special_tokens(special_tokens)

        model = MemBert(model_name="bert-base-uncased", num_labels=2, num_mem_tokens=4)
        model.mem_token_id = tokenizer.convert_tokens_to_ids("[MEM]")
        model.bert.resize_token_embeddings(len(tokenizer))

        max_length = 512

    elif model_name == "bert_attn_bias":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = AttnBiasBert(
            model_name="bert-base-uncased",
            num_labels=2,
            bias_strength=1.0,
        )
        max_length = 512
    
    else:
        raise NotImplementedError
    
    return model, tokenizer, max_length
