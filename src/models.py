from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    LongformerForSequenceClassfication,
    LongformerTokenizer
)



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
        max_length = 256

    elif model_name == "longformer":
        tokenizer = LongformerTokenizer.from_predtrained("allenai/longformer-base-4096")
        model = LongformerForSequenceClassfication.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels = 2
        )
        max_length = 1024
    
    else:
        raise NotImplementedError
    
    return model, tokenizer, max_length