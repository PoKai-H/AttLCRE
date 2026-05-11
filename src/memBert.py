import torch
import torch.nn as nn

from transformers import (
    BertModel,
    BertTokenizer
)

from transformers.modeling_outputs import SequenceClassifierOutput

class MemBert(nn.Module):

    def __init__(self, model_name="bert-base-uncased", num_labels=2, num_mem_tokens=4):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.num_mem_tokens = num_mem_tokens

        # classifier input: [CLS] + averaged MEM
        self.classifier = nn.Linear(
            self.hidden_size * 2,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        mem_token_id=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden = outputs.last_hidden_state
        batch_size = input_ids.size(0)

        cls_repr = hidden[:, 0, :]

        mem_token_id = self.mem_token_id if mem_token_id is None else mem_token_id
        mem_mask = input_ids == mem_token_id
        mem_hidden = hidden[mem_mask].view(
            batch_size,
            self.num_mem_tokens,
            self.hidden_size,
        )

        mem_repr = mem_hidden.mean(dim=1)

        final_repr = torch.cat([cls_repr, mem_repr], dim=-1)
        logits = self.classifier(final_repr)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
