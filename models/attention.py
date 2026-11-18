import math

import torch
import torch.nn as nn

from config import BERTConfig


class BERTSelfAttention(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = (
            config.hidden_size // config.num_attention_heads
        )
        self.all_head_size = config.hidden_size

        self.query = nn.Linear(
            config.hidden_size,
            self.all_head_size
        )

        self.key = nn.Linear(
            config.hidden_size,
            self.all_head_size
        )

        self.value = nn.Linear(
            config.hidden_size,
            self.all_head_size
        )

        self.dropout = nn.Dropout(
            config.attention_probs_dropout_prob
        )

    def transpose_for_scores(self, x):

        new_shape = (
            x.size()[:-1]
            + (
                self.num_attention_heads,
                self.attention_head_size,
            )
        )

        x = x.view(*new_shape)

        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        query_layer = self.transpose_for_scores(
            self.query(hidden_states)
        )

        key_layer = self.transpose_for_scores(
            self.key(hidden_states)
        )

        value_layer = self.transpose_for_scores(
            self.value(hidden_states)
        )

        attention_scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )

        if attention_mask is not None:

            if attention_mask.dim() == 2:
                attention_mask = attention_mask[
                    :, None, None, :
                ]

            elif attention_mask.dim() != 4:
                raise ValueError(
                    "attention_mask must have 2 or 4 dimensions"
                )

            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                torch.finfo(attention_scores.dtype).min
            )

        attention_probs = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_probs = self.dropout(
            attention_probs
        )

        context_layer = torch.matmul(
            attention_probs,
            value_layer
        )

        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()

        new_context_shape = (
            context_layer.size()[:-2]
            + (self.all_head_size,)
        )

        context_layer = context_layer.view(
            *new_context_shape
        )

        return context_layer, attention_probs