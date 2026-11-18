import torch.nn as nn

from config import BERTConfig
from models.attention import BERTSelfAttention


class BERTSelfOutput(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        input_tensor
    ):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(
            hidden_states + input_tensor
        )

        return hidden_states


class BERTAttention(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        self_output, attention_probs = self.self(
            hidden_states,
            attention_mask
        )

        attention_output = self.output(
            self_output,
            hidden_states
        )

        return attention_output, attention_probs


class BERTIntermediate(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size
        )

        if config.hidden_act.lower() == "gelu":
            self.intermediate_act_fn = nn.GELU()
        elif config.hidden_act.lower() == "relu":
            self.intermediate_act_fn = nn.ReLU()
        else:
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}"
            )

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(
            hidden_states
        )

        return hidden_states


class BERTOutput(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        input_tensor
    ):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(
            hidden_states + input_tensor
        )

        return hidden_states


class BERTLayer(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask
        )

        intermediate_output = self.intermediate(
            attention_output
        )

        layer_output = self.output(
            intermediate_output,
            attention_output
        )

        return layer_output, attention_probs


class BERTEncoder(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.layer = nn.ModuleList(
            [
                BERTLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        all_attention_probs = []

        for layer_module in self.layer:

            hidden_states, attention_probs = layer_module(
                hidden_states,
                attention_mask
            )

            all_attention_probs.append(
                attention_probs
            )

        return hidden_states, all_attention_probs


class BERTPooler(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.activation = nn.Tanh()

    def forward(self, hidden_states):

        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(
            first_token_tensor
        )

        pooled_output = self.activation(
            pooled_output
        )

        return pooled_output