import torch.nn as nn

from config import BERTConfig


class BERTPredictionHeadTransform(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        if config.hidden_act.lower() == "gelu":
            self.transform_act_fn = nn.GELU()
        elif config.hidden_act.lower() == "relu":
            self.transform_act_fn = nn.ReLU()
        else:
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}"
            )

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(
            hidden_states
        )
        hidden_states = self.LayerNorm(
            hidden_states
        )

        return hidden_states


class BERTLMPredictionHead(nn.Module):

    def __init__(
        self,
        config: BERTConfig,
        embedding_weights
    ):
        super().__init__()

        self.transform = BERTPredictionHeadTransform(
            config
        )

        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        self.decoder.weight = embedding_weights

        self.bias = nn.Parameter(
            embedding_weights.new_zeros(
                config.vocab_size
            )
        )

    def forward(self, hidden_states):

        hidden_states = self.transform(
            hidden_states
        )

        hidden_states = self.decoder(
            hidden_states
        )

        hidden_states = hidden_states + self.bias

        return hidden_states


class BERTPreTrainingHeads(nn.Module):

    def __init__(
        self,
        config: BERTConfig,
        embedding_weights
    ):
        super().__init__()

        self.predictions = BERTLMPredictionHead(
            config,
            embedding_weights
        )

        self.seq_relationship = nn.Linear(
            config.hidden_size,
            2
        )

    def forward(
        self,
        sequence_output,
        pooled_output
    ):

        prediction_scores = self.predictions(
            sequence_output
        )

        seq_relationship_score = self.seq_relationship(
            pooled_output
        )

        return (
            prediction_scores,
            seq_relationship_score
        )