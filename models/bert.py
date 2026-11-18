import torch
import torch.nn as nn

from config import BERTConfig
from models.embeddings import BERTEmbeddings
from models.encoder import BERTEncoder, BERTPooler
from models.heads import BERTPreTrainingHeads


class BERTModel(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.config = config

        self.embeddings = BERTEmbeddings(
            config
        )

        self.encoder = BERTEncoder(
            config
        )

        self.pooler = BERTPooler(
            config
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):

        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(
                input_ids
            )

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids
        )

        sequence_output, all_attention_probs = (
            self.encoder(
                embedding_output,
                attention_mask
            )
        )

        pooled_output = self.pooler(
            sequence_output
        )

        return (
            sequence_output,
            pooled_output,
            all_attention_probs
        )


class BERTForPreTraining(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.config = config

        self.bert = BERTModel(
            config
        )

        self.cls = BERTPreTrainingHeads(
            config,
            self.bert.embeddings.word_embeddings.weight
        )

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range
                )

            elif isinstance(module, nn.LayerNorm):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

        with torch.no_grad():
            self.bert.embeddings.word_embeddings.weight[
                self.config.pad_token_id
            ].zero_()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):

        (
            sequence_output,
            pooled_output,
            all_attention_probs
        ) = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        prediction_scores, seq_relationship_score = (
            self.cls(
                sequence_output,
                pooled_output
            )
        )

        return {
            "prediction_scores": prediction_scores,
            "seq_relationship_score": seq_relationship_score,
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "attention_probs": all_attention_probs
        }