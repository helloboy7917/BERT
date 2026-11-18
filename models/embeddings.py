import torch
import torch.nn as nn

from config import BERTConfig


class BERTEmbeddings(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        # Token Embedding
        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )

        # Position Embedding
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )

        # Segment(Token Type) Embedding
        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size
        )

        # Layer Normalization
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

        # Dropout
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None
    ):

        # input_ids shape:
        # [batch_size, sequence_length]

        batch_size, sequence_length = input_ids.shape

        # Token Embedding
        inputs_embeds = self.word_embeddings(
            input_ids
        )

        # Position ID 생성
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # [sequence_length]
        # →
        # [batch_size, sequence_length]
        position_ids = position_ids.unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        # Position Embedding
        position_embeds = self.position_embeddings(
            position_ids
        )

        # Token Type ID가 주어지지 않은 경우
        # 모든 토큰을 Sentence A로 처리
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(
                input_ids
            )

        # Segment Embedding
        token_type_embeds = self.token_type_embeddings(
            token_type_ids
        )

        # 세 Embedding을 더함
        embeddings = (
            inputs_embeds
            + position_embeds
            + token_type_embeds
        )

        # Layer Normalization
        embeddings = self.LayerNorm(
            embeddings
        )

        # Dropout
        embeddings = self.dropout(
            embeddings
        )

        return embeddings