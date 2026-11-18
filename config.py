
from dataclasses import dataclass


@dataclass
class BERTConfig:

    # Vocabulary
    vocab_size: int = 8000

    # Transformer Architecture
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024

    # Input
    max_position_embeddings: int = 256
    type_vocab_size: int = 2

    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Activation
    hidden_act: str = "gelu"

    # Special Tokens
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4

    # Model Initialization
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )
