
from pathlib import Path

from tokenizers import Tokenizer


class BertTokenizer:

    def __init__(self, tokenizer_path):

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        # 학습된 Tokenizer 불러오기
        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        # Special Token ID
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")

        # Vocabulary 크기
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Special Token 확인
        special_ids = [
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
            self.unk_token_id
        ]

        if any(token_id is None for token_id in special_ids):
            raise ValueError("Missing special tokens.")

    def tokenize(self, text):

        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        return encoding.tokens

    def encode(self, text, text_pair=None):

        encoding = self.tokenizer.encode(
            text,
            text_pair
        )

        return {
            "input_ids": encoding.ids,
            "token_type_ids": encoding.type_ids,
            "attention_mask": encoding.attention_mask
        }

    def decode(self, ids, skip_special_tokens=True):

        return self.tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens
        )

    def get_vocab_size(self):

        return self.tokenizer.get_vocab_size()
