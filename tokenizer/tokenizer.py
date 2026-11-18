
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import models
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers

from config import BERTConfig


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 데이터 경로
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.txt"

# Tokenizer 저장 경로
OUTPUT_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def train_tokenizer():

    config = BERTConfig()

    # 1. 데이터 확인
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    if CORPUS_PATH.stat().st_size == 0:
        raise ValueError("Corpus file is empty.")

    # 2. WordPiece 모델 생성
    tokenizer = Tokenizer(
        models.WordPiece(
            unk_token="[UNK]"
        )
    )

    # 3. 텍스트 정규화
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )

    # 4. Pre-tokenization
    tokenizer.pre_tokenizer = (
        pre_tokenizers.BertPreTokenizer()
    )

    # 5. WordPiece Trainer 설정
    special_tokens = [
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[UNK]"
    ]

    trainer = trainers.WordPieceTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        continuing_subword_prefix="##"
    )

    # 6. Vocabulary 학습
    tokenizer.train(
        files=[str(CORPUS_PATH)],
        trainer=trainer
    )

    # 7. Special Token ID 검증
    expected_ids = {
        "[PAD]": config.pad_token_id,
        "[CLS]": config.cls_token_id,
        "[SEP]": config.sep_token_id,
        "[MASK]": config.mask_token_id,
        "[UNK]": config.unk_token_id
    }

    for token, expected_id in expected_ids.items():

        actual_id = tokenizer.token_to_id(token)

        if actual_id != expected_id:
            raise ValueError(
                f"{token}: expected {expected_id}, "
                f"got {actual_id}"
            )

    # 8. BERT 입력 형식 설정
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = (
        processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_id),
                ("[SEP]", sep_id)
            ]
        )
    )

    # 9. WordPiece Decoder
    tokenizer.decoder = decoders.WordPiece(
        prefix="##"
    )

    # 10. Tokenizer 저장
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save(str(OUTPUT_PATH))

    # 11. 결과 확인
    actual_vocab_size = tokenizer.get_vocab_size()

    print("=" * 40)
    print("Tokenizer training completed!")
    print(f"Vocabulary size: {actual_vocab_size}")
    print(f"Tokenizer saved: {OUTPUT_PATH}")
    print("=" * 40)

    if actual_vocab_size != config.vocab_size:
        print(
            "[WARNING] Update config.py vocab_size "
            f"to {actual_vocab_size} before model construction."
        )


if __name__ == "__main__":
    train_tokenizer()
