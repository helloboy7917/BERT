
import argparse
import json
import math
import random

from pathlib import Path

import torch

from torch.utils.data import DataLoader

from config import BERTConfig
from tokenizer.tokenizer import BertTokenizer

from data.dataset import BERTPretrainingDataset

from models.bert import BERTForPreTraining

from training.optimizer import build_optimizer
from training.scheduler import get_linear_schedule_with_warmup
from training.trainer import BERTTrainer


PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_PATH = (
    PROJECT_ROOT / "tokenizer" / "tokenizer.json"
)

DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "documents.jsonl"
)


def load_documents(path):

    documents = []

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            document = json.loads(line)

            if not isinstance(document, dict):
                raise ValueError(
                    f"Invalid document at line {line_number}"
                )

            sentences = document.get("sentences")

            if not isinstance(sentences, list):
                raise ValueError(
                    f"Missing sentences at line {line_number}"
                )

            documents.append({
                "sentences": sentences
            })

    return documents


def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):

    if device_name == "auto":

        return torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    return device


def main():

    parser = argparse.ArgumentParser(
        description="Train BERT from scratch"
    )

    # Training Arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--use-amp",
        action="store_true"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints")
    )

    args = parser.parse_args()

    # 설정 검증
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation must be positive")

    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup-ratio must be between 0 and 1")

    if not 0 < args.max_length <= 256:
        raise ValueError("max-length must be between 1 and 256")

    set_seed(args.seed)

    device = get_device(args.device)

    print(f"Device: {device}")

    # 1. Tokenizer
    tokenizer = BertTokenizer(
        TOKENIZER_PATH
    )

    # 2. BERT Configuration
    config = BERTConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        unk_token_id=tokenizer.unk_token_id
    )

    # 3. Corpus
    documents = load_documents(
        DOCUMENTS_PATH
    )

    print(f"Documents: {len(documents)}")

    # 4. Dataset
    train_dataset = BERTPretrainingDataset(
        documents=documents,
        tokenizer=tokenizer,
        config=config,
        max_length=args.max_length
    )

    print(f"Training examples: {len(train_dataset)}")

    # 5. DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # 6. Model
    model = BERTForPreTraining(
        config
    )

    # 7. Optimizer
    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 8. Scheduler
    updates_per_epoch = math.ceil(
        len(train_loader)
        / args.gradient_accumulation_steps
    )

    total_training_steps = (
        updates_per_epoch * args.epochs
    )

    warmup_steps = int(
        total_training_steps * args.warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # 9. Trainer
    trainer = BERTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp
    )

    # 10. Training
    history = trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs
    )

    print("\nTraining completed!")
    print(f"Checkpoint: {args.output_dir}/last.pt")


if __name__ == "__main__":
    main()
