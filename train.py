
import argparse
import json
import math
import random
import hashlib

from pathlib import Path

import torch

from torch.utils.data import DataLoader

from config import BERTConfig

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


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_documents(documents, validation_ratio, seed):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation-ratio must be between 0 and 1")
    cleaned = []
    for document in documents:
        sentences = document["sentences"]
        if any(not isinstance(s, str) for s in sentences):
            raise ValueError("Each sentence must be a string")
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            cleaned.append({"sentences": sentences})
    eligible = [i for i, doc in enumerate(cleaned) if len(doc["sentences"]) >= 2]
    if len(cleaned) < 4 or len(eligible) < 2:
        raise ValueError("Train and validation each need two non-empty documents and a document with two sentences")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    validation_ids = {eligible[0]}
    reserved_train = eligible[1]
    others = [i for i in range(len(cleaned)) if i not in validation_ids and i != reserved_train]
    rng.shuffle(others)
    size = min(len(cleaned) - 2, max(2, math.ceil(len(cleaned) * validation_ratio)))
    validation_ids.update(others[:size - 1])
    train = [doc for i, doc in enumerate(cleaned) if i not in validation_ids]
    validation = [doc for i, doc in enumerate(cleaned) if i in validation_ids]
    return train, validation


class ResumableBatchSampler:
    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch, self.start_batch = 0, 0
        self.total_batches = math.ceil(len(dataset) / batch_size)

    def set_epoch(self, epoch, start_batch=0):
        if not 0 <= start_batch <= self.total_batches:
            raise ValueError("Invalid checkpoint batch position")
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        for start in range(self.start_batch * self.batch_size, len(indices), self.batch_size):
            yield indices[start:start + self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_batch


def main():

    from tokenizer.tokenizer import BertTokenizer

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

    parser.add_argument("--documents", default=str(DOCUMENTS_PATH))
    parser.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--resume", help="Checkpoint to resume at its saved optimizer update")

    args = parser.parse_args()

    # 설정 검증
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation must be positive")

    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup-ratio must be between 0 and 1")

    if not 3 <= args.max_length <= BERTConfig().max_position_embeddings:
        raise ValueError("max-length must fit the three special tokens and model positions")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid worker count, learning rate, gradient norm, or weight decay")

    set_seed(args.seed)

    device = get_device(args.device)

    print(f"Device: {device}")

    # 1. Tokenizer
    tokenizer = BertTokenizer(
        args.tokenizer
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
        args.documents
    )

    print(f"Documents: {len(documents)}")

    train_documents, validation_documents = split_documents(documents, args.validation_ratio, args.seed)

    # 4. Dataset
    train_dataset = BERTPretrainingDataset(
        documents=train_documents,
        tokenizer=tokenizer,
        config=config,
        max_length=args.max_length,
        seed=args.seed
    )
    validation_dataset = BERTPretrainingDataset(
        documents=validation_documents, tokenizer=tokenizer, config=config,
        max_length=args.max_length, seed=args.seed + 1
    )

    print(f"Training examples: {len(train_dataset)}")

    # 5. DataLoader
    batch_sampler = ResumableBatchSampler(train_dataset, args.batch_size, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed)
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed + 1)
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

    # Resumption is only valid with the same input and optimizer schedule.
    metadata = {
        "documents_sha256": file_digest(args.documents),
        "tokenizer_sha256": file_digest(args.tokenizer),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "validation_ratio": args.validation_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length, "max_grad_norm": args.max_grad_norm,
        "seed": args.seed, "device_type": device.type, "use_amp": args.use_amp,
    }

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
        use_amp=args.use_amp,
        resume_metadata=metadata
    )

    resume_state = trainer.load_checkpoint(args.resume) if args.resume else None

    # 10. Training
    history = trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        validation_loader=validation_loader,
        resume_state=resume_state
    )

    print("\nTraining completed!")
    print(f"Checkpoint: {args.output_dir}/last.pt")


if __name__ == "__main__":
    main()
