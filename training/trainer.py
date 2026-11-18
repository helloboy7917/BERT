
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.nn.functional as F


class BERTTrainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        output_dir="checkpoints",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_amp=False
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device(device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        self.max_grad_norm = max_grad_norm

        self.use_amp = (
            use_amp and self.device.type == "cuda"
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp
        )

        self.model.to(self.device)

    def compute_loss(self, outputs, batch):

        # MLM
        prediction_scores = outputs[
            "prediction_scores"
        ]

        mlm_labels = batch["mlm_labels"]

        if (mlm_labels != -100).any():
            mlm_loss = F.cross_entropy(
                prediction_scores.reshape(
                    -1,
                    prediction_scores.size(-1)
                ),
                mlm_labels.reshape(-1),
                ignore_index=-100
            )
        else:
            # No masked tokens in this batch: keep a zero loss connected to
            # the model so the NSP objective can still be trained.
            mlm_loss = prediction_scores.float().sum() * 0.0

        # NSP
        nsp_logits = outputs[
            "seq_relationship_score"
        ]

        nsp_labels = batch["nsp_label"]

        nsp_loss = F.cross_entropy(
            nsp_logits,
            nsp_labels
        )

        # Total Loss
        total_loss = mlm_loss + nsp_loss

        return total_loss, mlm_loss, nsp_loss

    def train_epoch(self, train_loader):

        self.model.train()

        total_loss_sum = 0.0
        mlm_loss_sum = 0.0
        nsp_loss_sum = 0.0

        total_batches = len(train_loader)

        if total_batches == 0:
            raise ValueError("Training DataLoader is empty")

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(train_loader):

            # Tensor를 Device로 이동
            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }

            # Forward Pass
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                )
                if self.use_amp
                else nullcontext()
            )

            with autocast_context:

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"]
                )

                total_loss, mlm_loss, nsp_loss = (
                    self.compute_loss(
                        outputs,
                        batch
                    )
                )

            # Gradient Accumulation
            group_start = (
                step // self.gradient_accumulation_steps
            ) * self.gradient_accumulation_steps

            group_size = min(
                self.gradient_accumulation_steps,
                total_batches - group_start
            )

            scaled_loss = total_loss / group_size

            self.scaler.scale(
                scaled_loss
            ).backward()

            total_loss_sum += total_loss.item()
            mlm_loss_sum += mlm_loss.item()
            nsp_loss_sum += nsp_loss.item()

            # Optimizer 업데이트 시점
            should_update = (
                (step + 1) % self.gradient_accumulation_steps == 0
                or step + 1 == total_batches
            )

            if should_update:

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                previous_scale = self.scaler.get_scale()

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                # AMP Overflow가 없을 때 Scheduler 진행
                if (
                    self.scheduler is not None
                    and self.scaler.get_scale() >= previous_scale
                ):
                    self.scheduler.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

        return {
            "total_loss": total_loss_sum / total_batches,
            "mlm_loss": mlm_loss_sum / total_batches,
            "nsp_loss": nsp_loss_sum / total_batches
        }

    def save_checkpoint(self, epoch, history):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "config": asdict(self.model.config),
            "history": history
        }

        checkpoint_path = (
            self.output_dir / "last.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    def fit(
        self,
        train_loader,
        epochs
    ):

        history = []

        for epoch in range(epochs):

            metrics = self.train_epoch(
                train_loader
            )

            history.append(metrics)

            print(
                f"\nEpoch {epoch + 1}/{epochs}"
            )

            print(
                f"Total Loss: {metrics['total_loss']:.4f}"
            )

            print(
                f"MLM Loss: {metrics['mlm_loss']:.4f}"
            )

            print(
                f"NSP Loss: {metrics['nsp_loss']:.4f}"
            )

            self.save_checkpoint(
                epoch + 1,
                history
            )

        return history
