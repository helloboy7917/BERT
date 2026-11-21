from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import random

import torch
import torch.nn.functional as F


class BERTTrainer:
    def __init__(self, model, optimizer, scheduler, device, output_dir="checkpoints",
                 gradient_accumulation_steps=1, max_grad_norm=1.0, use_amp=False,
                 resume_metadata=None):
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.resume_metadata = resume_metadata
        self.model.to(self.device)

    def compute_loss(self, outputs, batch):
        scores, labels = outputs["prediction_scores"], batch["mlm_labels"]
        if (labels != -100).any():
            mlm = F.cross_entropy(scores.reshape(-1, scores.size(-1)), labels.reshape(-1), ignore_index=-100)
        else:
            mlm = scores.float().sum() * 0.0
        nsp = F.cross_entropy(outputs["seq_relationship_score"], batch["nsp_label"])
        return mlm + nsp, mlm, nsp

    def _forward(self, batch):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if self.use_amp else nullcontext())
        with context:
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 token_type_ids=batch["token_type_ids"])
            return self.compute_loss(outputs, batch)

    def train_epoch(self, train_loader, epoch=0, start_batch=0, sums=None, history=None):
        self.model.train()
        total_batches = getattr(train_loader.batch_sampler, "total_batches", len(train_loader))
        if total_batches == 0:
            raise ValueError("Training DataLoader is empty")
        sums = dict(sums or {"total_loss": 0.0, "mlm_loss": 0.0, "nsp_loss": 0.0})
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=start_batch):
            total, mlm, nsp = self._forward(batch)
            group_start = (step // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            group_size = min(self.gradient_accumulation_steps, total_batches - group_start)
            self.scaler.scale(total / group_size).backward()
            for key, loss in zip(sums, (total, mlm, nsp)):
                sums[key] += loss.item()
            if (step + 1) % self.gradient_accumulation_steps == 0 or step + 1 == total_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None and self.scaler.get_scale() >= previous_scale:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                if history is not None:
                    self.save_checkpoint(epoch, history, next_batch=step + 1, sums=sums)
        return {key: value / total_batches for key, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, validation_loader):
        self.model.eval()
        totals = {"total_loss": 0.0, "mlm_loss": 0.0, "nsp_loss": 0.0}
        count = len(validation_loader)
        if not count:
            raise ValueError("Validation DataLoader is empty")
        for batch in validation_loader:
            for key, loss in zip(totals, self._forward(batch)):
                totals[key] += loss.item()
        return {key: value / count for key, value in totals.items()}

    def save_checkpoint(self, epoch, history, next_batch=0, sums=None):
        checkpoint = {
            "format_version": 2, "epoch": epoch, "next_batch": next_batch,
            "partial_sums": sums, "history": history, "config": asdict(self.model.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict(), "resume_metadata": self.resume_metadata,
            "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        temporary = self.output_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(self.output_dir / "last.pt")

    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format_version") != 2 or checkpoint.get("resume_metadata") is None:
            raise ValueError("Checkpoint lacks the metadata required for training resume")
        if checkpoint["resume_metadata"] != self.resume_metadata:
            raise ValueError("Checkpoint training settings, data, or tokenizer do not match")
        if checkpoint["config"] != asdict(self.model.config):
            raise ValueError("Checkpoint model configuration does not match")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        random.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if self.device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return checkpoint

    def fit(self, train_loader, epochs, validation_loader=None, resume_state=None):
        state = resume_state or {}
        history = list(state.get("history", []))
        start_epoch = state.get("epoch", 0)
        for epoch in range(start_epoch, epochs):
            start_batch = state.get("next_batch", 0) if epoch == start_epoch else 0
            sums = state.get("partial_sums") if start_batch else None
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            sampler = train_loader.batch_sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch, start_batch)
            elif start_batch:
                raise ValueError("Mid-epoch resume requires a resumable batch sampler")
            metrics = self.train_epoch(train_loader, epoch, start_batch, sums, history)
            validation = self.evaluate(validation_loader) if validation_loader is not None else None
            entry = dict(metrics)
            if validation is not None:
                entry["validation"] = validation
            history.append(entry)
            print(f"Epoch {epoch + 1}/{epochs}: train={metrics} validation={validation}")
            self.save_checkpoint(epoch + 1, history)
        return history
