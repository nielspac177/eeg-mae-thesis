"""Resumable, checkpointed training loop — the safety net for long MPS runs.

Every epoch, :meth:`ResumableTrainer.fit` writes a checkpoint containing the model,
optimizer, scheduler, loss history, and **all RNG states** (the trainer's own
DataLoader generator, plus the global torch/numpy/MPS generators). On construction,
if a checkpoint already exists in ``run_dir`` it is loaded and training continues
from the next epoch — so a run killed at any point (Ctrl-C, crash, closing the lid)
resumes cleanly with ``fit`` called again.

**Determinism guarantee (tested on CPU):** training N epochs straight is bit-for-bit
identical to training k epochs, reconstructing the trainer from the checkpoint, and
training the remaining N-k. See ``tests/test_resume_equivalence.py``. On MPS the
resume is *correct* (continues from saved weights/optimizer) but not guaranteed
bit-exact, because the Metal RNG is not always fully serialisable; capturing it is
attempted defensively below.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CKPT_NAME = "ckpt_last.pt"
STATE_NAME = "state.json"


class ResumableTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: object | None = None,
        device: str | torch.device = "cpu",
        run_dir: Path | str | None = None,
        grad_clip: float | None = 1.0,
        seed: int = 42,
        resume: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.grad_clip = grad_clip

        # Dedicated CPU generator drives DataLoader shuffling, so resume is reproducible
        # independently of the global RNG.
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.start_epoch = 0
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        if resume and self.run_dir is not None and (self.run_dir / CKPT_NAME).exists():
            self._load_checkpoint()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        make_train_loader: Callable[[torch.Generator], DataLoader],
        num_epochs: int,
        make_val_loader: Callable[[], DataLoader] | None = None,
        on_batch: Callable[[torch.Tensor], torch.Tensor] | None = None,
        progress: bool = False,
    ) -> dict[str, list[float]]:
        """Train (resuming if a checkpoint exists) up to ``num_epochs`` total.

        Parameters
        ----------
        make_train_loader
            ``generator -> DataLoader``. Receiving the trainer's generator keeps
            shuffling reproducible across resume. Recreated each epoch.
        make_val_loader
            Optional ``() -> DataLoader`` for monitoring; appended to ``history``.
        on_batch
            Optional ``x -> x`` hook applied to inputs before the forward pass
            (e.g. SpecAugment / mixup). For MAE pretraining the batch is the input
            and the loss is read from ``model(x)[0]``; see ``_step``.
        """
        if self.start_epoch >= num_epochs:
            return self.history

        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            loader = make_train_loader(self.generator)
            losses = []
            iterator = loader
            if progress:
                from tqdm.auto import tqdm

                iterator = tqdm(loader, desc=f"epoch {epoch + 1}/{num_epochs}", leave=False)
            for batch in iterator:
                loss = self._step(batch, on_batch)
                losses.append(float(loss.detach().cpu()))
            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
            self.history["val_loss"].append(
                self._validate(make_val_loader) if make_val_loader is not None else float("nan")
            )
            self._save_checkpoint(next_epoch=epoch + 1)
        return self.history

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        """Softmax predictions over a (label-free or labelled) loader, shape ``(n, C)``."""
        self.model.eval()
        out = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = self.model(x.to(self.device))
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _step(self, batch, on_batch) -> torch.Tensor:
        """One optimisation step. Supports both supervised ``(x, y)`` and MAE ``x`` batches."""
        self.optimizer.zero_grad()
        if isinstance(batch, (list, tuple)):  # supervised: (x, soft_label)
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            if on_batch is not None:
                x, y = on_batch(x, y)
            loss = self.loss_fn(self.model(x), y)
        else:  # self-supervised MAE: model(x) -> (loss, pred, mask)
            x = batch.to(self.device)
            if on_batch is not None:
                x = on_batch(x)
            out = self.model(x)
            loss = out[0] if isinstance(out, (list, tuple)) else self.loss_fn(out, x)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def _validate(self, make_val_loader) -> float:
        self.model.eval()
        losses = []
        for batch in make_val_loader():
            if isinstance(batch, (list, tuple)):
                x, y = batch
                loss = self.loss_fn(self.model(x.to(self.device)), y.to(self.device))
            else:
                x = batch.to(self.device)
                out = self.model(x)
                loss = out[0] if isinstance(out, (list, tuple)) else self.loss_fn(out, x)
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    # -- checkpoint I/O ----------------------------------------------------
    def _save_checkpoint(self, next_epoch: int) -> None:
        if self.run_dir is None:
            self.start_epoch = next_epoch
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "next_epoch": next_epoch,
            "history": self.history,
            "rng": self._collect_rng(),
        }
        # Atomic write: tmp then replace, so a crash mid-write never corrupts the checkpoint.
        tmp = self.run_dir / (CKPT_NAME + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(self.run_dir / CKPT_NAME)
        (self.run_dir / STATE_NAME).write_text(
            json.dumps({"next_epoch": next_epoch, "history": self.history}, indent=2)
        )
        self.start_epoch = next_epoch

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self.run_dir / CKPT_NAME, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["next_epoch"]
        self.history = ckpt.get("history", self.history)
        self._restore_rng(ckpt.get("rng", {}))

    def _collect_rng(self) -> dict:
        rng = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "generator": self.generator.get_state(),
        }
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            try:
                rng["mps"] = torch.mps.get_rng_state()
            except Exception:
                pass
        return rng

    def _restore_rng(self, rng: dict) -> None:
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "generator" in rng:
            self.generator.set_state(rng["generator"])
        if "mps" in rng and torch.backends.mps.is_available():
            try:
                torch.mps.set_rng_state(rng["mps"])
            except Exception:
                pass
