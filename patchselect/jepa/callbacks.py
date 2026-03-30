"""Custom PyTorch Lightning callbacks for JEPA training."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import lightning as pl
import torch

from patchselect.jepa.config import JEPAConfig
from patchselect.jepa.logging_utils import host_name, resolve_git_sha
from patchselect.io_utils import write_json

logger = logging.getLogger(__name__)


class StateFileCallback(pl.Callback):
    """Writes a state.json file to the run directory after each training step.

    Preserves the state-file-based monitoring from the original runner.
    """

    def __init__(
        self,
        cfg: JEPAConfig,
        run_dir: Path,
        repo_root: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_dir = run_dir
        self.repo_root = repo_root
        self._git_sha = resolve_git_sha(repo_root)
        self._metadata = {
            "family": cfg.model.family,
            "masking_strategy": cfg.masking.strategy,
            "regularizer_name": cfg.regularizer.name,
            "target_encoder_kind": cfg.target_encoder.kind,
            "projector_kind": cfg.projector.kind,
            "augment_profile": cfg.augment.profile,
        }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Only write state file periodically to avoid I/O overhead
        if trainer.global_step % self.cfg.logging.log_every_steps != 0:
            return

        data_module = trainer.datamodule
        health = {}
        if data_module is not None and hasattr(data_module, "dataset") and data_module.dataset is not None:
            health = data_module.dataset.health.to_dict()

        cursor = {}
        if data_module is not None and hasattr(data_module, "data_cursor"):
            cursor = data_module.data_cursor

        wandb_run_id = None
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment
            if hasattr(exp, "id"):
                wandb_run_id = exp.id

        payload = {
            "status": "running",
            "global_step": trainer.global_step,
            "host": host_name(),
            "git_sha": self._git_sha,
            "wandb_run_id": wandb_run_id,
            "data_cursor": cursor,
            "health": health,
            **self._metadata,
        }
        write_json(self.run_dir / "state.json", payload)


class CursorCheckpointCallback(pl.Callback):
    """Saves and restores the data cursor and model training state.

    On checkpoint save:  injects cursor state into the checkpoint.
    On checkpoint load:  restores cursor state so training resumes from
    the correct position in the tar stream.
    """

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        data_module = trainer.datamodule
        extra: dict[str, Any] = {}

        # Save data cursor
        if data_module is not None and hasattr(data_module, "data_cursor"):
            extra["data_cursor"] = data_module.data_cursor

        # Save model training state (mask sampler step, etc.)
        if hasattr(pl_module, "model"):
            extra["model_training_state"] = pl_module.model.get_training_state()

        # Save metric smoother state
        if hasattr(pl_module, "smoother"):
            extra["meter_state"] = pl_module.smoother.state_dict()

        checkpoint["jepa_state"] = extra

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        extra = checkpoint.get("jepa_state", {})

        # Restore data cursor
        cursor = extra.get("data_cursor", {})
        data_module = trainer.datamodule
        if data_module is not None and hasattr(data_module, "set_cursor") and cursor:
            data_module.set_cursor(
                tar_index=int(cursor.get("tar_index", 0)),
                member_index=int(cursor.get("member_index", 0)),
            )
            logger.info(
                "Restored data cursor from checkpoint: tar_index=%d member_index=%d",
                cursor.get("tar_index", 0),
                cursor.get("member_index", 0),
            )

        # Restore model training state
        model_state = extra.get("model_training_state")
        if model_state and hasattr(pl_module, "model"):
            pl_module.model.load_training_state(model_state)

        # Restore metric smoother
        meter_state = extra.get("meter_state")
        if meter_state and hasattr(pl_module, "smoother"):
            pl_module.smoother.load_state_dict(meter_state)


class ConsoleLogCallback(pl.Callback):
    """Periodically prints a formatted console log line like the old runner did."""

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._step_start_time: float = 0.0

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        max_steps = self.cfg.trainer.max_steps
        if not (step == 1 or step == max_steps or step % self.cfg.logging.log_every_steps == 0):
            return

        batch_time = time.perf_counter() - self._step_start_time
        batch_size = batch["images"].shape[0] if "images" in batch else 0

        # Collect logged metrics
        logged = trainer.callback_metrics
        loss = float(logged.get("train/loss", 0.0))
        mse = logged.get("train/mse")
        reg = logged.get("train/regularizer")
        lr = float(logged.get("lr-AdamW", logged.get("optim/lr", 0.0)))
        ema = logged.get("target_encoder/ema_drift")

        parts = [
            f"step={step}",
            f"loss={loss:.4f}",
        ]
        if mse is not None:
            parts.append(f"mse={float(mse):.4f}")
        if reg is not None:
            parts.append(f"reg={float(reg):.4f}")
        parts.append(f"lr={lr:.2e}")
        if batch_size > 0:
            parts.append(f"img/s={batch_size / max(batch_time, 1e-8):.1f}")
        if ema is not None:
            parts.append(f"ema={float(ema):.3e}")

        logger.info(" | ".join(parts))


class TargetEncoderEMACallback(pl.Callback):
    """Updates the exponential moving average (EMA) of the target encoder after each optimizer step."""

    def __init__(self, ema_momentum: float) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # stable_pretraining spt.Module expects pl_module.model to be the base PyTorch module
        model = getattr(pl_module, "model", None)
        if model is not None and hasattr(model, "has_ema_target") and model.has_ema_target():
            ema_drift = model.update_target_encoder(self.ema_momentum)
            pl_module.log("target_encoder/ema_drift", ema_drift, on_step=True, logger=True)
