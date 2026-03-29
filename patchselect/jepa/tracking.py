"""Experiment tracking integrations."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from patchselect.jepa.config import WandbConfig

logger = logging.getLogger(__name__)


class NoOpTracker:
    run_id: str | None = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        return

    def log_checkpoint(self, path: Path, aliases: list[str] | None = None) -> None:
        return

    def finish(self, status: str = "completed") -> None:
        return


class WandbTracker:
    def __init__(
        self,
        cfg: WandbConfig,
        *,
        run_dir: Path,
        resolved_config: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        api_key = os.environ.get("WANDB_API_KEY") or getattr(getattr(wandb, "api", None), "api_key", None)
        if cfg.mode == "online" and not api_key:
            raise RuntimeError(
                "wandb.mode=online requires authentication. Set WANDB_API_KEY or use wandb.mode=offline/disabled."
            )
        self.run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            group=cfg.group,
            job_type=cfg.job_type,
            tags=list(cfg.tags),
            notes=cfg.notes,
            mode=cfg.mode,
            dir=str(run_dir),
            config=resolved_config,
            id=run_id,
            resume="allow" if run_id else None,
        )
        self.cfg = cfg
        self.run_id = self.run.id

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_checkpoint(self, path: Path, aliases: list[str] | None = None) -> None:
        if not self.cfg.log_checkpoints:
            return
        artifact = self._wandb.Artifact(
            name=f"{self.run.project}-{self.run.id}-checkpoint-{path.stem}",
            type="model",
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases or [])

    def finish(self, status: str = "completed") -> None:
        self.run.finish(exit_code=0 if status == "completed" else 1)


def build_tracker(
    cfg: WandbConfig,
    *,
    run_dir: Path,
    resolved_config: dict[str, Any],
    run_id: str | None = None,
) -> NoOpTracker | WandbTracker:
    if cfg.mode == "disabled":
        logger.info("Weights & Biases tracking is disabled.")
        return NoOpTracker()
    return WandbTracker(cfg, run_dir=run_dir, resolved_config=resolved_config, run_id=run_id)
