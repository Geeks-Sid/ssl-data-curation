"""CLI entrypoint for JEPA training using PyTorch Lightning and stable_pretraining."""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from patchselect.jepa.callbacks import (
    ConsoleLogCallback,
    CursorCheckpointCallback,
    StateFileCallback,
    TargetEncoderEMACallback,
)
from patchselect.jepa.config import load_config
from patchselect.jepa.lightning_data import JEPADataModule
from patchselect.jepa.lightning_module import spatial_jepa_forward
from patchselect.jepa.model import TimmPathologySpatialJEPA
from patchselect.jepa.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train JEPA on curated tar streams.")
    parser.add_argument(
        "--config_file",
        action="append",
        default=None,
        help="Optional YAML config file(s) merged on top of configs/jepa/base.yaml",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the resolved config before training starts",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional shortcut for checkpoint.resume=none|auto|<path>",
    )
    return parser.parse_known_args()


def _build_run_dir(cfg) -> Path:
    run_name = cfg.runtime.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(cfg.runtime.output_root) / cfg.runtime.experiment_name / run_name


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def _resolve_ckpt_path(resume: str, run_dir: Path) -> str | None:
    """Resolve Lightning checkpoint path for resuming."""
    if resume == "none":
        return None
    if resume == "auto":
        # Look for Lightning's last.ckpt
        last_ckpt = run_dir / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)
        return None
    # Explicit path
    candidate = Path(resume)
    if candidate.exists():
        return str(candidate)
    return None


def main() -> None:
    args, overrides = parse_args()
    if args.resume is not None:
        overrides.append(f"checkpoint.resume={args.resume}")

    cfg, resolved_config, omega_cfg = load_config(args.config_file, overrides)

    # Build run directory
    run_dir = _build_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = logs_dir / "train.log"
    configure_logging(log_path, cfg.logging.log_level)

    if args.print_config:
        print(OmegaConf.to_yaml(omega_cfg))

    # Save resolved config
    config_path = run_dir / "config.resolved.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=omega_cfg, f=str(config_path))

    # Seed
    _seed_everything(cfg.runtime.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.runtime.matmul_precision)

    repo_root = Path(__file__).resolve().parents[2]

    # -------------------------------------------------- #
    #  Model & Optimizer Setup                           #
    # -------------------------------------------------- #
    model = TimmPathologySpatialJEPA(cfg, mask_seed=cfg.runtime.seed)
    sigreg_module = model.sigreg if cfg.regularizer.name == "sigreg" else None

    # Construct stable_pretraining optimizer format
    scheduler_type = "None"
    if cfg.scheduler.name == "cosine_annealing":
        scheduler_type = "LinearWarmupCosineAnnealingLR"
    elif cfg.scheduler.name == "cosine_annealing_warm_restarts":
        scheduler_type = "LinearWarmupCosineAnnealingWarmRestarts"

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": {
                "type": "AdamW",
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
                "betas": cfg.optimizer.betas,
                "eps": cfg.optimizer.eps,
            },
            "scheduler": {"type": scheduler_type},
            "interval": "step",
        },
    }

    # Wrap model with stable_pretraining Module
    module = spt.Module(
        model=model,
        sigreg=sigreg_module,
        forward=partial(spatial_jepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # -------------------------------------------------- #
    #  Lightning DataModule                              #
    # -------------------------------------------------- #
    data_module = JEPADataModule(cfg)

    # -------------------------------------------------- #
    #  Logger                                            #
    # -------------------------------------------------- #
    pl_logger = None
    if cfg.wandb.mode != "disabled":
        pl_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=list(cfg.wandb.tags),
            notes=cfg.wandb.notes,
            save_dir=str(run_dir),
            offline=(cfg.wandb.mode == "offline"),
        )
        pl_logger.log_hyperparams(resolved_config)

    # -------------------------------------------------- #
    #  Callbacks                                         #
    # -------------------------------------------------- #
    callbacks: list[pl.Callback] = []

    # Lightning ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="step_{step:08d}",
        every_n_train_steps=cfg.checkpoint.save_every_steps,
        save_top_k=cfg.checkpoint.keep_last_k,
        monitor="train/loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Custom JEPA callbacks
    callbacks.append(CursorCheckpointCallback())
    callbacks.append(StateFileCallback(cfg, run_dir=run_dir, repo_root=repo_root))
    callbacks.append(ConsoleLogCallback(cfg))
    
    if cfg.target_encoder.kind == "ema":
        callbacks.append(TargetEncoderEMACallback(cfg.target_encoder.ema_momentum))

    # -------------------------------------------------- #
    #  Trainer                                           #
    # -------------------------------------------------- #
    trainer_kwargs = cfg.trainer.to_trainer_kwargs()
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=str(run_dir),
    )

    # -------------------------------------------------- #
    #  Resume & Run                                      #
    # -------------------------------------------------- #
    ckpt_path = _resolve_ckpt_path(cfg.checkpoint.resume, run_dir)
    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    else:
        logger.info("Starting fresh training run in %s", run_dir)

    logger.info(
        "Starting JEPA Lightning training with stable_pretraining: model=%s device=%s steps=%d batch_size=%d",
        cfg.model.model_name,
        cfg.trainer.accelerator,
        cfg.trainer.max_steps,
        cfg.runtime.batch_size,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_path,
    )
    manager()

    logger.info(
        "JEPA training complete: global_step=%d run_dir=%s",
        trainer.global_step,
        run_dir,
    )


if __name__ == "__main__":
    main()
