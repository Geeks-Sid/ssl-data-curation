"""Custom step-based JEPA training runner."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from patchselect.jepa.checkpointing import (
    compute_resume_signature,
    config_hash,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint_bundle,
    write_state_file,
)
from patchselect.jepa.config import JEPAConfig
from patchselect.jepa.data import (
    TarImageStream,
    build_train_transform,
    collate_image_samples,
    discover_tar_manifest,
    manifest_metadata,
)
from patchselect.jepa.logging_utils import host_name, resolve_git_sha
from patchselect.jepa.metrics import (
    JsonlMetricWriter,
    MetricSmoother,
    compute_representation_metrics,
    parameter_l2_norm,
)
from patchselect.jepa.model import ForwardOutput, TimmPathologySpatialJEPA, compute_losses
from patchselect.jepa.tracking import build_tracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_path: Path
    log_path: Path
    config_path: Path
    state_path: Path


def build_run_paths(cfg: JEPAConfig) -> RunPaths:
    run_name = cfg.runtime.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.runtime.output_root) / cfg.runtime.experiment_name / run_name
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        logs_dir=run_dir / "logs",
        metrics_path=run_dir / "metrics.jsonl",
        log_path=run_dir / "logs" / "train.log",
        config_path=run_dir / "config.resolved.yaml",
        state_path=run_dir / "state.json",
    )


def ensure_run_directory(paths: RunPaths, cfg: JEPAConfig) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    if (
        cfg.checkpoint.resume == "none"
        and not cfg.runtime.allow_existing_run_dir
        and (paths.state_path.exists() or (paths.checkpoints_dir / "latest.pt").exists())
    ):
        raise RuntimeError(
            f"Run directory {paths.run_dir} already contains state or checkpoints. "
            "Set checkpoint.resume=auto/<path> to resume, or override runtime.run_name."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def amp_context(precision: str, device: torch.device):
    if device.type == "cuda" and precision == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16-mixed":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def scaler_enabled(precision: str, device: torch.device) -> bool:
    return device.type == "cuda" and precision == "16-mixed"


def build_optimizer(cfg: JEPAConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
    )


def build_scheduler(
    cfg: JEPAConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.scheduler.name == "none":
        return None
    if cfg.scheduler.name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.t_0,
            T_mult=cfg.scheduler.t_mult,
            eta_min=cfg.scheduler.eta_min,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_metadata(cfg: JEPAConfig) -> dict[str, str]:
    return {
        "family": cfg.model.family,
        "masking_strategy": cfg.masking.strategy,
        "regularizer_name": cfg.regularizer.name,
        "target_encoder_kind": cfg.target_encoder.kind,
        "projector_kind": cfg.projector.kind,
        "augment_profile": cfg.augment.profile,
    }


def _state_payload(
    *,
    status: str,
    cfg_hash: str,
    signature: dict[str, Any],
    run_id: str | None,
    git_sha: str | None,
    device: torch.device,
    global_step: int,
    images_seen: int,
    batches_seen: int,
    data_cursor: dict[str, int],
    health: dict[str, Any],
    last_checkpoint: str | None,
    resume_source: str | None,
    metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "status": status,
        "config_hash": cfg_hash,
        "resume_signature": signature,
        "wandb_run_id": run_id,
        "git_sha": git_sha,
        "host": host_name(),
        "device": str(device),
        "global_step": global_step,
        "images_seen": images_seen,
        "batches_seen": batches_seen,
        "data_cursor": data_cursor,
        "health": health,
        "last_checkpoint": last_checkpoint,
        "resume_source": resume_source,
        **metadata,
    }


def _checkpoint_payload(
    *,
    model: TimmPathologySpatialJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    cfg_hash: str,
    signature: dict[str, Any],
    global_step: int,
    images_seen: int,
    batches_seen: int,
    data_cursor: dict[str, int],
    meter_state: dict[str, Any],
    run_id: str | None,
    metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "model_training_state": model.get_training_state(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "rng_state": _rng_state(),
        "config_hash": cfg_hash,
        "resume_signature": signature,
        "global_step": global_step,
        "images_seen": images_seen,
        "batches_seen": batches_seen,
        "data_cursor": data_cursor,
        "meter_state": meter_state,
        "wandb_run_id": run_id,
        **metadata,
    }


def _averaged_metric_inputs(log_payload: dict[str, float]) -> dict[str, float]:
    keys = (
        "train/loss",
        "train/mse",
        "train/regularizer",
        "train/sigreg",
        "train/gaussian_sketch",
    )
    return {key: log_payload[key] for key in keys if key in log_payload}


def _mask_metrics(mask_metadata: dict[str, float | int]) -> dict[str, float]:
    return {f"mask/{key}": float(value) for key, value in mask_metadata.items()}


def _representation_metrics(cfg: JEPAConfig, output: ForwardOutput) -> dict[str, float]:
    return compute_representation_metrics(
        output.z_pred.detach(),
        output.z_tgt.detach(),
        regularizer_embeddings=output.regularizer_embeddings.detach(),
        collapse_threshold=cfg.loss.collapse_threshold,
        projection_count=min(cfg.regularizer.num_projections, 64),
    )


def _handle_nonfinite_grad_norm(
    *,
    grad_norm: float,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    global_step: int,
) -> bool:
    if math.isfinite(grad_norm):
        return False
    if scaler.is_enabled():
        grad_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        logger.warning(
            "Skipping optimizer step %d due to non-finite gradients under AMP "
            "(grad_norm=%s, grad_scale=%s).",
            global_step + 1,
            grad_norm,
            grad_scale,
        )
        return True
    raise RuntimeError(f"Non-finite gradient norm at step {global_step + 1}: {grad_norm}")


def train(
    cfg: JEPAConfig,
    resolved_config: dict[str, Any],
    *,
    paths: RunPaths,
    repo_root: Path,
) -> dict[str, Any]:
    seed_everything(cfg.runtime.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.runtime.matmul_precision)
    device = resolve_device(cfg.runtime.device)
    git_sha = resolve_git_sha(repo_root)
    manifest = discover_tar_manifest(cfg.data)
    signature = compute_resume_signature(resolved_config, manifest_metadata(manifest))
    cfg_hash = config_hash(resolved_config)
    metadata = _run_metadata(cfg)

    transform = build_train_transform(cfg.augment)
    model = TimmPathologySpatialJEPA(cfg, mask_seed=cfg.runtime.seed).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled(cfg.runtime.precision, device))
    smoother = MetricSmoother(cfg.logging.avg_window)

    start_tar_index = 0
    start_member_index = 0
    global_step = 0
    images_seen = 0
    batches_seen = 0
    resume_source: str | None = None
    resume_run_id: str | None = None

    resume_path = resolve_resume_path(cfg.checkpoint.resume, paths.run_dir)
    if cfg.checkpoint.resume != "none" and resume_path is None:
        raise FileNotFoundError(
            f"Requested resume mode {cfg.checkpoint.resume!r}, but no checkpoint was found for run {paths.run_dir}."
        )
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device="cpu")
        if checkpoint["resume_signature"]["hash"] != signature["hash"]:
            raise RuntimeError(
                "Resume checkpoint signature does not match the current JEPA config/data manifest."
            )
        model.load_state_dict(checkpoint["model"])
        model.load_training_state(checkpoint.get("model_training_state"))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        _restore_rng_state(checkpoint["rng_state"])
        global_step = int(checkpoint["global_step"])
        images_seen = int(checkpoint["images_seen"])
        batches_seen = int(checkpoint["batches_seen"])
        start_tar_index = int(checkpoint["data_cursor"]["tar_index"])
        start_member_index = int(checkpoint["data_cursor"]["member_index"])
        smoother.load_state_dict(checkpoint.get("meter_state", {}))
        resume_source = str(resume_path)
        resume_run_id = checkpoint.get("wandb_run_id")
        logger.info(
            "Resuming JEPA run from %s at step=%d images_seen=%d cursor=%s",
            resume_path,
            global_step,
            images_seen,
            checkpoint["data_cursor"],
        )

    dataset = TarImageStream(
        manifest=manifest,
        transform=transform,
        start_tar_index=start_tar_index,
        start_member_index=start_member_index,
        cache_in_ram=cfg.data.cache_in_ram,
        local_cache_dir=cfg.data.local_cache_dir,
        download_cmd=cfg.data.download_cmd,
        prefetch_depth=cfg.data.prefetch_depth,
        prefetch_timeout_sec=cfg.data.prefetch_timeout_sec,
        extensions=cfg.data.extensions,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.runtime.batch_size,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_image_samples,
    )
    loader_iter = iter(loader)
    metric_writer = JsonlMetricWriter(paths.metrics_path)
    tracker = build_tracker(
        cfg.wandb,
        run_dir=paths.run_dir,
        resolved_config=resolved_config,
        run_id=resume_run_id,
    )
    model.train()

    logger.info(
        "Starting JEPA training family=%s model=%s device=%s steps=%d batch_size=%d resume=%s",
        cfg.model.family,
        cfg.model.model_name,
        device,
        cfg.runtime.max_steps,
        cfg.runtime.batch_size,
        resume_source or "none",
    )
    logger.info(
        "Run metadata: host=%s git_sha=%s manifest_tars=%d config_hash=%s metadata=%s",
        host_name(),
        git_sha or "unknown",
        len(manifest),
        cfg_hash,
        json.dumps(metadata, sort_keys=True),
    )

    latest_state = _state_payload(
        status="running",
        cfg_hash=cfg_hash,
        signature=signature,
        run_id=tracker.run_id,
        git_sha=git_sha,
        device=device,
        global_step=global_step,
        images_seen=images_seen,
        batches_seen=batches_seen,
        data_cursor={"tar_index": start_tar_index, "member_index": start_member_index},
        health=dataset.health.to_dict(),
        last_checkpoint=resume_source,
        resume_source=resume_source,
        metadata=metadata,
    )
    write_state_file(paths.run_dir, latest_state)

    final_status = "completed"
    last_checkpoint_path: Path | None = None
    caught_exception: Exception | None = None
    try:
        while global_step < cfg.runtime.max_steps:
            optimizer.zero_grad(set_to_none=True)
            step_metric_totals: dict[str, float] = {"train/loss": 0.0}
            step_images = 0
            step_batches = 0
            step_cursor = {"tar_index": start_tar_index, "member_index": start_member_index}
            step_model_metrics: dict[str, float] = {}
            step_mask_metrics: dict[str, float] = {}
            step_data_time = 0.0
            step_compute_start = time.perf_counter()

            for _micro_step in range(cfg.runtime.grad_accumulation_steps):
                fetch_start = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    batch = None
                step_data_time += time.perf_counter() - fetch_start
                if batch is None:
                    break

                images = batch["images"].to(device, non_blocking=device.type == "cuda")
                step_cursor = {
                    "tar_index": int(batch["last_cursor"]["tar_index"]),
                    "member_index": int(batch["last_cursor"]["member_index"]),
                }
                step_images += int(images.shape[0])
                step_batches += 1

                with amp_context(cfg.runtime.precision, device):
                    forward_output = model(images)
                    loss, loss_components = compute_losses(
                        forward_output,
                        cfg.loss,
                        cfg.regularizer,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at step {global_step + 1}: {loss.item()}")

                scaled_loss = loss / cfg.runtime.grad_accumulation_steps
                scaler.scale(scaled_loss).backward()

                step_metric_totals["train/loss"] += float(loss.detach().item())
                for metric_name, metric_value in loss_components.items():
                    step_metric_totals.setdefault(metric_name, 0.0)
                    step_metric_totals[metric_name] += float(metric_value.detach().item())
                step_model_metrics = _representation_metrics(cfg, forward_output)
                step_mask_metrics = _mask_metrics(forward_output.mask_metadata)

            if step_batches == 0:
                logger.info("JEPA stream exhausted after %d optimizer step(s).", global_step)
                break

            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.runtime.grad_clip_norm)
            )
            if _handle_nonfinite_grad_norm(
                grad_norm=grad_norm,
                optimizer=optimizer,
                scaler=scaler,
                global_step=global_step,
            ):
                continue

            scaler.step(optimizer)
            scaler.update()
            ema_drift = model.update_target_encoder(cfg.target_encoder.ema_momentum)
            if scheduler is not None:
                scheduler.step(global_step + 1)
            global_step += 1
            images_seen += step_images
            batches_seen += step_batches
            batch_time = time.perf_counter() - step_compute_start
            checkpoint_time = 0.0

            averaged_loss_metrics = {
                key: value / step_batches for key, value in step_metric_totals.items()
            }
            lr_value = float(optimizer.param_groups[0]["lr"])
            grad_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
            param_norm = parameter_l2_norm(model.parameters())
            instantaneous_metrics = {
                **averaged_loss_metrics,
                "optim/lr": lr_value,
                "optim/grad_norm": grad_norm,
                "optim/param_norm": param_norm,
                "optim/grad_scale": grad_scale,
                "perf/images_per_sec": step_images / max(batch_time, 1e-8),
                "perf/batch_time_ms": batch_time * 1000.0,
                "perf/data_time_ms": step_data_time * 1000.0,
                "perf/checkpoint_time_ms": checkpoint_time,
                "data/images_seen": float(images_seen),
                "data/batches_seen": float(batches_seen),
                "data/tar_index": float(step_cursor["tar_index"]),
                "data/corrupt_images": float(dataset.health.corrupt_images),
                "data/read_errors": float(dataset.health.read_errors),
                "data/skipped_samples": float(dataset.health.skipped_samples),
                "data/prefetch_stalls": float(dataset.health.prefetch_stalls),
                "target_encoder/ema_drift": ema_drift,
                **step_model_metrics,
                **step_mask_metrics,
            }
            if "train/sigreg" not in instantaneous_metrics:
                instantaneous_metrics["train/sigreg"] = 0.0
            averaged_metrics = smoother.update(_averaged_metric_inputs(instantaneous_metrics))
            log_payload = {**instantaneous_metrics, **averaged_metrics}

            if global_step % cfg.checkpoint.save_every_steps == 0 or global_step == cfg.runtime.max_steps:
                checkpoint_start = time.perf_counter()
                checkpoint_payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg_hash=cfg_hash,
                    signature=signature,
                    global_step=global_step,
                    images_seen=images_seen,
                    batches_seen=batches_seen,
                    data_cursor=step_cursor,
                    meter_state=smoother.state_dict(),
                    run_id=tracker.run_id,
                    metadata=metadata,
                )
                step_checkpoint_path, latest_checkpoint_path = save_checkpoint_bundle(
                    checkpoint_dir=paths.checkpoints_dir,
                    payload=checkpoint_payload,
                    global_step=global_step,
                    keep_last_k=cfg.checkpoint.keep_last_k,
                )
                checkpoint_time = (time.perf_counter() - checkpoint_start) * 1000.0
                log_payload["perf/checkpoint_time_ms"] = checkpoint_time
                last_checkpoint_path = latest_checkpoint_path
                tracker.log_checkpoint(step_checkpoint_path, aliases=["latest", f"step-{global_step}"])

            if global_step % cfg.logging.log_every_steps == 0 or global_step == 1:
                tracker.log(log_payload, step=global_step)
                metric_writer.write({"step": global_step, **log_payload})
                logger.info("Step %d metrics: %s", global_step, json.dumps(log_payload, sort_keys=True))

            latest_state = _state_payload(
                status="running",
                cfg_hash=cfg_hash,
                signature=signature,
                run_id=tracker.run_id,
                git_sha=git_sha,
                device=device,
                global_step=global_step,
                images_seen=images_seen,
                batches_seen=batches_seen,
                data_cursor=step_cursor,
                health=dataset.health.to_dict(),
                last_checkpoint=str(last_checkpoint_path) if last_checkpoint_path else resume_source,
                resume_source=resume_source,
                metadata=metadata,
            )
            write_state_file(paths.run_dir, latest_state)
    except Exception as exc:
        final_status = "failed"
        caught_exception = exc
    finally:
        latest_state["status"] = final_status
        write_state_file(paths.run_dir, latest_state)

    if final_status == "completed" and global_step > 0 and (
        last_checkpoint_path is None or last_checkpoint_path.stem != f"step_{global_step:08d}"
    ):
        checkpoint_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg_hash=cfg_hash,
            signature=signature,
            global_step=global_step,
            images_seen=images_seen,
            batches_seen=batches_seen,
            data_cursor=latest_state["data_cursor"],
            meter_state=smoother.state_dict(),
            run_id=tracker.run_id,
            metadata=metadata,
        )
        _step_checkpoint_path, latest_checkpoint_path = save_checkpoint_bundle(
            checkpoint_dir=paths.checkpoints_dir,
            payload=checkpoint_payload,
            global_step=global_step,
            keep_last_k=cfg.checkpoint.keep_last_k,
        )
        last_checkpoint_path = latest_checkpoint_path
        latest_state["last_checkpoint"] = str(latest_checkpoint_path)
        write_state_file(paths.run_dir, latest_state)

    tracker.finish(status=final_status)
    if caught_exception is not None:
        raise caught_exception

    summary = {
        "status": final_status,
        "global_step": global_step,
        "images_seen": images_seen,
        "batches_seen": batches_seen,
        "run_dir": str(paths.run_dir),
        "last_checkpoint": str(last_checkpoint_path) if last_checkpoint_path else None,
        "health": dataset.health.to_dict(),
        "wandb_run_id": tracker.run_id,
        **metadata,
    }
    logger.info("JEPA training summary: %s", json.dumps(summary, sort_keys=True))
    return summary
