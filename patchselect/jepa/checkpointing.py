"""Checkpoint and state management for JEPA training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from patchselect.io_utils import write_json


def compute_resume_signature(
    resolved_config: dict[str, Any],
    manifest_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    critical_config = {
        "data": resolved_config.get("data", {}),
        "augment": resolved_config.get("augment", {}),
        "model": resolved_config.get("model", {}),
        "loss": resolved_config.get("loss", {}),
        "optimizer": resolved_config.get("optimizer", {}),
        "scheduler": resolved_config.get("scheduler", {}),
        "runtime": {
            "batch_size": resolved_config.get("runtime", {}).get("batch_size"),
            "grad_accumulation_steps": resolved_config.get("runtime", {}).get(
                "grad_accumulation_steps"
            ),
            "seed": resolved_config.get("runtime", {}).get("seed"),
            "precision": resolved_config.get("runtime", {}).get("precision"),
        },
    }
    payload = {"critical_config": critical_config, "manifest": manifest_metadata}
    signature_json = json.dumps(payload, sort_keys=True)
    return {
        "payload": payload,
        "hash": hashlib.sha256(signature_json.encode("utf-8")).hexdigest(),
    }


def config_hash(resolved_config: dict[str, Any]) -> str:
    payload = json.dumps(resolved_config, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device)


def resolve_resume_path(resume: str, run_dir: Path) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        path = run_dir / "checkpoints" / "latest.pt"
        return path if path.exists() else None
    candidate = Path(resume)
    return candidate if candidate.exists() else None


def prune_old_step_checkpoints(checkpoint_dir: Path, keep_last_k: int) -> None:
    numbered = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda path: path.name,
    )
    while len(numbered) > keep_last_k:
        numbered.pop(0).unlink(missing_ok=True)


def save_checkpoint_bundle(
    *,
    checkpoint_dir: Path,
    payload: dict[str, Any],
    global_step: int,
    keep_last_k: int,
) -> tuple[Path, Path]:
    step_path = checkpoint_dir / f"step_{global_step:08d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    atomic_torch_save(payload, step_path)
    atomic_torch_save(payload, latest_path)
    prune_old_step_checkpoints(checkpoint_dir, keep_last_k)
    return step_path, latest_path


def write_state_file(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "state.json", payload)
