"""Metric utilities for JEPA training."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


class SlidingAverageMeter:
    def __init__(self, window: int) -> None:
        self.window = int(window)
        self.values: deque[float] = deque(maxlen=self.window)

    def update(self, value: float) -> None:
        self.values.append(float(value))

    def average(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def state_dict(self) -> dict[str, object]:
        return {"window": self.window, "values": list(self.values)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.window = int(state["window"])
        self.values = deque((float(v) for v in state["values"]), maxlen=self.window)


class MetricSmoother:
    def __init__(self, window: int) -> None:
        self.window = int(window)
        self._meters: dict[str, SlidingAverageMeter] = {}

    def update(self, metrics: dict[str, float]) -> dict[str, float]:
        averaged: dict[str, float] = {}
        for name, value in metrics.items():
            meter = self._meters.setdefault(name, SlidingAverageMeter(self.window))
            meter.update(float(value))
            if name.startswith("train/"):
                averaged[f"{name}_avg_{self.window}"] = meter.average()
        return averaged

    def state_dict(self) -> dict[str, object]:
        return {
            "window": self.window,
            "meters": {name: meter.state_dict() for name, meter in self._meters.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.window = int(state.get("window", self.window))
        meters = state.get("meters", {})
        self._meters = {}
        for name, meter_state in meters.items():
            meter = SlidingAverageMeter(self.window)
            meter.load_state_dict(meter_state)
            self._meters[name] = meter


class JsonlMetricWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _projection_moment_stats(
    embeddings: torch.Tensor,
    *,
    num_projections: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    dim = embeddings.shape[-1]
    directions = torch.randn(dim, num_projections, device=embeddings.device)
    directions = F.normalize(directions, p=2, dim=0)
    projected = embeddings @ directions
    mean = projected.mean(dim=0)
    centered = projected - mean
    var = centered.pow(2).mean(dim=0).clamp_min(eps)
    standardized = centered / var.sqrt()
    skew = standardized.pow(3).mean(dim=0)
    kurtosis = standardized.pow(4).mean(dim=0) - 3.0
    proxy = (
        mean.abs().mean()
        + (var - 1.0).abs().mean()
        + skew.abs().mean()
        + kurtosis.abs().mean()
    )
    return {
        "projection_mean_abs": mean.abs().mean().item(),
        "projection_std_mean": var.sqrt().mean().item(),
        "projection_skew_abs": skew.abs().mean().item(),
        "projection_kurtosis_abs": kurtosis.abs().mean().item(),
        "projection_normality_proxy": proxy.item(),
    }


def compute_representation_metrics(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    regularizer_embeddings: torch.Tensor,
    collapse_threshold: float,
    projection_count: int = 32,
) -> dict[str, float]:
    pred_flat = z_pred.reshape(-1, z_pred.shape[-1])
    tgt_flat = z_tgt.reshape(-1, z_tgt.shape[-1])
    reg_flat = regularizer_embeddings.reshape(-1, regularizer_embeddings.shape[-1])

    pred_norm = pred_flat.norm(dim=-1).mean().item()
    target_norm = tgt_flat.norm(dim=-1).mean().item()
    pred_std = pred_flat.std(dim=0)
    target_std = tgt_flat.std(dim=0)
    reg_var = reg_flat.var(dim=0, unbiased=False)
    collapse_frac = (
        ((pred_std < collapse_threshold) | (target_std < collapse_threshold))
        .float()
        .mean()
        .item()
    )
    pred_target_cosine = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean().item()

    centered = reg_flat - reg_flat.mean(dim=0, keepdim=True)
    if centered.shape[0] > 1:
        covariance = centered.T @ centered / float(centered.shape[0] - 1)
        offdiag = covariance - torch.diag(torch.diag(covariance))
        covariance_offdiag_energy = offdiag.pow(2).mean().item()
    else:
        covariance_offdiag_energy = 0.0
    isotropy_score = (reg_var.min() / reg_var.mean().clamp_min(1e-6)).item()
    projection_stats = _projection_moment_stats(
        reg_flat,
        num_projections=max(4, int(projection_count)),
    )

    return {
        "model/pred_target_cosine": pred_target_cosine,
        "model/pred_norm": pred_norm,
        "model/target_norm": target_norm,
        "model/pred_std_mean": pred_std.mean().item(),
        "model/target_std_mean": target_std.mean().item(),
        "model/collapse_frac": collapse_frac,
        "model/variance_floor": reg_var.min().item(),
        "model/covariance_offdiag_energy": covariance_offdiag_energy,
        "model/isotropy_score": isotropy_score,
        "model/projection_mean_abs": projection_stats["projection_mean_abs"],
        "model/projection_std_mean": projection_stats["projection_std_mean"],
        "model/projection_skew_abs": projection_stats["projection_skew_abs"],
        "model/projection_kurtosis_abs": projection_stats[
            "projection_kurtosis_abs"
        ],
        "model/projection_normality_proxy": projection_stats[
            "projection_normality_proxy"
        ],
    }


def parameter_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().pow(2).sum().item())
    return total**0.5
