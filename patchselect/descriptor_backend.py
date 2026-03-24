"""Descriptor backend dispatch for CPU and optional cuCIM/CuPy execution."""

from __future__ import annotations

import numpy as np

from patchselect.config import PatchSelectionConfig
from patchselect.descriptors import (
    SlideStats,
    compute_patch_descriptors_cpu,
    compute_slide_stats,
)


def backend_is_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "cucim":
        try:
            import cupy  # noqa: F401
            import cucim  # noqa: F401
        except Exception:
            return False
        return True
    raise ValueError(f"Unsupported descriptor backend: {name}")


def available_descriptor_backends() -> list[str]:
    backends = ["cpu"]
    if backend_is_available("cucim"):
        backends.append("cucim")
    return backends


def compute_slide_stats_for_backend(
    rgb: np.ndarray,
    cfg: PatchSelectionConfig,
    foreground_mask: np.ndarray | None = None,
) -> SlideStats:
    if cfg.descriptor_backend == "cpu":
        return compute_slide_stats(rgb, cfg, foreground_mask=foreground_mask)
    if cfg.descriptor_backend == "cucim":
        from patchselect.descriptors_cucim import compute_slide_stats_cucim

        return compute_slide_stats_cucim(rgb, cfg, foreground_mask=foreground_mask)
    raise ValueError(f"Unsupported descriptor backend: {cfg.descriptor_backend}")


def compute_patch_descriptors_for_backend(
    patches: list[np.ndarray],
    slide_stats: SlideStats,
    cfg: PatchSelectionConfig,
    foreground_masks: list[np.ndarray | None] | None = None,
) -> list[np.ndarray | None]:
    if cfg.descriptor_backend == "cpu":
        return compute_patch_descriptors_cpu(
            patches,
            slide_stats,
            cfg,
            foreground_masks=foreground_masks,
        )
    if cfg.descriptor_backend == "cucim":
        from patchselect.descriptors_cucim import compute_patch_descriptors_cucim

        return compute_patch_descriptors_cucim(
            patches,
            slide_stats,
            cfg,
            foreground_masks=foreground_masks,
        )
    raise ValueError(f"Unsupported descriptor backend: {cfg.descriptor_backend}")
