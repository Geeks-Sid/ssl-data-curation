"""PyTorch Lightning module for Spatial-JEPA training."""

from __future__ import annotations

import logging
from typing import Any

import lightning as pl
import torch

from patchselect.jepa.config import JEPAConfig
from patchselect.jepa.model import (
    ForwardOutput,
    compute_losses,
)
from patchselect.jepa.metrics import (
    compute_representation_metrics,
    parameter_l2_norm,
)

logger = logging.getLogger(__name__)


def _should_emit_diagnostics(step: int, cfg: JEPAConfig) -> bool:
    if cfg.logging.metrics_mode == "full":
        return True
    if cfg.logging.diagnostics_every_steps <= 0:
        return False
    max_steps = cfg.trainer.max_steps
    return step == 1 or step == max_steps or step % cfg.logging.diagnostics_every_steps == 0


def _compute_diagnostics(output: ForwardOutput, model: torch.nn.Module, cfg: JEPAConfig) -> dict[str, float]:
    diag: dict[str, float] = {}

    rep_metrics = compute_representation_metrics(
        output.z_pred.detach(),
        output.z_tgt.detach(),
        regularizer_embeddings=output.regularizer_embeddings.detach() if output.regularizer_embeddings is not None else None,
        collapse_threshold=cfg.loss.collapse_threshold,
        projection_count=min(cfg.regularizer.num_projections, 64),
    )
    diag.update(rep_metrics)

    for key, value in output.mask_metadata.items():
        diag[f"mask/{key}"] = float(value)

    diag["optim/param_norm"] = parameter_l2_norm(model.parameters())

    return diag


def spatial_jepa_forward(self: pl.LightningModule, batch: dict[str, Any], stage: str, cfg: JEPAConfig) -> dict[str, Any]:
    """Forward pass to inject into stable_pretraining Module."""
    images = batch["images"]

    # Forward pass
    forward_output: ForwardOutput = self.model(images)

    # Compute losses
    sigreg_module = getattr(self, "sigreg", None)
    loss, loss_components = compute_losses(
        forward_output,
        cfg.loss,
        cfg.regularizer,
        sigreg_module=sigreg_module,
    )

    # Handle non-finite loss (fallback, expecting stable_pretraining might mitigate this)
    if not torch.isfinite(loss):
        logger.warning(
            "Non-finite loss at step %d, returning zero loss.",
            self.global_step,
        )
        loss = loss.new_zeros((), requires_grad=True)

    # Log core metrics
    losses_dict = {
        f"{stage}/{k}" if not k.startswith(f"{stage}/") else k: v.detach()
        for k, v in loss_components.items()
    }
    losses_dict[f"{stage}/loss"] = loss.detach()
    self.log_dict(losses_dict, on_step=True, prog_bar=True, sync_dist=True)

    # Periodic diagnostics
    step = self.global_step + 1
    if _should_emit_diagnostics(step, cfg):
        diag = _compute_diagnostics(forward_output, self.model, cfg)
        self.log_dict(diag, on_step=True, sync_dist=True)

    # stable_pretraining logic expects a return dictionary with "loss" at minimum
    return {"loss": loss, **losses_dict}
