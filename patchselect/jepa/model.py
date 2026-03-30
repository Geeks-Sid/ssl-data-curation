"""JEPA model, masking, and regularizer definitions."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchselect.jepa.config import (
    JEPAConfig,
    LossConfig,
    ProjectorConfig,
    RegularizerConfig,
)

_MAX_REGULARIZER_LOSS = 50.0


@dataclass(slots=True)
class ForwardOutput:
    z_pred: torch.Tensor
    z_tgt: torch.Tensor
    regularizer_embeddings: torch.Tensor
    mask_metadata: dict[str, float | int]


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (Epps-Pulley).

    Ported directly from the original LeWM implementation.
    Uses numerical quadrature with windowed weights to compare the
    empirical characteristic function to a Gaussian reference.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (N, D) flattened patch embeddings."""
        A = torch.randn(embeddings.size(-1), self.num_proj, device=embeddings.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (embeddings @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * embeddings.size(0)
        return statistic.mean()


def gaussian_sketch_regularizer(
    embeddings: torch.Tensor,
    *,
    num_projections: int = 256,
    eps: float = 1e-6,
) -> torch.Tensor:
    dim = embeddings.shape[-1]
    directions = torch.randn(dim, num_projections, device=embeddings.device)
    directions = F.normalize(directions, p=2, dim=0)
    projected = embeddings @ directions
    mean = projected.mean(dim=0)
    centered = projected - mean
    var = centered.pow(2).mean(dim=0).clamp_min(eps)
    standardized = (centered / var.sqrt()).clamp(min=-10.0, max=10.0)
    skew = standardized.pow(3).mean(dim=0)
    kurtosis = standardized.pow(4).mean(dim=0) - 3.0
    loss = (
        mean.pow(2).mean()
        + (var - 1.0).pow(2).mean()
        + skew.pow(2).mean()
        + kurtosis.pow(2).mean()
    )
    return torch.nan_to_num(loss, nan=_MAX_REGULARIZER_LOSS, posinf=_MAX_REGULARIZER_LOSS, neginf=0.0)


class HeavyProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape
        x_flat = x.reshape(-1, embed_dim)
        x_proj = self.net(x_flat)
        return x_proj.reshape(batch_size, num_tokens, -1)


class LayerNormProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape
        x_flat = x.reshape(-1, embed_dim)
        x_proj = self.net(x_flat)
        return x_proj.reshape(batch_size, num_tokens, -1)


class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def build_projector(cfg: ProjectorConfig, in_dim: int) -> nn.Module:
    if cfg.kind == "heavy_bn":
        return HeavyProjector(in_dim, cfg.hidden_dim, cfg.out_dim)
    if cfg.kind == "mlp_ln":
        return LayerNormProjector(in_dim, cfg.hidden_dim, cfg.out_dim)
    return LinearProjector(in_dim, cfg.out_dim)


class TimmPathologySpatialJEPA(nn.Module):
    @staticmethod
    def _resolve_predictor_heads(d_model: int, preferred_nhead: int = 6) -> int:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        max_heads = min(d_model, max(preferred_nhead * 2, 16))
        for distance in range(max_heads):
            for candidate in (preferred_nhead + distance, preferred_nhead - distance):
                if candidate < 1 or candidate > d_model:
                    continue
                if d_model % candidate == 0:
                    return candidate
        return 1

    def __init__(self, cfg: JEPAConfig, *, mask_seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_seed = int(mask_seed)
        self.mask_sampler_step = 0
        self.last_mask_metadata: dict[str, float | int] = {}

        self.encoder = timm.create_model(
            cfg.model.model_name,
            pretrained=cfg.model.pretrained,
            num_classes=0,
        )
        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches
        self.grid_size = self._resolve_grid_size()
        self.projector = build_projector(cfg.projector, self.embed_dim)

        self.target_encoder = None
        self.target_projector = None
        if cfg.target_encoder.kind == "ema":
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_projector = copy.deepcopy(self.projector)
            self.target_encoder.requires_grad_(False)
            self.target_projector.requires_grad_(False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.projector.out_dim))
        self.predictor_num_heads = self._resolve_predictor_heads(
            cfg.projector.out_dim,
            preferred_nhead=cfg.model.pred_num_heads,
        )
        self.predictor_pos_proj = nn.Linear(
            self.embed_dim,
            cfg.projector.out_dim,
            bias=False,
        )
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=cfg.projector.out_dim,
            nhead=self.predictor_num_heads,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=cfg.model.pred_depth,
        )

        # Regularizer module (original Epps-Pulley SIGReg)
        if cfg.regularizer.name == "sigreg":
            self.sigreg = SIGReg(
                knots=cfg.regularizer.sigreg_knots,
                num_proj=cfg.regularizer.num_projections,
            )
        else:
            self.sigreg = None

    def _resolve_grid_size(self) -> tuple[int, int]:
        patch_embed = self.encoder.patch_embed
        grid_size = getattr(patch_embed, "grid_size", None)
        if grid_size is not None:
            return tuple(int(value) for value in grid_size)
        side = int(round(math.sqrt(self.num_patches)))
        if side * side != self.num_patches:
            raise ValueError(
                f"Unable to infer a square patch grid from num_patches={self.num_patches}"
            )
        return (side, side)

    def _extract_pos_embed(self, encoder: nn.Module) -> torch.Tensor:
        num_prefix = getattr(encoder, "num_prefix_tokens", 1)
        return encoder.pos_embed[:, num_prefix:, :]

    def _encode_subset(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        for block in encoder.blocks:
            x = block(x)
        return encoder.norm(x)

    def _new_mask_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.mask_seed + self.mask_sampler_step)
        self.mask_sampler_step += 1
        return generator

    def _rand_uniform(
        self,
        generator: torch.Generator,
        low: float,
        high: float,
    ) -> float:
        return float((high - low) * torch.rand(1, generator=generator).item() + low)

    def _randint(
        self,
        generator: torch.Generator,
        low: int,
        high: int,
    ) -> int:
        if low >= high:
            return int(low)
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    def _generate_random_token_masks(
        self,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        num_mask = int(self.num_patches * self.cfg.masking.mask_ratio)
        num_keep = self.num_patches - num_mask
        noise = torch.rand(self.num_patches, generator=generator)
        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep].to(device)
        ids_mask = ids_shuffle[num_keep:].to(device)
        mask_metadata = {
            "num_targets_used": 1,
            "target_token_count": int(ids_mask.numel()),
            "context_token_count": int(ids_keep.numel()),
            "mask_coverage": float(ids_mask.numel() / max(self.num_patches, 1)),
            "target_block_size_mean": float(ids_mask.numel()),
            "target_block_size_min": float(ids_mask.numel()),
            "target_block_size_max": float(ids_mask.numel()),
        }
        return ids_keep, ids_mask, mask_metadata

    def _generate_block_target_masks(
        self,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        grid_h, grid_w = self.grid_size
        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        block_sizes: list[int] = []
        low_scale, high_scale = self.cfg.masking.target_scale_range
        low_ar, high_ar = self.cfg.masking.aspect_ratio_range

        for _target_index in range(self.cfg.masking.num_targets):
            placed = False
            for _attempt in range(64):
                area = max(
                    1,
                    int(
                        round(
                            self._rand_uniform(generator, low_scale, high_scale)
                            * self.num_patches
                        )
                    ),
                )
                aspect = self._rand_uniform(generator, low_ar, high_ar)
                height = max(1, min(grid_h, int(round(math.sqrt(area * aspect)))))
                width = max(1, min(grid_w, int(round(area / max(height, 1)))))
                if height * width >= self.num_patches - self.cfg.masking.context_min_keep:
                    continue
                top = self._randint(generator, 0, max(grid_h - height, 0))
                left = self._randint(generator, 0, max(grid_w - width, 0))
                candidate = torch.zeros_like(mask)
                candidate[top : top + height, left : left + width] = True
                if (
                    not self.cfg.masking.allow_overlap
                    and (mask & candidate).any()
                ):
                    continue
                combined = mask | candidate
                if (
                    self.num_patches - int(combined.sum().item())
                    < self.cfg.masking.context_min_keep
                ):
                    continue
                mask = combined
                block_sizes.append(int(candidate.sum().item()))
                placed = True
                break
            if not placed:
                break

        if not block_sizes:
            return self._generate_random_token_masks(device, generator)

        flat_mask = mask.reshape(-1)
        ids_mask = torch.nonzero(flat_mask, as_tuple=False).squeeze(1).to(device)
        ids_keep = torch.nonzero(~flat_mask, as_tuple=False).squeeze(1).to(device)
        mask_metadata = {
            "num_targets_used": len(block_sizes),
            "target_token_count": int(ids_mask.numel()),
            "context_token_count": int(ids_keep.numel()),
            "mask_coverage": float(ids_mask.numel() / max(self.num_patches, 1)),
            "target_block_size_mean": float(sum(block_sizes) / len(block_sizes)),
            "target_block_size_min": float(min(block_sizes)),
            "target_block_size_max": float(max(block_sizes)),
        }
        return ids_keep, ids_mask, mask_metadata

    def generate_masks(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        generator = self._new_mask_generator()
        if self.cfg.masking.strategy == "random_tokens":
            ids_keep, ids_mask, metadata = self._generate_random_token_masks(
                device,
                generator,
            )
        else:
            ids_keep, ids_mask, metadata = self._generate_block_target_masks(
                device,
                generator,
            )
        metadata = {
            **metadata,
            "strategy_is_block_targets": int(
                self.cfg.masking.strategy == "block_targets"
            ),
            "mask_sampler_step": self.mask_sampler_step,
        }
        self.last_mask_metadata = metadata
        return ids_keep, ids_mask, metadata

    def has_ema_target(self) -> bool:
        return self.target_encoder is not None and self.target_projector is not None

    def update_target_encoder(self, momentum: float) -> float:
        if not self.has_ema_target():
            return 0.0
        drift_total = 0.0
        drift_count = 0
        with torch.no_grad():
            for online_param, target_param in zip(
                self.encoder.parameters(),
                self.target_encoder.parameters(),
            ):
                drift_total += float(
                    (online_param.detach() - target_param.detach()).abs().mean().item()
                )
                drift_count += 1
                target_param.mul_(momentum).add_(online_param, alpha=1.0 - momentum)
            for online_param, target_param in zip(
                self.projector.parameters(),
                self.target_projector.parameters(),
            ):
                drift_total += float(
                    (online_param.detach() - target_param.detach()).abs().mean().item()
                )
                drift_count += 1
                target_param.mul_(momentum).add_(online_param, alpha=1.0 - momentum)
            for online_buffer, target_buffer in zip(
                self.encoder.buffers(),
                self.target_encoder.buffers(),
            ):
                target_buffer.copy_(online_buffer)
            for online_buffer, target_buffer in zip(
                self.projector.buffers(),
                self.target_projector.buffers(),
            ):
                target_buffer.copy_(online_buffer)
        return drift_total / max(drift_count, 1)

    def get_training_state(self) -> dict[str, object]:
        return {
            "mask_sampler_step": self.mask_sampler_step,
            "last_mask_metadata": self.last_mask_metadata,
        }

    def load_training_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self.mask_sampler_step = int(state.get("mask_sampler_step", 0))
        raw_metadata = state.get("last_mask_metadata", {})
        if isinstance(raw_metadata, dict):
            self.last_mask_metadata = raw_metadata

    def forward(self, imgs: torch.Tensor) -> ForwardOutput:
        batch_size = imgs.shape[0]
        ids_keep, ids_mask, mask_metadata = self.generate_masks(imgs.device)

        x = self.encoder.patch_embed(imgs)
        pos_embed = self._extract_pos_embed(self.encoder)
        x = x + pos_embed
        ctx_patches = torch.gather(
            x,
            dim=1,
            index=ids_keep.view(1, -1, 1).expand(batch_size, -1, self.embed_dim),
        )
        ctx_patches = self._encode_subset(self.encoder, ctx_patches)
        z_ctx_proj = self.projector(ctx_patches)

        if self.has_ema_target():
            with torch.no_grad():
                target_x = self.target_encoder.patch_embed(imgs)
                target_pos_embed = self._extract_pos_embed(self.target_encoder)
                target_x = target_x + target_pos_embed
                tgt_patches = torch.gather(
                    target_x,
                    dim=1,
                    index=ids_mask.view(1, -1, 1).expand(
                        batch_size, -1, self.embed_dim
                    ),
                )
                tgt_patches = self._encode_subset(self.target_encoder, tgt_patches)
                z_tgt_proj = self.target_projector(tgt_patches)
            regularizer_embeddings = z_ctx_proj
        else:
            tgt_patches = torch.gather(
                x,
                dim=1,
                index=ids_mask.view(1, -1, 1).expand(batch_size, -1, self.embed_dim),
            )
            tgt_patches = self._encode_subset(self.encoder, tgt_patches)
            target_projection = self.projector(tgt_patches)
            # Keep the shared target branch stop-grad for the prediction loss while
            # still regularizing the online features.
            z_tgt_proj = target_projection.detach()
            regularizer_embeddings = target_projection

        target_pos_embeds = torch.gather(
            pos_embed.expand(batch_size, -1, -1),
            dim=1,
            index=ids_mask.view(1, -1, 1).expand(batch_size, -1, self.embed_dim),
        )
        target_pos_embeds = self.predictor_pos_proj(target_pos_embeds)
        mask_tokens = self.mask_token.expand(batch_size, ids_mask.numel(), -1)
        pred_input = torch.cat([z_ctx_proj, mask_tokens + target_pos_embeds], dim=1)
        pred_output = self.predictor(pred_input)
        z_pred = pred_output[:, z_ctx_proj.shape[1] :, :]
        return ForwardOutput(
            z_pred=z_pred,
            z_tgt=z_tgt_proj,
            regularizer_embeddings=regularizer_embeddings,
            mask_metadata=mask_metadata,
        )


def compute_regularizer(
    embeddings: torch.Tensor,
    cfg: RegularizerConfig,
    sigreg_module: SIGReg | None = None,
) -> torch.Tensor:
    if cfg.name == "none" or cfg.weight == 0:
        return embeddings.new_zeros(())
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        flat = embeddings.float().reshape(-1, embeddings.shape[-1])
        if cfg.name == "sigreg":
            if sigreg_module is None:
                raise RuntimeError("SIGReg regularizer requested but no SIGReg module provided")
            return sigreg_module(flat)
        return gaussian_sketch_regularizer(
            flat,
            num_projections=cfg.num_projections,
            eps=cfg.eps,
        )


def compute_losses(
    forward_output: ForwardOutput,
    loss_cfg: LossConfig,
    regularizer_cfg: RegularizerConfig,
    sigreg_module: SIGReg | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_cfg.prediction != "mse":
        raise ValueError(f"Unsupported prediction loss: {loss_cfg.prediction}")
    with torch.autocast(device_type=forward_output.z_pred.device.type, enabled=False):
        loss_pred = F.mse_loss(
            forward_output.z_pred.float(),
            forward_output.z_tgt.float(),
        )
        loss_reg = compute_regularizer(
            forward_output.regularizer_embeddings,
            regularizer_cfg,
            sigreg_module=sigreg_module,
        )
        loss_reg = torch.nan_to_num(
            loss_reg,
            nan=0.0,
            posinf=_MAX_REGULARIZER_LOSS,
            neginf=0.0,
        ).clamp(max=_MAX_REGULARIZER_LOSS)
        loss = loss_pred + (regularizer_cfg.weight * loss_reg)
    metrics: dict[str, torch.Tensor] = {
        "train/mse": loss_pred,
        "train/regularizer": loss_reg,
    }
    if regularizer_cfg.name == "sigreg":
        metrics["train/sigreg"] = loss_reg
    if regularizer_cfg.name == "gaussian_sketch":
        metrics["train/gaussian_sketch"] = loss_reg
    return loss, metrics
