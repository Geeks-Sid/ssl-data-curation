"""JEPA model and loss definitions."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchselect.jepa.config import LossConfig, ModelConfig


def sigreg_loss(
    embeddings: torch.Tensor,
    *,
    num_projections: int = 1024,
    gamma: float = 1.0,
) -> torch.Tensor:
    _batch_size, dim = embeddings.shape
    directions = torch.randn(dim, num_projections, device=embeddings.device)
    directions = F.normalize(directions, p=2, dim=0)
    x = torch.matmul(embeddings, directions)

    diff2 = (x.unsqueeze(1) - x.unsqueeze(0)) ** 2
    term1 = torch.exp(-diff2 / (2 * gamma**2)).mean(dim=(0, 1))
    c = gamma / math.sqrt(gamma**2 + 1)
    term2 = c * torch.exp(-(x**2) / (2 * (gamma**2 + 1))).mean(dim=0)
    term3 = gamma / math.sqrt(gamma**2 + 2)
    loss_per_projection = term1 - (2 * term2) + term3
    return loss_per_projection.mean()


class HeavyProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 1024) -> None:
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

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0,
        )
        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches
        self.projector = HeavyProjector(
            in_dim=self.embed_dim,
            hidden_dim=cfg.proj_hidden_dim,
            out_dim=cfg.proj_out_dim,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.proj_out_dim))
        self.predictor_num_heads = self._resolve_predictor_heads(
            cfg.proj_out_dim,
            preferred_nhead=cfg.pred_num_heads,
        )
        self.predictor_pos_proj = nn.Linear(self.embed_dim, cfg.proj_out_dim, bias=False)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=cfg.proj_out_dim,
            nhead=self.predictor_num_heads,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=cfg.pred_depth)

    def generate_random_masks(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_mask = int(num_tokens * self.cfg.mask_ratio)
        num_keep = num_tokens - num_mask
        noise = torch.rand(batch_size, num_tokens, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]
        return ids_keep, ids_mask

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = imgs.shape[0]
        x = self.encoder.patch_embed(imgs)
        num_prefix = getattr(self.encoder, "num_prefix_tokens", 1)
        pos_embed = self.encoder.pos_embed[:, num_prefix:, :]
        x = x + pos_embed

        ids_keep, ids_mask = self.generate_random_masks(
            batch_size,
            self.num_patches,
            imgs.device,
        )
        ctx_patches = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        tgt_patches = torch.gather(
            x,
            dim=1,
            index=ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )

        for block in self.encoder.blocks:
            ctx_patches = block(ctx_patches)
            tgt_patches = block(tgt_patches)

        ctx_patches = self.encoder.norm(ctx_patches)
        tgt_patches = self.encoder.norm(tgt_patches)
        z_ctx_proj = self.projector(ctx_patches)
        z_tgt_proj = self.projector(tgt_patches)

        num_target_tokens = z_tgt_proj.shape[1]
        mask_tokens = self.mask_token.expand(batch_size, num_target_tokens, -1)
        tgt_pos_embeds = torch.gather(
            pos_embed.expand(batch_size, -1, -1),
            dim=1,
            index=ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        tgt_pos_embeds = self.predictor_pos_proj(tgt_pos_embeds)
        mask_tokens = mask_tokens + tgt_pos_embeds

        pred_input = torch.cat([z_ctx_proj, mask_tokens], dim=1)
        pred_output = self.predictor(pred_input)
        num_context_tokens = z_ctx_proj.shape[1]
        z_pred = pred_output[:, num_context_tokens:, :]
        return z_pred, z_tgt_proj


def compute_losses(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    cfg: LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_mse = F.mse_loss(z_pred, z_tgt)
    loss_sigreg = sigreg_loss(
        z_tgt.reshape(-1, z_tgt.shape[-1]),
        num_projections=cfg.sigreg_num_projections,
        gamma=cfg.sigreg_gamma,
    )
    loss = loss_mse + (cfg.lambda_sigreg * loss_sigreg)
    return loss, {"train/mse": loss_mse, "train/sigreg": loss_sigreg}
