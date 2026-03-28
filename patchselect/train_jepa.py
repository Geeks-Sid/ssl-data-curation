import glob
import math
import os
import queue
import subprocess
import tarfile
import threading
from io import BytesIO

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset


# ==============================================================================
# 1. PREFETCHING TAR DATALOADER (QUILT Curated Stream)
# ==============================================================================
class PrefetchingTarDataset(IterableDataset):
    def __init__(
        self,
        tar_urls,
        local_dir="/tmp/quilt_tars",
        transform=None,
        download_cmd="cp",
        cache_in_ram=True,
    ):
        self.tar_urls = tar_urls
        self.local_dir = local_dir
        self.transform = transform or T.ToTensor()
        self.download_cmd = download_cmd.split()
        self.cache_in_ram = cache_in_ram

        if self.local_dir and not self.cache_in_ram:
            os.makedirs(self.local_dir, exist_ok=True)
        self.ready_queue = queue.Queue(maxsize=2)

    def download_worker(self):
        for url in self.tar_urls:
            filename = os.path.basename(url)
            if self.cache_in_ram and os.path.isfile(url):
                print(f"\n[Prefetcher] Caching {url} in RAM...")
                with open(url, "rb") as f:
                    self.ready_queue.put((filename, f.read(), None))
                continue

            local_path = os.path.join(self.local_dir, filename)
            print(f"\n[Prefetcher] Downloading {url} -> {local_path}...")
            cmd = self.download_cmd + [url, local_path]
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.ready_queue.put((filename, None, local_path))
        self.ready_queue.put(None)

    def __iter__(self):
        downloader_thread = threading.Thread(target=self.download_worker, daemon=True)
        downloader_thread.start()

        while True:
            tar_item = self.ready_queue.get()
            if tar_item is None:
                break

            tar_name, tar_bytes, local_tar_path = tar_item

            try:
                if tar_bytes is not None:
                    tar_stream = BytesIO(tar_bytes)
                    tar = tarfile.open(fileobj=tar_stream, mode="r:*")
                else:
                    tar = tarfile.open(local_tar_path, "r")

                with tar:
                    for member in tar:
                        if member.isfile() and member.name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tif")
                        ):
                            f = tar.extractfile(member)
                            if f is not None:
                                with Image.open(BytesIO(f.read())) as img:
                                    yield self.transform(img.convert("RGB"))
            except Exception as e:
                print(f"[Dataloader] Error reading {tar_name}: {e}")

            if local_tar_path and os.path.exists(local_tar_path):
                os.remove(local_tar_path)


# ==============================================================================
# 2. THE SIGREG LOSS
# ==============================================================================
def sigreg_loss(embeddings, num_projections=1024, gamma=1.0):
    B, D = embeddings.shape
    directions = torch.randn(D, num_projections, device=embeddings.device)
    directions = F.normalize(directions, p=2, dim=0)
    x = torch.matmul(embeddings, directions)  # (B, Proj)

    x_i = x.unsqueeze(1)
    x_j = x.unsqueeze(0)
    diff2 = (x_i - x_j) ** 2

    term1 = torch.exp(-diff2 / (2 * gamma**2)).mean(dim=(0, 1))
    c = gamma / math.sqrt(gamma**2 + 1)
    term2 = c * torch.exp(-(x**2) / (2 * (gamma**2 + 1))).mean(dim=0)
    term3 = gamma / math.sqrt(gamma**2 + 2)

    loss_per_projection = term1 - (2 * term2) + term3
    return loss_per_projection.mean()


# ==============================================================================
# 3. HEAVY PROJECTOR
# ==============================================================================
class HeavyProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=1024):
        super().__init__()
        # 3-Layer MLP with BatchNorm and GELU, typical for advanced SSL
        # Crucially, the final layer ends with BatchNorm (no activation)
        # to break the LayerNorm hypersphere constraint.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        # x is expected to be shape (Batch, Num_Patches, Dim)
        B, N, D = x.shape
        # Flatten to (B*N, D) for BatchNorm1d
        x_flat = x.reshape(-1, D)
        x_proj = self.net(x_flat)
        # Reshape back to (B, N, Out_Dim)
        return x_proj.reshape(B, N, -1)


# ==============================================================================
# 4. SPATIAL JEPA MODEL (Using timm)
# ==============================================================================
class TimmPathologySpatialJEPA(pl.LightningModule):
    @staticmethod
    def _resolve_predictor_heads(d_model, preferred_nhead=6):
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

    def __init__(
        self,
        model_name="vit_small_patch16_224",
        pred_depth=6,
        pred_num_heads=6,
        proj_hidden_dim=2048,
        proj_out_dim=1024,
        lambda_sigreg=0.1,
        lr=1e-3,
        weight_decay=0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_sigreg = lambda_sigreg

        # --- A. Encoder (Fetched from timm) ---
        # num_classes=0 removes the classification head
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.embed_dim = self.encoder.embed_dim

        # Determine number of patches from timm's patch_embed
        self.num_patches = self.encoder.patch_embed.num_patches

        # --- B. Heavy Projector ---
        self.projector = HeavyProjector(
            in_dim=self.embed_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim
        )

        # --- C. Predictor ---
        # The predictor maps the projected context back to the projected target space
        self.mask_token = nn.Parameter(torch.zeros(1, 1, proj_out_dim))
        self.predictor_num_heads = self._resolve_predictor_heads(
            proj_out_dim, preferred_nhead=pred_num_heads
        )
        self.predictor_pos_proj = nn.Linear(self.embed_dim, proj_out_dim, bias=False)
        pred_layer = nn.TransformerEncoderLayer(
            d_model=proj_out_dim,
            nhead=self.predictor_num_heads,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.predictor = nn.TransformerEncoder(pred_layer, num_layers=pred_depth)

    def generate_random_masks(self, B, num_tokens, device, mask_ratio=0.6):
        num_mask = int(num_tokens * mask_ratio)
        num_keep = num_tokens - num_mask

        noise = torch.rand(B, num_tokens, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]
        return ids_keep, ids_mask

    def forward(self, imgs):
        B = imgs.shape[0]

        # 1. Patchify using timm's module
        x = self.encoder.patch_embed(imgs)  # (B, N, D)

        # Extract positional embeddings (ignoring CLS token pos_embed if present)
        # timm pos_embed shape is usually (1, num_patches + num_prefix_tokens, D)
        num_prefix = getattr(self.encoder, "num_prefix_tokens", 1)
        pos_embed = self.encoder.pos_embed[:, num_prefix:, :]

        # Add spatial positional embeddings
        x = x + pos_embed

        # 2. Masking
        ids_keep, ids_mask = self.generate_random_masks(
            B, self.num_patches, imgs.device
        )

        ctx_patches = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        tgt_patches = torch.gather(
            x, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        # 3. Pass through timm blocks
        # We manually pass the subset of tokens through the sequence of blocks
        for blk in self.encoder.blocks:
            ctx_patches = blk(ctx_patches)
            tgt_patches = blk(tgt_patches)

        ctx_patches = self.encoder.norm(ctx_patches)
        tgt_patches = self.encoder.norm(tgt_patches)

        # 4. Heavy Projector
        z_ctx_proj = self.projector(ctx_patches)
        z_tgt_proj = self.projector(tgt_patches)

        # 5. Predict Targets from Context
        N_tgt = z_tgt_proj.shape[1]
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)

        tgt_pos_embeds = torch.gather(
            pos_embed.expand(B, -1, -1),
            dim=1,
            index=ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        tgt_pos_embeds = self.predictor_pos_proj(tgt_pos_embeds)
        mask_tokens = mask_tokens + tgt_pos_embeds

        pred_input = torch.cat([z_ctx_proj, mask_tokens], dim=1)
        pred_output = self.predictor(pred_input)

        # Extract predictions corresponding to mask tokens
        N_ctx = z_ctx_proj.shape[1]
        z_pred = pred_output[:, N_ctx:, :]

        return z_pred, z_tgt_proj

    def training_step(self, batch, batch_idx):
        imgs = batch[0] if isinstance(batch, list) else batch

        z_pred, z_tgt = self(imgs)

        loss_mse = F.mse_loss(z_pred, z_tgt)

        # Apply SIGReg to flattened targets
        loss_sigreg = sigreg_loss(z_tgt.reshape(-1, z_tgt.shape[-1]))

        loss = loss_mse + (self.hparams.lambda_sigreg * loss_sigreg)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss_mse", loss_mse, on_step=True, prog_bar=True)
        self.log("loss_sigreg", loss_sigreg, on_step=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50
        )
        return [optimizer], [scheduler]


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    os.makedirs("./dummy_data", exist_ok=True)
    tar_urls = glob.glob(
        "D:\\FMIHCS\\ssl-data-curation\\output_multi_full\\global_selection\\final_selection_tars\\*.tar"
    )  # Replace with your QUILT tar list

    transform = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PrefetchingTarDataset(
        tar_urls=tar_urls,
        local_dir="/tmp/quilt_tars",
        transform=transform,
        download_cmd="cp",
        cache_in_ram=True,
    )
    train_loader = DataLoader(dataset, batch_size=8, num_workers=0, pin_memory=True)

    # Now you can swap this for 'vit_base_patch16_224', 'vit_tiny_patch16_224', etc.
    model = TimmPathologySpatialJEPA(
        model_name="vit_small_patch16_224",
        pred_depth=6,
        proj_hidden_dim=2048,  # Heavy hidden dimension
        proj_out_dim=1024,  # Output dim matches ViT-Small embed dim
        lambda_sigreg=0.1,
    )

    callbacks = [
        ModelCheckpoint(dirpath="checkpoints/", save_top_k=3, monitor="train_loss"),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_steps=1000,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    print(f"Initializing TIMM {model.hparams.model_name} + Heavy Projector JEPA...")
    try:
        trainer.fit(model, train_dataloaders=train_loader)
    except Exception as e:
        print(f"Training stopped/failed: {e}")


if __name__ == "__main__":
    main()
