"""Optional cuCIM/CuPy descriptor backend for batched patch extraction."""

from __future__ import annotations

import numpy as np

from patchselect.config import PatchSelectionConfig
from patchselect.constants import BASE_FEATURE_NAMES
from patchselect.descriptors import (
    _HD_BASIS,
    _HD_PINV,
    SlideStats,
    disk,
    resize_mask,
    resize_rgb,
)


def _load_gpu_modules():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cnd
        import cucim  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "descriptor_backend='cucim' requires both 'cupy' and 'cucim' to be installed."
        ) from exc
    return cp, cnd


def _scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _q(values, cp, quantile: float, default: float = 0.0) -> float:
    if int(values.size) == 0:
        return default
    return _scalar(cp.quantile(values, quantile))


def _safe_mean(values, cp) -> float:
    if int(values.size) == 0:
        return 0.0
    return _scalar(values.mean())


def _safe_std(values, cp) -> float:
    if int(values.size) == 0:
        return 0.0
    return _scalar(values.std())


def _normalized_histogram(
    values, cp, bins: int = 4, value_range: tuple[float, float] = (0.0, 1.0)
):
    if int(values.size) == 0:
        return cp.zeros(bins, dtype=cp.float32)
    hist, _ = cp.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(cp.float32)
    total = hist.sum()
    if _scalar(total) <= 0:
        return cp.zeros(bins, dtype=cp.float32)
    return hist / total


def _remove_small_components(mask, min_size: int, cp, cnd):
    labels, count = cnd.label(mask)
    if int(count) == 0:
        return mask.astype(bool)
    sizes = cp.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _component_areas(mask, cp, cnd):
    labels, count = cnd.label(mask)
    if int(count) == 0:
        return cp.empty((0,), dtype=cp.float32)
    sizes = cp.bincount(labels.ravel())[1:]
    return sizes.astype(cp.float32)


def _rgb_to_hsv(rgb_norm, cp):
    maxc = rgb_norm.max(axis=-1)
    minc = rgb_norm.min(axis=-1)
    delta = maxc - minc
    sat = cp.where(maxc > 0, delta / cp.clip(maxc, 1e-6, None), 0.0)
    value = maxc
    return sat.astype(cp.float32), value.astype(cp.float32)


def _rgb_to_gray(rgb_u8, cp):
    rgb = rgb_u8.astype(cp.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def _rgb_to_hd(rgb_norm, cp):
    """2-component IHC deconvolution on GPU: returns (od, h, d, residual)."""
    od = -cp.log(cp.clip(rgb_norm, 1.0 / 255.0, 1.0))
    pinv = cp.asarray(_HD_PINV.T, dtype=cp.float32)   # (3, 2)
    basis = cp.asarray(_HD_BASIS.T, dtype=cp.float32)  # (2, 3)
    hd = od @ pinv                                     # (..., 2)
    reconstructed = hd @ basis                          # (..., 3)
    residual = cp.sqrt(((od - reconstructed) ** 2).sum(axis=-1))
    h = cp.clip(hd[..., 0], 0.0, None)
    d = cp.clip(hd[..., 1], 0.0, None)
    return od, h, d, residual


def _local_std(gray, cnd, cp, size: int = 7):
    mean = cnd.uniform_filter(gray, size=size)
    sq_mean = cnd.uniform_filter(gray * gray, size=size)
    return cp.sqrt(cp.clip(sq_mean - mean * mean, 0.0, None))


def _sobel_mag(gray, cnd, cp):
    gx = cnd.sobel(gray, axis=0)
    gy = cnd.sobel(gray, axis=1)
    return cp.hypot(gx, gy)


def _build_tissue_mask(
    od_sum, sat, value, cfg: PatchSelectionConfig, cp, cnd, foreground_mask=None
):
    mask = (
        (od_sum > cfg.od_tissue_threshold) & (value < cfg.value_tissue_threshold)
    ) | (sat > cfg.sat_tissue_threshold)
    mask = _remove_small_components(mask, cfg.min_component_size, cp, cnd)
    mask = cnd.binary_closing(mask, structure=cp.asarray(disk(1)))
    mask = cnd.binary_fill_holes(mask)
    if foreground_mask is not None:
        mask = mask & foreground_mask.astype(bool)
    return mask.astype(bool)


def _prepare_slide_inputs(
    rgb: np.ndarray,
    foreground_mask: np.ndarray | None,
    cfg: PatchSelectionConfig,
):
    if cfg.slide_stats_size is None:
        return rgb, foreground_mask
    resized_rgb = resize_rgb(rgb, cfg.slide_stats_size)
    resized_mask = (
        resize_mask(foreground_mask, cfg.slide_stats_size)
        if foreground_mask is not None
        else None
    )
    return resized_rgb, resized_mask


def _prepare_patch_inputs(
    patches: list[np.ndarray],
    foreground_masks: list[np.ndarray | None] | None,
    cfg: PatchSelectionConfig,
):
    if cfg.downsample_size is None:
        return patches, foreground_masks
    prepared_patches = [resize_rgb(patch, cfg.downsample_size) for patch in patches]
    if foreground_masks is None:
        return prepared_patches, None
    prepared_masks = [
        None if mask is None else resize_mask(mask, cfg.downsample_size)
        for mask in foreground_masks
    ]
    return prepared_patches, prepared_masks


def compute_slide_stats_cucim(
    rgb: np.ndarray,
    cfg: PatchSelectionConfig,
    foreground_mask: np.ndarray | None = None,
) -> SlideStats:
    cp, cnd = _load_gpu_modules()
    slide_rgb, slide_foreground = _prepare_slide_inputs(rgb, foreground_mask, cfg)
    rgb_norm = cp.clip(
        cp.asarray(slide_rgb, dtype=cp.float32) / 255.0, 1.0 / 255.0, 1.0
    )
    od, h, d, residual = _rgb_to_hd(rgb_norm, cp)
    sat, value = _rgb_to_hsv(rgb_norm, cp)
    od_sum = od.sum(axis=-1)
    foreground_gpu = (
        cp.asarray(slide_foreground.astype(bool))
        if slide_foreground is not None
        else None
    )
    tissue = _build_tissue_mask(
        od_sum, sat, value, cfg, cp, cnd, foreground_mask=foreground_gpu
    )
    if int(tissue.sum()) > 0:
        tissue_mask = tissue
    elif foreground_gpu is not None and int(foreground_gpu.sum()) > 0:
        tissue_mask = foreground_gpu
    else:
        tissue_mask = cp.ones_like(tissue, dtype=bool)
    return SlideStats(
        dab_q05=_q(d[tissue_mask], cp, 0.05),
        dab_q95=_q(d[tissue_mask], cp, 0.95, default=1.0),
        h_q05=_q(h[tissue_mask], cp, 0.05),
        h_q95=_q(h[tissue_mask], cp, 0.95, default=1.0),
        residual_q95=_q(residual[tissue_mask], cp, 0.95, default=1.0),
    )


def compute_patch_descriptors_cucim(
    patches: list[np.ndarray],
    slide_stats: SlideStats,
    cfg: PatchSelectionConfig,
    foreground_masks: list[np.ndarray | None] | None = None,
) -> list[np.ndarray | None]:
    if not patches:
        return []

    cp, cnd = _load_gpu_modules()
    prepared_patches, prepared_masks = _prepare_patch_inputs(
        patches, foreground_masks, cfg
    )
    batch = cp.asarray(np.stack(prepared_patches, axis=0), dtype=cp.float32)
    rgb_norm = cp.clip(batch / 255.0, 1.0 / 255.0, 1.0)
    od, h, d, residual = _rgb_to_hd(rgb_norm, cp)
    gray = _rgb_to_gray(batch, cp)
    sat, value = _rgb_to_hsv(rgb_norm, cp)
    od_sum = od.sum(axis=-1)
    raw_tissue = (
        (od_sum > cfg.od_tissue_threshold) & (value < cfg.value_tissue_threshold)
    ) | (sat > cfg.sat_tissue_threshold)
    mask_batch = None
    if prepared_masks is not None and prepared_masks and prepared_masks[0] is not None:
        mask_batch = cp.asarray(np.stack(prepared_masks, axis=0).astype(np.bool_))

    structure_1 = cp.asarray(disk(1))
    structure_2 = cp.asarray(disk(2))
    h_scale = max(slide_stats.h_q95 - slide_stats.h_q05, 1e-3)
    d_scale = max(slide_stats.dab_q95 - slide_stats.dab_q05, 1e-3)
    r_scale = max(slide_stats.residual_q95, 1e-3)

    descriptors: list[np.ndarray | None] = []
    for index in range(len(prepared_patches)):
        tissue = _remove_small_components(
            raw_tissue[index], cfg.min_component_size, cp, cnd
        )
        tissue = cnd.binary_closing(tissue, structure=structure_1)
        tissue = cnd.binary_fill_holes(tissue)
        if mask_batch is not None:
            tissue = tissue & mask_batch[index]

        tissue_frac = _scalar(tissue.mean())
        if tissue_frac < cfg.tissue_min_fraction:
            descriptors.append(None)
            continue

        h_i = h[index]
        d_i = d[index]
        residual_i = residual[index]
        gray_i = gray[index]
        sat_i = sat[index]
        od_sum_i = od_sum[index]

        hn = cp.clip((h_i - slide_stats.h_q05) / h_scale, 0.0, 1.0)
        dn = cp.clip((d_i - slide_stats.dab_q05) / d_scale, 0.0, 1.0)
        rn = cp.clip(residual_i / r_scale, 0.0, 1.0)

        nuclei = tissue & (hn > cfg.nuclei_h_threshold)
        dab = tissue & (dn > cfg.dab_positive_threshold)
        nuclei_dil_1 = cnd.binary_dilation(nuclei, structure=structure_1)
        nuclei_dil_2 = cnd.binary_dilation(nuclei, structure=structure_2)
        ring = tissue & nuclei_dil_2 & ~nuclei_dil_1
        extra = tissue & ~nuclei_dil_2
        holes = cnd.binary_fill_holes(tissue) & ~tissue

        tissue_pixels = max(_scalar(tissue.sum()), 1.0)
        nuclei_areas = _component_areas(nuclei, cp, cnd)
        dab_areas = _component_areas(dab, cp, cnd)
        lap = cnd.laplace(gray_i)
        grad = _sobel_mag(gray_i, cnd, cp)
        loc_std = _local_std(gray_i, cnd, cp, size=7)
        dark_threshold = _q(gray_i[tissue], cp, 0.10, default=0.20)
        texture_threshold = _q(loc_std[tissue], cp, 0.30, default=0.0)
        fold = tissue & (gray_i < dark_threshold) & (loc_std < texture_threshold)

        features = cp.zeros(len(BASE_FEATURE_NAMES), dtype=cp.float32)
        features[0] = tissue_frac
        features[1] = _safe_mean(od_sum_i[tissue], cp)
        features[2] = _safe_mean(gray_i[tissue], cp)
        features[3] = _safe_mean(sat_i[tissue], cp)
        features[4] = _safe_mean(hn[tissue], cp)
        features[5] = _safe_std(hn[tissue], cp)
        features[6] = _q(hn[tissue], cp, 0.10)
        features[7] = _q(hn[tissue], cp, 0.50)
        features[8] = _q(hn[tissue], cp, 0.90)
        features[9] = _safe_mean(dn[tissue], cp)
        features[10] = _safe_std(dn[tissue], cp)
        features[11] = _q(dn[tissue], cp, 0.10)
        features[12] = _q(dn[tissue], cp, 0.50)
        features[13] = _q(dn[tissue], cp, 0.90)
        features[14] = _safe_mean(rn[tissue], cp)
        features[15] = _q(rn[tissue], cp, 0.90)
        features[16:20] = _normalized_histogram(
            hn[tissue], cp, bins=4, value_range=(0.0, 1.0)
        )
        features[20:24] = _normalized_histogram(
            dn[tissue], cp, bins=4, value_range=(0.0, 1.0)
        )
        features[24] = _safe_mean(dn[dab], cp)
        features[25] = _scalar(nuclei.sum()) / tissue_pixels
        features[26] = len(nuclei_areas) / tissue_pixels
        features[27] = _safe_mean(nuclei_areas, cp) / tissue_pixels
        features[28] = _safe_std(nuclei_areas, cp) / max(
            _safe_mean(nuclei_areas, cp), 1e-6
        )
        features[29] = _scalar((dab & nuclei).sum()) / tissue_pixels
        features[30] = _scalar((dab & ring).sum()) / tissue_pixels
        features[31] = _scalar((dab & extra).sum()) / tissue_pixels
        features[32] = len(dab_areas) / tissue_pixels
        features[33] = _scalar(
            (cnd.binary_dilation(dab, structure=structure_1) ^ dab).sum()
        ) / max(
            _scalar(dab.sum()),
            1.0,
        )
        features[34] = _scalar(cp.log(cp.var(lap) + 1e-6))
        features[35] = _safe_mean(grad[tissue], cp)
        features[36] = _q(grad[tissue], cp, 0.90)
        features[37] = _scalar(holes.sum()) / tissue_pixels
        features[38] = _scalar((tissue & (rn > 0.30)).sum()) / tissue_pixels
        features[39] = _scalar(fold.sum()) / tissue_pixels

        border = cp.zeros_like(tissue, dtype=bool)
        border[: cfg.border_width, :] = True
        border[-cfg.border_width :, :] = True
        border[:, : cfg.border_width] = True
        border[:, -cfg.border_width :] = True
        features[40] = _scalar((tissue & border).sum()) / tissue_pixels
        loc_fracs = cp.asarray(
            [features[29], features[30], features[31]], dtype=cp.float32
        )
        sorted_loc = cp.sort(loc_fracs)[::-1]
        features[41] = (
            _scalar(sorted_loc[0] - sorted_loc[1]) if int(sorted_loc.size) >= 2 else 0.0
        )
        descriptors.append(cp.asnumpy(features).astype(np.float32))

    return descriptors
