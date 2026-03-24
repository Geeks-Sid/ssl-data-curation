"""Cheap stain-aware patch descriptors for IHC patch selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from patchselect.config import PatchSelectionConfig
from patchselect.constants import BASE_FEATURE_NAMES, FEATURE_NAMES


RGB_FROM_HED = np.array(
    [
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.776],
    ],
    dtype=np.float32,
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)


@dataclass(slots=True)
class SlideStats:
    dab_q05: float
    dab_q95: float
    h_q05: float
    h_q95: float
    e_q95: float


def disk(radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    image = Image.fromarray(rgb)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def q(values: np.ndarray, quantile: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.quantile(values, quantile))


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.std())


def normalized_histogram(
    values: np.ndarray,
    bins: int = 4,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total <= 0:
        return np.zeros(bins, dtype=np.float32)
    return hist / total


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labeled]


def component_areas(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    sizes = np.bincount(labeled.ravel())[1:]
    return sizes.astype(np.float32)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    sat = np.where(maxc > 0, delta / np.clip(maxc, 1e-6, None), 0.0)
    value = maxc
    return sat.astype(np.float32), value.astype(np.float32)


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32) / 255.0, 1.0 / 255.0, 1.0)
    od = -np.log(rgb)
    return od @ HED_FROM_RGB.T


def build_tissue_mask(
    od_sum: np.ndarray,
    sat: np.ndarray,
    value: np.ndarray,
    cfg: PatchSelectionConfig,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = (
        (od_sum > cfg.od_tissue_threshold) & (value < cfg.value_tissue_threshold)
    ) | (sat > cfg.sat_tissue_threshold)
    mask = remove_small_components(mask, cfg.min_component_size)
    mask = ndimage.binary_closing(mask, structure=disk(1))
    mask = ndimage.binary_fill_holes(mask)
    if foreground_mask is not None:
        mask = mask & foreground_mask.astype(bool)
    return mask.astype(bool)


def normalize_channel(channel: np.ndarray, low: float, high: float) -> np.ndarray:
    scale = max(high - low, 1e-3)
    return np.clip((channel - low) / scale, 0.0, 1.0)


def compute_slide_stats(
    rgb: np.ndarray,
    cfg: PatchSelectionConfig,
    foreground_mask: np.ndarray | None = None,
) -> SlideStats:
    if cfg.slide_stats_size is not None:
        slide_rgb = resize_rgb(rgb, cfg.slide_stats_size)
        slide_foreground = (
            resize_mask(foreground_mask, cfg.slide_stats_size)
            if foreground_mask is not None
            else None
        )
    else:
        slide_rgb = rgb
        slide_foreground = foreground_mask
    hed = rgb_to_hed(slide_rgb)
    h = np.clip(hed[..., 0], 0.0, None)
    d = np.clip(hed[..., 2], 0.0, None)
    e = np.abs(hed[..., 1])
    sat, value = rgb_to_hsv(slide_rgb)
    od_sum = (
        -np.log(np.clip(slide_rgb.astype(np.float32) / 255.0, 1.0 / 255.0, 1.0))
    ).sum(axis=-1)
    tissue = build_tissue_mask(
        od_sum, sat, value, cfg, foreground_mask=slide_foreground
    )
    if tissue.any():
        tissue_mask = tissue
    elif slide_foreground is not None and slide_foreground.any():
        tissue_mask = slide_foreground
    else:
        tissue_mask = np.ones_like(tissue, dtype=bool)
    return SlideStats(
        dab_q05=q(d[tissue_mask], 0.05),
        dab_q95=q(d[tissue_mask], 0.95, default=1.0),
        h_q05=q(h[tissue_mask], 0.05),
        h_q95=q(h[tissue_mask], 0.95, default=1.0),
        e_q95=q(e[tissue_mask], 0.95, default=1.0),
    )


def border_fraction(mask: np.ndarray, tissue_pixels: float, width: int) -> float:
    if tissue_pixels <= 0:
        return 0.0
    border = np.zeros_like(mask, dtype=bool)
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    return float((mask & border).sum() / tissue_pixels)


def local_std(gray: np.ndarray, size: int = 7) -> np.ndarray:
    mean = ndimage.uniform_filter(gray, size=size)
    sq_mean = ndimage.uniform_filter(gray * gray, size=size)
    return np.sqrt(np.clip(sq_mean - mean * mean, 0.0, None))


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    return np.hypot(gx, gy)


def tile_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def compute_patch_descriptor(
    patch_rgb: np.ndarray,
    slide_stats: SlideStats,
    cfg: PatchSelectionConfig,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if cfg.downsample_size is not None:
        patch_rgb = resize_rgb(patch_rgb, cfg.downsample_size)
    patch_foreground = None
    if foreground_mask is not None:
        patch_foreground = (
            resize_mask(foreground_mask, cfg.downsample_size)
            if cfg.downsample_size is not None
            else foreground_mask
        )
    rgb = patch_rgb.astype(np.float32)
    hed = rgb_to_hed(rgb)
    h = np.clip(hed[..., 0], 0.0, None)
    d = np.clip(hed[..., 2], 0.0, None)
    e = np.abs(hed[..., 1])
    gray = rgb_to_gray(patch_rgb)
    sat, value = rgb_to_hsv(patch_rgb)
    od_sum = (-np.log(np.clip(rgb / 255.0, 1.0 / 255.0, 1.0))).sum(axis=-1)
    tissue = build_tissue_mask(
        od_sum, sat, value, cfg, foreground_mask=patch_foreground
    )
    tissue_frac = float(tissue.mean())
    if tissue_frac < cfg.tissue_min_fraction:
        return None

    hn = normalize_channel(h, slide_stats.h_q05, slide_stats.h_q95)
    dn = normalize_channel(d, slide_stats.dab_q05, slide_stats.dab_q95)
    en = np.clip(e / max(slide_stats.e_q95, 1e-3), 0.0, 1.0)

    nuclei = tissue & (hn > cfg.nuclei_h_threshold)
    dab = tissue & (dn > cfg.dab_positive_threshold)
    nuclei_dil_1 = ndimage.binary_dilation(nuclei, structure=disk(1))
    nuclei_dil_2 = ndimage.binary_dilation(nuclei, structure=disk(2))
    ring = tissue & nuclei_dil_2 & ~nuclei_dil_1
    extra = tissue & ~nuclei_dil_2
    holes = ndimage.binary_fill_holes(tissue) & ~tissue

    tissue_pixels = float(tissue.sum())
    nuclei_areas = component_areas(nuclei)
    dab_areas = component_areas(dab)
    lap = ndimage.laplace(gray)
    grad = sobel_mag(gray)
    loc_std = local_std(gray, size=7)
    dark_threshold = q(gray[tissue], 0.10, default=0.20)
    texture_threshold = q(loc_std[tissue], 0.30, default=0.0)
    fold = tissue & (gray < dark_threshold) & (loc_std < texture_threshold)

    features = np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float32)
    features[0] = tissue_frac
    features[1] = safe_mean(od_sum[tissue])
    features[2] = safe_mean(gray[tissue])
    features[3] = safe_mean(sat[tissue])
    features[4] = safe_mean(hn[tissue])
    features[5] = safe_std(hn[tissue])
    features[6] = q(hn[tissue], 0.10)
    features[7] = q(hn[tissue], 0.50)
    features[8] = q(hn[tissue], 0.90)
    features[9] = safe_mean(dn[tissue])
    features[10] = safe_std(dn[tissue])
    features[11] = q(dn[tissue], 0.10)
    features[12] = q(dn[tissue], 0.50)
    features[13] = q(dn[tissue], 0.90)
    features[14] = safe_mean(en[tissue])
    features[15] = q(en[tissue], 0.90)
    features[16:20] = normalized_histogram(hn[tissue], bins=4, value_range=(0.0, 1.0))
    features[20:24] = normalized_histogram(dn[tissue], bins=4, value_range=(0.0, 1.0))
    features[24] = safe_mean(dn[dab])
    features[25] = float(nuclei.sum() / tissue_pixels)
    features[26] = float(len(nuclei_areas) / tissue_pixels)
    features[27] = safe_mean(nuclei_areas) / tissue_pixels
    features[28] = safe_std(nuclei_areas) / max(safe_mean(nuclei_areas), 1e-6)
    features[29] = float((dab & nuclei).sum() / tissue_pixels)
    features[30] = float((dab & ring).sum() / tissue_pixels)
    features[31] = float((dab & extra).sum() / tissue_pixels)
    features[32] = float(len(dab_areas) / tissue_pixels)
    features[33] = float(
        (ndimage.binary_dilation(dab, structure=disk(1)) ^ dab).sum()
        / max(dab.sum(), 1.0)
    )
    features[34] = float(np.log(np.var(lap) + 1e-6))
    features[35] = safe_mean(grad[tissue])
    features[36] = q(grad[tissue], 0.90)
    features[37] = float(holes.sum() / tissue_pixels)
    features[38] = float((tissue & (en > 0.30)).sum() / tissue_pixels)
    features[39] = float(fold.sum() / tissue_pixels)
    features[40] = border_fraction(tissue, tissue_pixels, cfg.border_width)
    loc_fracs = np.array([features[29], features[30], features[31]], dtype=np.float32)
    sorted_loc = np.sort(loc_fracs)[::-1]
    features[41] = float(sorted_loc[0] - sorted_loc[1]) if sorted_loc.size >= 2 else 0.0
    return features


def compute_patch_descriptors_cpu(
    patches: list[np.ndarray],
    slide_stats: SlideStats,
    cfg: PatchSelectionConfig,
    foreground_masks: list[np.ndarray | None] | None = None,
) -> list[np.ndarray | None]:
    if foreground_masks is None:
        foreground_masks = [None] * len(patches)
    return [
        compute_patch_descriptor(
            patch, slide_stats, cfg, foreground_mask=foreground_mask
        )
        for patch, foreground_mask in zip(patches, foreground_masks)
    ]


def feature_columns() -> list[str]:
    return list(FEATURE_NAMES)
