"""High-level per-image patch selection pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from patchselect.arrow_utils import decode_rle_mask, metadata_without_large_fields, normalize_metadata, slugify
from patchselect.config import PatchSelectionConfig
from patchselect.constants import FEATURE_NAMES
from patchselect.descriptor_backend import compute_patch_descriptors_for_backend, compute_slide_stats_for_backend
from patchselect.descriptors import tile_starts
from patchselect.selection import add_neighborhood_features, compute_local_scores, role_based_local_selection


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def feature_dict(descriptor: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURE_NAMES, descriptor)}


def build_patch_filename(record: dict, image_format: str) -> str:
    marker = slugify(record["gene"], "unknown_marker")
    tissue = slugify(record["tissue"], "unknown_tissue")
    cell_type = slugify(record["cell_type"], "unknown_celltype")
    return (
        f"{marker}_{tissue}_{cell_type}_{record['sample_slug']}"
        f"_r{record['grid_row']:02d}_c{record['grid_col']:02d}"
        f"_x{record['patch_x']}_y{record['patch_y']}.{image_format}"
    )


def select_patches_from_image(
    image: Image.Image,
    sample_id: str,
    metadata: dict,
    cfg: PatchSelectionConfig,
    source_shard: str,
    source_index: int,
) -> tuple[list[dict], list[tuple[np.ndarray, dict]]]:
    rgb = image_to_rgb_array(image)
    rle_mask = None
    if cfg.use_rle_mask:
        rle_mask = decode_rle_mask(metadata.get("rle_mask"), rgb.shape[0], rgb.shape[1])
    slide_stats = compute_slide_stats_for_backend(rgb, cfg, foreground_mask=rle_mask)
    starts_y = tile_starts(rgb.shape[0], cfg.patch_size, cfg.patch_stride)
    starts_x = tile_starts(rgb.shape[1], cfg.patch_size, cfg.patch_stride)
    normalized_metadata = normalize_metadata(metadata)
    compact_metadata = metadata_without_large_fields(metadata)
    sample_slug = slugify(sample_id, "sample")
    rle_available = int(rle_mask is not None)
    rle_slide_foreground_frac = float(rle_mask.mean()) if rle_mask is not None else None

    candidate_records: list[dict] = []
    candidate_patches: list[np.ndarray] = []
    candidate_masks: list[np.ndarray | None] = []
    patch_index = 0
    for grid_row, top in enumerate(starts_y):
        for grid_col, left in enumerate(starts_x):
            patch = rgb[top : top + cfg.patch_size, left : left + cfg.patch_size]
            patch_rle_mask = None
            patch_rle_fraction = None
            if rle_mask is not None:
                patch_rle_mask = rle_mask[top : top + cfg.patch_size, left : left + cfg.patch_size]
                patch_rle_fraction = float(patch_rle_mask.mean()) if patch_rle_mask.size else 0.0
                if patch_rle_fraction < cfg.rle_min_fraction:
                    continue
            candidate_patches.append(patch)
            candidate_masks.append(patch_rle_mask)
            candidate_records.append(
                {
                    "sample_id": sample_id,
                    "sample_slug": sample_slug,
                    "source_shard": source_shard,
                    "source_index": source_index,
                    "patch_index": patch_index,
                    "patch_size": cfg.patch_size,
                    "patch_x": left,
                    "patch_y": top,
                    "grid_row": grid_row,
                    "grid_col": grid_col,
                    "image_width": int(rgb.shape[1]),
                    "image_height": int(rgb.shape[0]),
                    "descriptor_backend": cfg.descriptor_backend,
                    "rle_available": rle_available,
                    "rle_patch_foreground_frac": patch_rle_fraction,
                    "rle_slide_foreground_frac": rle_slide_foreground_frac,
                    "metadata_json": json.dumps(compact_metadata, sort_keys=True),
                    **normalized_metadata,
                }
            )
            patch_index += 1

    if not candidate_records:
        return [], []

    descriptors = compute_patch_descriptors_for_backend(
        candidate_patches,
        slide_stats,
        cfg,
        foreground_masks=candidate_masks,
    )
    records: list[dict] = []
    for record, descriptor in zip(candidate_records, descriptors):
        if descriptor is None:
            continue
        record["descriptor_base"] = descriptor
        records.append(record)

    if not records:
        return [], []

    add_neighborhood_features(records)
    compute_local_scores(records, cfg)
    selected = role_based_local_selection(records, cfg)
    selected.sort(key=lambda item: item["selection_rank"])
    total_valid = len(records)

    selected_rows = []
    selected_patches = []
    for record in selected:
        row = {
            "sample_id": record["sample_id"],
            "sample_slug": record["sample_slug"],
            "source_shard": record["source_shard"],
            "source_index": record["source_index"],
            "patch_index": record["patch_index"],
            "patch_size": record["patch_size"],
            "patch_x": record["patch_x"],
            "patch_y": record["patch_y"],
            "grid_row": record["grid_row"],
            "grid_col": record["grid_col"],
            "image_width": record["image_width"],
            "image_height": record["image_height"],
            "local_valid_patch_count": total_valid,
            "selection_rank": record["selection_rank"],
            "selection_role": record["selection_role"],
            "descriptor_backend": record["descriptor_backend"],
            "rle_available": record["rle_available"],
            "rle_patch_foreground_frac": record["rle_patch_foreground_frac"],
            "rle_slide_foreground_frac": record["rle_slide_foreground_frac"],
            "objective_score": record["objective_score"],
            "utility": record["utility"],
            "quality_score": record["quality_score"],
            "nuisance_score": record["nuisance_score"],
            "semantic_coverage_score": record["semantic_coverage_score"],
            "interface_score": record["interface_score"],
            "redundancy_penalty": record["redundancy_penalty"],
            "rarity_score": record["rarity_score"],
            "prototype_score": record["prototype_score"],
            "positive_tail_score": record["positive_tail_score"],
            "interface_role_score": record["interface_role_score"],
            "rare_state_score": record["rare_state_score"],
            "state_bin": record["state_bin"],
            "interface_bin": record["interface_bin"],
            "gene": record["gene"],
            "tissue": record["tissue"],
            "cell_type": record["cell_type"],
            "diagnosis": record["diagnosis"],
            "snomed_code": record["snomed_code"],
            "md5": record["md5"],
            "is_cancer": record["is_cancer"],
            "metadata_json": record["metadata_json"],
            **feature_dict(record["descriptor"]),
        }
        selected_rows.append(row)
        patch_rgb = rgb[
            record["patch_y"] : record["patch_y"] + cfg.patch_size,
            record["patch_x"] : record["patch_x"] + cfg.patch_size,
        ]
        selected_patches.append((patch_rgb, row))

    return selected_rows, selected_patches


def save_selected_patch(patch_rgb: np.ndarray, record: dict, output_dir: Path, cfg: PatchSelectionConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = build_patch_filename(record, cfg.image_format)
    out_path = output_dir / filename
    image = Image.fromarray(patch_rgb)
    if cfg.image_format.lower() in {"jpg", "jpeg"}:
        image.save(out_path, quality=cfg.jpeg_quality)
    else:
        image.save(out_path)
    return out_path
