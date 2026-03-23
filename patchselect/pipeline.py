"""High-level per-image patch selection pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from patchselect.arrow_utils import normalize_metadata, slugify
from patchselect.config import PatchSelectionConfig
from patchselect.constants import FEATURE_NAMES
from patchselect.descriptors import compute_patch_descriptor, compute_slide_stats, tile_starts
from patchselect.selection import add_neighborhood_features, compute_local_scores, greedy_farthest_point


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
    slide_stats = compute_slide_stats(rgb, cfg)
    starts_y = tile_starts(rgb.shape[0], cfg.patch_size, cfg.patch_stride)
    starts_x = tile_starts(rgb.shape[1], cfg.patch_size, cfg.patch_stride)
    normalized_metadata = normalize_metadata(metadata)
    sample_slug = slugify(sample_id, "sample")

    records: list[dict] = []
    patch_index = 0
    for grid_row, top in enumerate(starts_y):
        for grid_col, left in enumerate(starts_x):
            patch = rgb[top : top + cfg.patch_size, left : left + cfg.patch_size]
            descriptor = compute_patch_descriptor(patch, slide_stats, cfg)
            if descriptor is None:
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_slug": sample_slug,
                    "source_shard": source_shard,
                    "source_index": source_index,
                    "patch_index": patch_index,
                    "patch_x": left,
                    "patch_y": top,
                    "grid_row": grid_row,
                    "grid_col": grid_col,
                    "image_width": int(rgb.shape[1]),
                    "image_height": int(rgb.shape[0]),
                    "descriptor_base": descriptor,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                    **normalized_metadata,
                }
            )
            patch_index += 1

    if not records:
        return [], []

    add_neighborhood_features(records)
    compute_local_scores(records)
    selected = greedy_farthest_point(records, cfg)
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
            "patch_x": record["patch_x"],
            "patch_y": record["patch_y"],
            "grid_row": record["grid_row"],
            "grid_col": record["grid_col"],
            "image_width": record["image_width"],
            "image_height": record["image_height"],
            "local_valid_patch_count": total_valid,
            "selection_rank": record["selection_rank"],
            "utility": record["utility"],
            "quality_score": record["quality_score"],
            "interface_score": record["interface_score"],
            "rarity_score": record["rarity_score"],
            "state_bin": record["state_bin"],
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
