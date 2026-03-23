"""Local and global selection logic for patchselect."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from patchselect.config import GlobalSelectionConfig, PatchSelectionConfig
from patchselect.constants import NEIGHBOR_FEATURE_NAMES, SEMANTIC_FEATURE_INDICES
from patchselect.io_utils import write_dataframe_part


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = values.std()
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def robust_scale(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    fallback = matrix.std(axis=0)
    scale = np.where(mad > 1e-6, 1.4826 * mad, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (matrix - median) / scale


def stain_state_bin(descriptor: np.ndarray) -> int:
    dab_fraction = descriptor[17]
    if dab_fraction < 0.02:
        stain_bin = 0
    elif dab_fraction < 0.10:
        stain_bin = 1
    elif dab_fraction < 0.30:
        stain_bin = 2
    else:
        stain_bin = 3
    location_bin = int(np.argmax([descriptor[29], descriptor[30], descriptor[31]]))
    return 4 * location_bin + stain_bin


def add_neighborhood_features(records: list[dict]) -> None:
    if not records:
        return
    base = np.stack([record["descriptor_base"] for record in records]).astype(np.float32)
    grid_lookup = {(record["grid_row"], record["grid_col"]): idx for idx, record in enumerate(records)}
    semantic = base[:, SEMANTIC_FEATURE_INDICES]

    for idx, record in enumerate(records):
        neighbors = []
        row, col = record["grid_row"], record["grid_col"]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_idx = grid_lookup.get((row + dr, col + dc))
            if neighbor_idx is not None:
                neighbors.append(neighbor_idx)

        if not neighbors:
            neighbor_features = np.zeros(len(NEIGHBOR_FEATURE_NAMES), dtype=np.float32)
        else:
            neighbor_sem = semantic[neighbors]
            l1 = np.abs(neighbor_sem - semantic[idx]).sum(axis=1)
            neighbor_features = np.array(
                [
                    float(l1.mean()),
                    float(l1.max()),
                    float(np.abs(base[idx, 9] - base[neighbors, 9]).mean()),
                    float(np.abs(base[idx, 17] - base[neighbors, 17]).mean()),
                    float(np.abs(base[idx, 25] - base[neighbors, 25]).mean()),
                    float(
                        np.mean(
                            [stain_state_bin(base[idx]) != stain_state_bin(base[neighbor]) for neighbor in neighbors]
                        )
                    ),
                ],
                dtype=np.float32,
            )

        record["descriptor"] = np.concatenate([base[idx], neighbor_features]).astype(np.float32)
        record["state_bin"] = stain_state_bin(base[idx])


def compute_local_scores(records: list[dict]) -> None:
    if not records:
        return
    final_matrix = np.stack([record["descriptor"] for record in records]).astype(np.float32)
    semantic = robust_scale(final_matrix[:, SEMANTIC_FEATURE_INDICES])
    if len(records) == 1:
        rarity_raw = np.array([0.0], dtype=np.float32)
    else:
        distances = np.abs(semantic[:, None, :] - semantic[None, :, :]).sum(axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbor_count = min(3, len(records) - 1)
        nearest = np.partition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
        rarity_raw = nearest.mean(axis=1)

    quality_raw = final_matrix[:, 34] - 0.5 * final_matrix[:, 39] - 0.5 * final_matrix[:, 40] - 0.25 * final_matrix[:, 38]
    interface_raw = final_matrix[:, 42] + 0.5 * final_matrix[:, 47]
    rarity = zscore(rarity_raw)
    quality = zscore(quality_raw)
    interface = zscore(interface_raw)
    utility = 0.45 * rarity + 0.35 * interface + 0.20 * quality

    for idx, record in enumerate(records):
        record["rarity_score"] = float(rarity[idx])
        record["quality_score"] = float(quality[idx])
        record["interface_score"] = float(interface[idx])
        record["utility"] = float(utility[idx])


def target_local_keep(count: int, cfg: PatchSelectionConfig) -> int:
    scaled = int(np.ceil(cfg.local_keep_ratio * count))
    return min(cfg.local_keep_max, max(cfg.local_keep_min, scaled))


def greedy_farthest_point(records: list[dict], cfg: PatchSelectionConfig) -> list[dict]:
    if not records:
        return []
    final_matrix = np.stack([record["descriptor"] for record in records]).astype(np.float32)
    semantic = robust_scale(final_matrix[:, SEMANTIC_FEATURE_INDICES])
    utilities = np.array([record["utility"] for record in records], dtype=np.float32)
    keep = min(len(records), target_local_keep(len(records), cfg))
    if keep >= len(records):
        for rank, record in enumerate(sorted(records, key=lambda item: item["utility"], reverse=True), start=1):
            record["selection_rank"] = rank
        return records

    selected = [int(np.argmax(utilities))]
    min_dist = np.abs(semantic - semantic[selected[0]]).sum(axis=1)
    min_dist[selected[0]] = -np.inf
    utility_norm = utilities - utilities.min()
    if utility_norm.max() > 1e-6:
        utility_norm = utility_norm / utility_norm.max()

    while len(selected) < keep:
        candidate_scores = min_dist + cfg.utility_fps_weight * utility_norm
        candidate_scores[selected] = -np.inf
        next_idx = int(np.argmax(candidate_scores))
        selected.append(next_idx)
        next_dist = np.abs(semantic - semantic[next_idx]).sum(axis=1)
        min_dist = np.minimum(min_dist, next_dist)

    for rank, idx in enumerate(selected, start=1):
        records[idx]["selection_rank"] = rank
    return [records[idx] for idx in selected]


def bin_key_from_row(row: pd.Series, bin_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("" if pd.isna(row[column]) else str(row[column]) for column in bin_columns)


def serialize_bin_key(bin_key: tuple[str, ...]) -> str:
    raw = json.dumps(bin_key, sort_keys=False)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return digest


def count_bin_frequencies(candidate_files: list[Path], bin_columns: tuple[str, ...]) -> Counter[tuple[str, ...]]:
    dataset = ds.dataset([str(path) for path in candidate_files], format="parquet")
    counts: Counter[tuple[str, ...]] = Counter()
    for batch in dataset.scanner(columns=list(bin_columns)).to_batches():
        frame = batch.to_pandas()
        for row in frame.itertuples(index=False, name=None):
            counts[tuple("" if value is None else str(value) for value in row)] += 1
    return counts


def allocate_bin_quotas(
    counts: Counter[tuple[str, ...]],
    target_size: int,
    alpha: float,
    min_quota: int,
) -> dict[tuple[str, ...], int]:
    if not counts:
        return {}
    bins = list(counts.keys())
    if target_size <= 0:
        return {bin_key: 0 for bin_key in bins}

    weights = np.array([counts[bin_key] ** alpha for bin_key in bins], dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        weights = np.ones_like(weights)
        weight_sum = weights.sum()

    raw = target_size * weights / weight_sum
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(quotas, np.array([counts[bin_key] for bin_key in bins], dtype=int))

    if min_quota > 0 and target_size >= len(bins) * min_quota:
        quotas = np.maximum(quotas, min_quota)
        quotas = np.minimum(quotas, np.array([counts[bin_key] for bin_key in bins], dtype=int))

    remainder = target_size - int(quotas.sum())
    if remainder > 0:
        fractions = raw - np.floor(raw)
        order = np.argsort(-fractions)
        for idx in order:
            if remainder <= 0:
                break
            if quotas[idx] < counts[bins[idx]]:
                quotas[idx] += 1
                remainder -= 1

    if remainder > 0:
        order = np.argsort(-np.array([counts[bin_key] for bin_key in bins]))
        for idx in order:
            if remainder <= 0:
                break
            if quotas[idx] < counts[bins[idx]]:
                quotas[idx] += 1
                remainder -= 1

    return {bins[idx]: int(quotas[idx]) for idx in range(len(bins))}


def partition_candidates(
    candidate_files: list[Path],
    partition_root: Path,
    bin_columns: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    partition_root.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset([str(path) for path in candidate_files], format="parquet")
    part_index = 0
    key_lookup: dict[str, tuple[str, ...]] = {}

    for batch in dataset.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["_bin_key"] = frame.apply(lambda row: bin_key_from_row(row, bin_columns), axis=1)
        for bin_key, group in frame.groupby("_bin_key", sort=False):
            token = serialize_bin_key(bin_key)
            key_lookup[token] = bin_key
            out_dir = partition_root / token
            out_dir.mkdir(parents=True, exist_ok=True)
            part_path = out_dir / f"part-{part_index:06d}.parquet"
            write_dataframe_part(group.drop(columns="_bin_key"), part_path)
            part_index += 1

    mapping_rows = [{"partition": token, "bin_key": json.dumps(list(bin_key))} for token, bin_key in key_lookup.items()]
    if mapping_rows:
        pd.DataFrame(mapping_rows).to_parquet(partition_root / "partition_map.parquet", index=False)
    return key_lookup


def select_top_by_bin(
    partition_root: Path,
    quotas: dict[tuple[str, ...], int],
    key_lookup: dict[str, tuple[str, ...]],
    output_dir: Path,
    utility_column: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    out_index = 0
    for token, bin_key in key_lookup.items():
        quota = quotas.get(bin_key, 0)
        if quota <= 0:
            continue
        partition_dir = partition_root / token
        files = sorted(partition_dir.glob("*.parquet"))
        if not files:
            continue
        frame = ds.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        if frame.empty:
            continue
        frame = frame.nlargest(quota, utility_column)
        part_path = output_dir / f"final_selection_part-{out_index:06d}.parquet"
        write_dataframe_part(frame, part_path)
        written += len(frame)
        out_index += 1
    return written


def run_global_selection(
    candidate_files: list[Path],
    output_dir: Path,
    config: GlobalSelectionConfig,
) -> dict[str, int]:
    counts = count_bin_frequencies(candidate_files, config.bin_columns)
    quotas = allocate_bin_quotas(
        counts=counts,
        target_size=config.target_size,
        alpha=config.bin_alpha,
        min_quota=config.per_bin_min_quota,
    )
    partition_root = output_dir / config.partition_dir_name
    key_lookup = partition_candidates(candidate_files, partition_root, config.bin_columns)
    selected_rows = select_top_by_bin(
        partition_root=partition_root,
        quotas=quotas,
        key_lookup=key_lookup,
        output_dir=output_dir / "final_selection",
        utility_column=config.utility_column,
    )
    return {
        "candidate_files": len(candidate_files),
        "bin_count": len(counts),
        "selected_rows": selected_rows,
    }
