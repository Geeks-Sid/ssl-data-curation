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
from patchselect.constants import (
    BASE_FEATURE_TO_INDEX,
    FEATURE_TO_INDEX,
    NEIGHBOR_FEATURE_NAMES,
    ROLE_NAMES,
    SEMANTIC_FEATURE_INDICES,
)
from patchselect.io_utils import write_dataframe_part


def base_idx(name: str) -> int:
    return BASE_FEATURE_TO_INDEX[name]


def feature_idx(name: str) -> int:
    return FEATURE_TO_INDEX[name]


def positive_mass(base_descriptor: np.ndarray) -> float:
    return float(
        base_descriptor[base_idx("d_hist_50_75")]
        + base_descriptor[base_idx("d_hist_75_100")]
    )


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
    dab_fraction = positive_mass(descriptor)
    if dab_fraction < 0.05:
        stain_bin = 0
    elif dab_fraction < 0.20:
        stain_bin = 1
    elif dab_fraction < 0.45:
        stain_bin = 2
    else:
        stain_bin = 3
    location_bin = int(
        np.argmax(
            [
                descriptor[base_idx("dab_in_nuc_frac")],
                descriptor[base_idx("dab_ring_frac")],
                descriptor[base_idx("dab_extra_frac")],
            ]
        )
    )
    return 4 * location_bin + stain_bin


def interface_bin_from_features(
    neigh_posfrac_diff: float,
    neigh_state_change_frac: float,
) -> int:
    if neigh_state_change_frac < 0.25 and neigh_posfrac_diff < 0.08:
        return 0
    if neigh_state_change_frac < 0.50 and neigh_posfrac_diff < 0.15:
        return 1
    if neigh_state_change_frac < 0.75 and neigh_posfrac_diff < 0.30:
        return 2
    return 3


def add_neighborhood_features(records: list[dict]) -> None:
    if not records:
        return
    base = np.stack([record["descriptor_base"] for record in records]).astype(
        np.float32
    )
    grid_lookup = {
        (record["grid_row"], record["grid_col"]): idx
        for idx, record in enumerate(records)
    }
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
            center_pos = positive_mass(base[idx])
            neighbor_pos = np.array(
                [positive_mass(base[neighbor]) for neighbor in neighbors],
                dtype=np.float32,
            )
            neighbor_features = np.array(
                [
                    float(l1.mean()),
                    float(l1.max()),
                    float(
                        np.abs(
                            base[idx, base_idx("d_mean")]
                            - base[neighbors, base_idx("d_mean")]
                        ).mean()
                    ),
                    float(np.abs(center_pos - neighbor_pos).mean()),
                    float(
                        np.abs(
                            base[idx, base_idx("nuclei_frac")]
                            - base[neighbors, base_idx("nuclei_frac")]
                        ).mean()
                    ),
                    float(
                        np.mean(
                            [
                                stain_state_bin(base[idx])
                                != stain_state_bin(base[neighbor])
                                for neighbor in neighbors
                            ]
                        )
                    ),
                ],
                dtype=np.float32,
            )

        record["descriptor"] = np.concatenate([base[idx], neighbor_features]).astype(
            np.float32
        )
        record["state_bin"] = stain_state_bin(base[idx])
        record["interface_bin"] = interface_bin_from_features(
            neigh_posfrac_diff=float(neighbor_features[3]),
            neigh_state_change_frac=float(neighbor_features[5]),
        )


def compute_local_scores(records: list[dict], cfg: PatchSelectionConfig) -> None:
    if not records:
        return
    final_matrix = np.stack([record["descriptor"] for record in records]).astype(
        np.float32
    )
    semantic = robust_scale(final_matrix[:, SEMANTIC_FEATURE_INDICES])
    semantic_center = semantic.mean(axis=0)
    centrality_raw = -np.abs(semantic - semantic_center).sum(axis=1)
    if len(records) == 1:
        rarity_raw = np.array([0.0], dtype=np.float32)
    else:
        distances = np.abs(semantic[:, None, :] - semantic[None, :, :]).sum(axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbor_count = min(3, len(records) - 1)
        nearest = np.partition(distances, neighbor_count - 1, axis=1)[
            :, :neighbor_count
        ]
        rarity_raw = nearest.mean(axis=1)

    positive_tail_raw = (
        final_matrix[:, feature_idx("d_hist_50_75")]
        + 2.0 * final_matrix[:, feature_idx("d_hist_75_100")]
        + 0.5 * final_matrix[:, feature_idx("d_pos_mean")]
    )
    quality_raw = (
        final_matrix[:, feature_idx("log_lap_var")]
        + 0.5 * final_matrix[:, feature_idx("grad_p90")]
    )
    interface_raw = (
        final_matrix[:, feature_idx("neigh_sem_l1_mean")]
        + 0.5 * final_matrix[:, feature_idx("neigh_posfrac_diff")]
        + 0.5 * final_matrix[:, feature_idx("neigh_state_change_frac")]
    )
    nuisance_raw = (
        final_matrix[:, feature_idx("unexpected_color_frac")]
        + final_matrix[:, feature_idx("fold_frac")]
        + 0.5 * final_matrix[:, feature_idx("hole_frac")]
        + 0.25 * final_matrix[:, feature_idx("border_tissue_frac")]
    )
    semantic_coverage_raw = (
        rarity_raw
        + 0.5 * positive_tail_raw
        + 0.25 * final_matrix[:, feature_idx("compartment_margin")]
    )
    redundancy_raw = -rarity_raw

    rarity = zscore(rarity_raw)
    quality = zscore(quality_raw)
    interface = zscore(interface_raw)
    nuisance = zscore(nuisance_raw)
    semantic_coverage = zscore(semantic_coverage_raw)
    redundancy_penalty = zscore(redundancy_raw)
    prototype = zscore(centrality_raw) + 0.5 * quality - 0.5 * nuisance
    positive_tail = zscore(positive_tail_raw) + 0.25 * quality - 0.5 * nuisance
    rare_state = semantic_coverage + 0.25 * interface - 0.25 * nuisance
    interface_role = interface + 0.25 * quality - 0.5 * nuisance
    # Practical surrogate for semantic coverage + lambda * interface - alpha * redundancy - beta * nuisance.
    objective = (
        cfg.semantic_weight * semantic_coverage
        + cfg.interface_weight * interface
        + cfg.quality_weight * quality
        - cfg.redundancy_weight * redundancy_penalty
        - cfg.nuisance_weight * nuisance
    )

    for idx, record in enumerate(records):
        record["quality_score"] = float(quality[idx])
        record["nuisance_score"] = float(nuisance[idx])
        record["semantic_coverage_score"] = float(semantic_coverage[idx])
        record["redundancy_penalty"] = float(redundancy_penalty[idx])
        record["rarity_score"] = float(rarity[idx])
        record["interface_score"] = float(interface[idx])
        record["prototype_score"] = float(prototype[idx])
        record["positive_tail_score"] = float(positive_tail[idx])
        record["rare_state_score"] = float(rare_state[idx])
        record["interface_role_score"] = float(interface_role[idx])
        record["objective_score"] = float(objective[idx])
        record["utility"] = float(objective[idx])


def target_local_keep(count: int, cfg: PatchSelectionConfig) -> int:
    scaled = int(np.ceil(cfg.local_keep_ratio * count))
    return min(cfg.local_keep_max, max(cfg.local_keep_min, scaled))


def role_based_local_selection(
    records: list[dict], cfg: PatchSelectionConfig
) -> list[dict]:
    if not records:
        return []
    keep = min(len(records), target_local_keep(len(records), cfg))
    if keep >= len(records):
        for rank, record in enumerate(
            sorted(records, key=lambda item: item["objective_score"], reverse=True),
            start=1,
        ):
            record["selection_rank"] = rank
            record["selection_role"] = "all_retained"
        return records

    role_to_score_key = {
        "prototype": "prototype_score",
        "positive_tail": "positive_tail_score",
        "interface": "interface_role_score",
        "rare_state": "rare_state_score",
    }
    selected: list[int] = []
    seen_states: set[int] = set()
    seen_interfaces: set[int] = set()

    for role in ROLE_NAMES:
        if len(selected) >= keep:
            break
        ordered = sorted(
            range(len(records)),
            key=lambda idx: (
                records[idx][role_to_score_key[role]],
                records[idx]["objective_score"],
            ),
            reverse=True,
        )
        for idx in ordered:
            if idx in selected:
                continue
            selected.append(idx)
            records[idx]["selection_role"] = role
            seen_states.add(records[idx]["state_bin"])
            seen_interfaces.add(records[idx]["interface_bin"])
            break

    while len(selected) < keep:
        best_idx = None
        best_score = -np.inf
        for idx, record in enumerate(records):
            if idx in selected:
                continue
            marginal = record["objective_score"]
            if record["state_bin"] not in seen_states:
                marginal += cfg.state_gain_bonus
            if record["interface_bin"] not in seen_interfaces:
                marginal += cfg.interface_gain_bonus
            if marginal > best_score:
                best_score = marginal
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        records[best_idx]["selection_role"] = "coverage_fill"
        seen_states.add(records[best_idx]["state_bin"])
        seen_interfaces.add(records[best_idx]["interface_bin"])

    selected.sort(key=lambda idx: records[idx]["objective_score"], reverse=True)
    for rank, idx in enumerate(selected, start=1):
        records[idx]["selection_rank"] = rank
    return [records[idx] for idx in selected]


def bin_key_from_row(row: pd.Series, bin_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        "" if pd.isna(row[column]) else str(row[column]) for column in bin_columns
    )


def serialize_bin_key(bin_key: tuple[str, ...]) -> str:
    raw = json.dumps(bin_key, sort_keys=False)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return digest


def count_bin_frequencies(
    candidate_files: list[Path], bin_columns: tuple[str, ...]
) -> Counter[tuple[str, ...]]:
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
    quotas = np.minimum(
        quotas, np.array([counts[bin_key] for bin_key in bins], dtype=int)
    )

    if min_quota > 0 and target_size >= len(bins) * min_quota:
        quotas = np.maximum(quotas, min_quota)
        quotas = np.minimum(
            quotas, np.array([counts[bin_key] for bin_key in bins], dtype=int)
        )

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
        frame["_bin_key"] = frame.apply(
            lambda row: bin_key_from_row(row, bin_columns), axis=1
        )
        for bin_key, group in frame.groupby("_bin_key", sort=False):
            token = serialize_bin_key(bin_key)
            key_lookup[token] = bin_key
            out_dir = partition_root / token
            out_dir.mkdir(parents=True, exist_ok=True)
            part_path = out_dir / f"part-{part_index:06d}.parquet"
            write_dataframe_part(group.drop(columns="_bin_key"), part_path)
            part_index += 1

    mapping_rows = [
        {"partition": token, "bin_key": json.dumps(list(bin_key))}
        for token, bin_key in key_lookup.items()
    ]
    if mapping_rows:
        pd.DataFrame(mapping_rows).to_parquet(
            partition_root / "partition_map.parquet", index=False
        )
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
        frame = (
            ds.dataset([str(path) for path in files], format="parquet")
            .to_table()
            .to_pandas()
        )
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
    key_lookup = partition_candidates(
        candidate_files, partition_root, config.bin_columns
    )
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
