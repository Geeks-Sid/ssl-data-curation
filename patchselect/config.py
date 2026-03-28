"""Configuration dataclasses for patch selection."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class PatchSelectionConfig:
    patch_size: int = 256
    patch_stride: int = 256
    descriptor_backend: str = "cpu"
    magnification_factors: tuple[float, ...] = (1.0, 0.5, 0.25)
    downsample_size: int | None = None
    slide_stats_size: int | None = None
    use_rle_mask: bool = True
    rle_min_fraction: float = 0.75
    tissue_min_fraction: float = 0.05
    od_tissue_threshold: float = 0.12
    sat_tissue_threshold: float = 0.08
    value_tissue_threshold: float = 0.95
    nuclei_h_threshold: float = 0.35
    dab_positive_threshold: float = 0.30
    local_keep_ratio: float = 0.10
    local_keep_min: int = 1
    local_keep_max: int = 4
    min_component_size: int = 16
    border_width: int = 4
    image_format: str = "jpg"
    jpeg_quality: int = 95
    semantic_weight: float = 0.50
    interface_weight: float = 0.30
    quality_weight: float = 0.15
    redundancy_weight: float = 0.20
    nuisance_weight: float = 0.35
    state_gain_bonus: float = 0.35
    interface_gain_bonus: float = 0.20

    def __post_init__(self) -> None:
        raw_factors = self.magnification_factors
        normalized: list[float] = []
        for raw_factor in raw_factors:
            factor = float(raw_factor)
            if factor <= 0:
                raise ValueError(
                    "magnification_factors must contain only positive values"
                )
            if factor not in normalized:
                normalized.append(factor)
        if not normalized:
            raise ValueError("magnification_factors must not be empty")
        self.magnification_factors = tuple(normalized)


@dataclass(slots=True)
class GlobalSelectionConfig:
    target_size: int
    bin_columns: tuple[str, ...] = ("tissue", "is_cancer", "state_bin", "interface_bin")
    bin_alpha: float = 0.5
    utility_column: str = "objective_score"
    dataframe_backend: str = "auto"
    show_progress: bool = True
    partition_dir_name: str = "bin_partitions"
    per_bin_min_quota: int = 0
    metadata_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "sample_id",
            "sample_slug",
            "source_shard",
            "source_index",
            "patch_index",
            "patch_size",
            "patch_x",
            "patch_y",
            "scale_level",
            "scale_factor",
            "grid_row",
            "grid_col",
            "gene",
            "tissue",
            "cell_type",
            "diagnosis",
            "scaled_image_width",
            "scaled_image_height",
            "is_cancer",
            "selection_role",
            "descriptor_backend",
            "rle_available",
            "rle_patch_foreground_frac",
            "rle_slide_foreground_frac",
            "objective_score",
            "utility",
            "quality_score",
            "nuisance_score",
            "semantic_coverage_score",
            "interface_score",
            "redundancy_penalty",
            "rarity_score",
            "state_bin",
            "interface_bin",
        )
    )
