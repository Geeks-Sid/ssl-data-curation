"""Named JEPA experiment presets and sweep groups."""

from __future__ import annotations

from pathlib import Path


_CONFIG_ROOT = Path("configs/jepa")

PRESETS: dict[str, tuple[str, ...]] = {
    "lewm_default": (
        str(_CONFIG_ROOT / "families/lewm.yaml"),
        str(_CONFIG_ROOT / "ablations/masking_block.yaml"),
        str(_CONFIG_ROOT / "ablations/regularizer_gaussian_sketch.yaml"),
        str(_CONFIG_ROOT / "ablations/projector_mlp_ln.yaml"),
        str(_CONFIG_ROOT / "ablations/augment_pathology_light.yaml"),
    ),
    "ijepa_default": (
        str(_CONFIG_ROOT / "families/ijepa.yaml"),
        str(_CONFIG_ROOT / "ablations/masking_block.yaml"),
        str(_CONFIG_ROOT / "ablations/projector_mlp_ln.yaml"),
    ),
    "lewm_tokenmask_ablation": (
        str(_CONFIG_ROOT / "families/lewm.yaml"),
        str(_CONFIG_ROOT / "ablations/masking_tokens.yaml"),
        str(_CONFIG_ROOT / "ablations/regularizer_gaussian_sketch.yaml"),
        str(_CONFIG_ROOT / "ablations/projector_mlp_ln.yaml"),
        str(_CONFIG_ROOT / "ablations/augment_pathology_light.yaml"),
    ),
    "ijepa_sigreg_ablation": (
        str(_CONFIG_ROOT / "families/ijepa.yaml"),
        str(_CONFIG_ROOT / "ablations/masking_block.yaml"),
        str(_CONFIG_ROOT / "ablations/regularizer_sigreg.yaml"),
        str(_CONFIG_ROOT / "ablations/projector_mlp_ln.yaml"),
    ),
    "lewm_heavy_projector_ablation": (
        str(_CONFIG_ROOT / "families/lewm.yaml"),
        str(_CONFIG_ROOT / "ablations/masking_block.yaml"),
        str(_CONFIG_ROOT / "ablations/regularizer_gaussian_sketch.yaml"),
        str(_CONFIG_ROOT / "ablations/projector_heavy_bn.yaml"),
        str(_CONFIG_ROOT / "ablations/augment_pathology_light.yaml"),
    ),
    "ijepa_legacy_aug_ablation": (
        str(_CONFIG_ROOT / "families/ijepa.yaml"),
        str(_CONFIG_ROOT / "ablations/masking_block.yaml"),
        str(_CONFIG_ROOT / "ablations/augment_legacy_ssl.yaml"),
        str(_CONFIG_ROOT / "ablations/projector_mlp_ln.yaml"),
    ),
}

SWEEP_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "lewm_default",
        "ijepa_default",
        "lewm_tokenmask_ablation",
        "ijepa_sigreg_ablation",
        "lewm_heavy_projector_ablation",
        "ijepa_legacy_aug_ablation",
    ),
    "masking": ("lewm_default", "lewm_tokenmask_ablation"),
    "regularizer": ("lewm_default", "ijepa_sigreg_ablation"),
    "projector": ("lewm_default", "lewm_heavy_projector_ablation"),
    "augment": ("ijepa_default", "ijepa_legacy_aug_ablation"),
}


def resolve_preset(name: str) -> tuple[str, ...]:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown JEPA preset: {name}") from exc


def resolve_sweep(group: str) -> tuple[str, ...]:
    try:
        return SWEEP_GROUPS[group]
    except KeyError as exc:
        raise KeyError(f"Unknown JEPA sweep group: {group}") from exc
