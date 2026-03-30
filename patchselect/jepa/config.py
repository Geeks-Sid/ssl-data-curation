"""Configuration loading and validation for JEPA training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


def _find_repo_root() -> Path | None:
    env_root = os.environ.get("PATCHSELECT_REPO_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    module_root = Path(__file__).resolve().parents[2]
    candidates.extend([module_root, *module_root.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "configs" / "jepa" / "base.yaml").exists():
            return candidate
    return None


def _repo_root() -> Path:
    repo_root = _find_repo_root()
    if repo_root is None:
        raise FileNotFoundError(
            "Unable to locate the JEPA repo root. Set PATCHSELECT_REPO_ROOT or run from the repository root."
        )
    return repo_root


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "jepa" / "base.yaml"


def _normalize_str_seq(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _normalize_optional_pair(
    values: Iterable[float] | None,
) -> tuple[float, float] | None:
    if values is None:
        return None
    items = tuple(float(value) for value in values)
    if len(items) != 2:
        raise ValueError("Expected exactly two numeric values")
    return items


@dataclass(slots=True)
class DataConfig:
    tar_glob: str = "output_multi_full/global_selection/final_selection_tars/*.tar"
    tar_paths: tuple[str, ...] = field(default_factory=tuple)
    local_cache_dir: str = "runs/jepa/cache"
    download_cmd: str = "cp"
    cache_in_ram: bool = True
    manifest_shuffle: bool = False
    manifest_seed: int = 0
    prefetch_depth: int = 4
    prefetch_timeout_sec: float = 30.0
    extensions: tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff")
    pin_memory: bool = True
    decode_threads: int = 4

    def __post_init__(self) -> None:
        self.tar_paths = _normalize_str_seq(self.tar_paths)
        self.extensions = tuple(ext.lower().lstrip(".") for ext in self.extensions)
        if self.prefetch_depth < 1:
            raise ValueError("data.prefetch_depth must be >= 1")
        if self.prefetch_timeout_sec <= 0:
            raise ValueError("data.prefetch_timeout_sec must be > 0")
        if self.decode_threads < 1:
            raise ValueError("data.decode_threads must be >= 1")


@dataclass(slots=True)
class AugmentConfig:
    profile: str = "pathology_light"
    image_size: int = 224
    crop_scale: tuple[float, float] | None = None
    horizontal_flip_prob: float | None = None
    vertical_flip_prob: float | None = None
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    hue: float | None = None
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        self.crop_scale = _normalize_optional_pair(self.crop_scale)
        if self.image_size <= 0:
            raise ValueError("augment.image_size must be positive")
        if self.profile not in {"pathology_light", "pathology_medium", "legacy_ssl"}:
            raise ValueError(
                "augment.profile must be one of: pathology_light, pathology_medium, legacy_ssl"
            )


@dataclass(slots=True)
class ModelConfig:
    family: str = "lewm"
    model_name: str = "vit_small_patch16_224"
    pred_depth: int = 6
    pred_num_heads: int = 6
    pretrained: bool = False

    def __post_init__(self) -> None:
        if self.family not in {"lewm", "ijepa"}:
            raise ValueError("model.family must be one of: lewm, ijepa")
        if self.pred_depth <= 0:
            raise ValueError("model.pred_depth must be positive")


@dataclass(slots=True)
class MaskingConfig:
    strategy: str = "block_targets"
    mask_ratio: float = 0.6
    num_targets: int = 4
    target_scale_range: tuple[float, float] = (0.15, 0.2)
    aspect_ratio_range: tuple[float, float] = (0.75, 1.5)
    context_min_keep: int = 32
    allow_overlap: bool = False

    def __post_init__(self) -> None:
        self.target_scale_range = _normalize_optional_pair(self.target_scale_range) or (
            0.15,
            0.2,
        )
        self.aspect_ratio_range = _normalize_optional_pair(self.aspect_ratio_range) or (
            0.75,
            1.5,
        )
        if self.strategy not in {"random_tokens", "block_targets"}:
            raise ValueError(
                "masking.strategy must be one of: random_tokens, block_targets"
            )
        if not 0 < self.mask_ratio < 1:
            raise ValueError("masking.mask_ratio must be between 0 and 1")
        if self.num_targets < 1:
            raise ValueError("masking.num_targets must be >= 1")
        if self.context_min_keep < 1:
            raise ValueError("masking.context_min_keep must be >= 1")


@dataclass(slots=True)
class TargetEncoderConfig:
    kind: str = "shared"
    ema_momentum: float = 0.996

    def __post_init__(self) -> None:
        if self.kind not in {"shared", "ema"}:
            raise ValueError("target_encoder.kind must be one of: shared, ema")
        if not 0 < self.ema_momentum < 1:
            raise ValueError("target_encoder.ema_momentum must be between 0 and 1")


@dataclass(slots=True)
class ProjectorConfig:
    kind: str = "mlp_ln"
    hidden_dim: int = 2048
    out_dim: int = 1024

    def __post_init__(self) -> None:
        if self.kind not in {"heavy_bn", "mlp_ln", "linear"}:
            raise ValueError("projector.kind must be one of: heavy_bn, mlp_ln, linear")
        if self.out_dim <= 0:
            raise ValueError("projector.out_dim must be positive")
        if self.kind != "linear" and self.hidden_dim <= 0:
            raise ValueError("projector.hidden_dim must be positive")


@dataclass(slots=True)
class LossConfig:
    prediction: str = "mse"
    collapse_threshold: float = 1e-3

    def __post_init__(self) -> None:
        if self.prediction != "mse":
            raise ValueError("loss.prediction currently supports only: mse")


@dataclass(slots=True)
class RegularizerConfig:
    name: str = "sigreg"
    weight: float = 0.09
    num_projections: int = 1024
    sigreg_knots: int = 17
    gamma: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.name not in {"none", "sigreg", "gaussian_sketch"}:
            raise ValueError(
                "regularizer.name must be one of: none, sigreg, gaussian_sketch"
            )
        if self.weight < 0:
            raise ValueError("regularizer.weight must be >= 0")
        if self.num_projections < 1:
            raise ValueError("regularizer.num_projections must be >= 1")
        if self.sigreg_knots < 2:
            raise ValueError("regularizer.sigreg_knots must be >= 2")
        if self.gamma <= 0:
            raise ValueError("regularizer.gamma must be > 0")
        if self.eps <= 0:
            raise ValueError("regularizer.eps must be > 0")


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine_annealing"
    warmup_steps: int = 500
    t_0: int = 50
    t_mult: int = 1
    eta_min: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in {"none", "cosine_annealing", "cosine_annealing_warm_restarts"}:
            raise ValueError(
                "scheduler.name must be one of: none, cosine_annealing, cosine_annealing_warm_restarts"
            )
        if self.warmup_steps < 0:
            raise ValueError("scheduler.warmup_steps must be >= 0")
        if self.t_0 <= 0:
            raise ValueError("scheduler.t_0 must be positive")
        if self.t_mult < 1:
            raise ValueError("scheduler.t_mult must be >= 1")
        if self.eta_min < 0:
            raise ValueError("scheduler.eta_min must be >= 0")


@dataclass(slots=True)
class TrainerConfig:
    """Configuration mapping to PyTorch Lightning Trainer kwargs."""

    max_steps: int = 100000
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True
    num_sanity_val_steps: int = 0
    accelerator: str = "auto"
    devices: str | int = "auto"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("trainer.max_steps must be positive")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("trainer.accumulate_grad_batches must be positive")

    def to_trainer_kwargs(self) -> dict[str, object]:
        """Return kwargs suitable for pl.Trainer(**kwargs)."""
        return {
            "max_steps": self.max_steps,
            "precision": self.precision,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "log_every_n_steps": self.log_every_n_steps,
            "enable_checkpointing": self.enable_checkpointing,
            "num_sanity_val_steps": self.num_sanity_val_steps,
            "accelerator": self.accelerator,
            "devices": self.devices,
        }


@dataclass(slots=True)
class RuntimeConfig:
    experiment_name: str = "default"
    run_name: str | None = "default"
    output_root: str = "runs/jepa"
    batch_size: int = 8
    seed: int = 42
    matmul_precision: str = "high"
    allow_existing_run_dir: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive")


@dataclass(slots=True)
class LoggingConfig:
    log_level: str = "INFO"
    log_every_steps: int = 50
    avg_window: int = 100
    metrics_mode: str = "essential"
    diagnostics_every_steps: int = 0

    def __post_init__(self) -> None:
        if self.log_every_steps <= 0:
            raise ValueError("logging.log_every_steps must be positive")
        if self.avg_window <= 0:
            raise ValueError("logging.avg_window must be positive")
        if self.metrics_mode not in {"essential", "full"}:
            raise ValueError("logging.metrics_mode must be one of: essential, full")
        if self.diagnostics_every_steps < 0:
            raise ValueError("logging.diagnostics_every_steps must be >= 0")


@dataclass(slots=True)
class CheckpointConfig:
    save_every_steps: int = 100
    keep_last_k: int = 3
    resume: str = "none"

    def __post_init__(self) -> None:
        if self.save_every_steps <= 0:
            raise ValueError("checkpoint.save_every_steps must be positive")
        if self.keep_last_k < 1:
            raise ValueError("checkpoint.keep_last_k must be >= 1")


@dataclass(slots=True)
class WandbConfig:
    mode: str = "online"
    project: str = "jepa-training"
    entity: str | None = None
    group: str | None = None
    job_type: str = "train"
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None
    log_checkpoints: bool = False

    def __post_init__(self) -> None:
        self.tags = _normalize_str_seq(self.tags)
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be one of: online, offline, disabled")


@dataclass(slots=True)
class JEPAConfig:
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    target_encoder: TargetEncoderConfig = field(default_factory=TargetEncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _section(section_type: type[Any], payload: dict[str, Any]) -> Any:
    return section_type(**payload)


def validate_config(cfg: JEPAConfig) -> None:
    if cfg.model.family == "ijepa" and cfg.target_encoder.kind != "ema":
        raise ValueError("ijepa requires target_encoder.kind=ema")
    if cfg.model.family == "lewm" and cfg.target_encoder.kind != "shared":
        raise ValueError("lewm requires target_encoder.kind=shared")
    if cfg.masking.strategy == "block_targets":
        low, high = cfg.masking.target_scale_range
        if not 0 < low <= high < 1:
            raise ValueError(
                "masking.target_scale_range must satisfy 0 < low <= high < 1"
            )
        ar_low, ar_high = cfg.masking.aspect_ratio_range
        if ar_low <= 0 or ar_high <= 0 or ar_low > ar_high:
            raise ValueError(
                "masking.aspect_ratio_range must satisfy 0 < low <= high"
            )
    if cfg.regularizer.name == "gaussian_sketch" and cfg.regularizer.num_projections < 1:
        raise ValueError(
            "gaussian_sketch requires regularizer.num_projections >= 1"
        )


def load_config(
    config_file: str | list[str] | tuple[str, ...] | None = None,
    cli_overrides: list[str] | None = None,
) -> tuple[JEPAConfig, dict[str, Any], Any]:
    base_cfg = OmegaConf.load(_default_config_path())
    cfg = base_cfg
    config_files: list[str] = []
    if isinstance(config_file, str):
        config_files = [config_file]
    elif config_file is not None:
        config_files = [str(path) for path in config_file]
    for path in config_files:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(path)))
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(cli_overrides))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    typed = JEPAConfig(
        data=_section(DataConfig, resolved.get("data", {})),
        augment=_section(AugmentConfig, resolved.get("augment", {})),
        model=_section(ModelConfig, resolved.get("model", {})),
        masking=_section(MaskingConfig, resolved.get("masking", {})),
        target_encoder=_section(
            TargetEncoderConfig, resolved.get("target_encoder", {})
        ),
        projector=_section(ProjectorConfig, resolved.get("projector", {})),
        loss=_section(LossConfig, resolved.get("loss", {})),
        regularizer=_section(RegularizerConfig, resolved.get("regularizer", {})),
        optimizer=_section(OptimizerConfig, resolved.get("optimizer", {})),
        scheduler=_section(SchedulerConfig, resolved.get("scheduler", {})),
        trainer=_section(TrainerConfig, resolved.get("trainer", {})),
        runtime=_section(RuntimeConfig, resolved.get("runtime", {})),
        logging=_section(LoggingConfig, resolved.get("logging", {})),
        checkpoint=_section(CheckpointConfig, resolved.get("checkpoint", {})),
        wandb=_section(WandbConfig, resolved.get("wandb", {})),
    )
    validate_config(typed)
    return typed, resolved, cfg


def config_to_dict(cfg: JEPAConfig) -> dict[str, Any]:
    return asdict(cfg)
