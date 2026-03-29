"""Configuration loading and validation for JEPA training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "jepa" / "base.yaml"


def _normalize_str_seq(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


@dataclass(slots=True)
class DataConfig:
    tar_glob: str = "output_multi_full/global_selection/final_selection_tars/*.tar"
    tar_paths: tuple[str, ...] = field(default_factory=tuple)
    local_cache_dir: str = "runs/jepa/cache"
    download_cmd: str = "cp"
    cache_in_ram: bool = True
    manifest_shuffle: bool = False
    manifest_seed: int = 0
    prefetch_depth: int = 2
    prefetch_timeout_sec: float = 30.0
    extensions: tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff")
    pin_memory: bool = True

    def __post_init__(self) -> None:
        self.tar_paths = _normalize_str_seq(self.tar_paths)
        self.extensions = tuple(ext.lower().lstrip(".") for ext in self.extensions)
        if self.prefetch_depth < 1:
            raise ValueError("data.prefetch_depth must be >= 1")
        if self.prefetch_timeout_sec <= 0:
            raise ValueError("data.prefetch_timeout_sec must be > 0")


@dataclass(slots=True)
class AugmentConfig:
    image_size: int = 224
    crop_scale: tuple[float, float] = (0.5, 1.0)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.05
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("augment.image_size must be positive")
        if len(self.crop_scale) != 2:
            raise ValueError("augment.crop_scale must contain exactly two floats")


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "vit_small_patch16_224"
    pred_depth: int = 6
    pred_num_heads: int = 6
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 1024
    mask_ratio: float = 0.6
    pretrained: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.mask_ratio < 1:
            raise ValueError("model.mask_ratio must be between 0 and 1")


@dataclass(slots=True)
class LossConfig:
    lambda_sigreg: float = 0.1
    sigreg_num_projections: int = 1024
    sigreg_gamma: float = 1.0
    collapse_threshold: float = 1e-3


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine_annealing_warm_restarts"
    t_0: int = 50
    t_mult: int = 1
    eta_min: float = 0.0


@dataclass(slots=True)
class RuntimeConfig:
    experiment_name: str = "default"
    run_name: str | None = "default"
    output_root: str = "runs/jepa"
    max_steps: int = 1000
    batch_size: int = 8
    num_workers: int = 0
    device: str = "auto"
    precision: str = "16-mixed"
    seed: int = 42
    grad_clip_norm: float = 1.0
    grad_accumulation_steps: int = 1
    matmul_precision: str = "high"
    allow_existing_run_dir: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("runtime.max_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive")
        if self.num_workers != 0:
            raise ValueError("runtime.num_workers must remain 0 for stateful resume safety")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("runtime.grad_accumulation_steps must be positive")


@dataclass(slots=True)
class LoggingConfig:
    log_level: str = "INFO"
    log_every_steps: int = 10
    avg_window: int = 100

    def __post_init__(self) -> None:
        if self.log_every_steps <= 0:
            raise ValueError("logging.log_every_steps must be positive")
        if self.avg_window <= 0:
            raise ValueError("logging.avg_window must be positive")


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
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _section(section_type: type[Any], payload: dict[str, Any]) -> Any:
    return section_type(**payload)


def load_config(
    config_file: str | None = None,
    cli_overrides: list[str] | None = None,
) -> tuple[JEPAConfig, dict[str, Any], Any]:
    base_cfg = OmegaConf.load(_default_config_path())
    cfg = base_cfg
    if config_file:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(config_file)))
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(cli_overrides))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    typed = JEPAConfig(
        data=_section(DataConfig, resolved.get("data", {})),
        augment=_section(AugmentConfig, resolved.get("augment", {})),
        model=_section(ModelConfig, resolved.get("model", {})),
        loss=_section(LossConfig, resolved.get("loss", {})),
        optimizer=_section(OptimizerConfig, resolved.get("optimizer", {})),
        scheduler=_section(SchedulerConfig, resolved.get("scheduler", {})),
        runtime=_section(RuntimeConfig, resolved.get("runtime", {})),
        logging=_section(LoggingConfig, resolved.get("logging", {})),
        checkpoint=_section(CheckpointConfig, resolved.get("checkpoint", {})),
        wandb=_section(WandbConfig, resolved.get("wandb", {})),
    )
    return typed, resolved, cfg


def config_to_dict(cfg: JEPAConfig) -> dict[str, Any]:
    return asdict(cfg)
