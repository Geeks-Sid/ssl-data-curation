"""Stateful tar-stream dataset and transform helpers for JEPA training."""

from __future__ import annotations

import glob
import logging
import queue
import random
import subprocess
import tarfile
import threading
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import IterableDataset

from patchselect.jepa.config import AugmentConfig, DataConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ManifestEntry:
    path: str
    size: int
    mtime_ns: int


@dataclass(slots=True)
class StreamHealth:
    corrupt_images: int = 0
    read_errors: int = 0
    skipped_samples: int = 0
    prefetch_stalls: int = 0
    end_of_stream: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(slots=True)
class TarSource:
    tar_index: int
    entry: ManifestEntry
    tar_bytes: bytes | None
    local_path: str | None
    cleanup_after: bool = False


_AUGMENT_PROFILE_DEFAULTS: dict[str, dict[str, float | tuple[float, float]]] = {
    "pathology_light": {
        "crop_scale": (0.8, 1.0),
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.0,
        "brightness": 0.04,
        "contrast": 0.04,
        "saturation": 0.04,
        "hue": 0.01,
    },
    "pathology_medium": {
        "crop_scale": (0.65, 1.0),
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.25,
        "brightness": 0.08,
        "contrast": 0.08,
        "saturation": 0.08,
        "hue": 0.02,
    },
    "legacy_ssl": {
        "crop_scale": (0.5, 1.0),
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.1,
        "hue": 0.05,
    },
}


def _resolve_augment_param(
    cfg: AugmentConfig,
    key: str,
) -> float | tuple[float, float]:
    value = getattr(cfg, key)
    if value is not None:
        return value
    return _AUGMENT_PROFILE_DEFAULTS[cfg.profile][key]


def build_train_transform(cfg: AugmentConfig) -> T.Compose:
    crop_scale = _resolve_augment_param(cfg, "crop_scale")
    horizontal_flip_prob = _resolve_augment_param(cfg, "horizontal_flip_prob")
    vertical_flip_prob = _resolve_augment_param(cfg, "vertical_flip_prob")
    brightness = _resolve_augment_param(cfg, "brightness")
    contrast = _resolve_augment_param(cfg, "contrast")
    saturation = _resolve_augment_param(cfg, "saturation")
    hue = _resolve_augment_param(cfg, "hue")
    return T.Compose(
        [
            T.RandomResizedCrop(cfg.image_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=horizontal_flip_prob),
            T.RandomVerticalFlip(p=vertical_flip_prob),
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            T.ToTensor(),
            T.Normalize(mean=list(cfg.normalize_mean), std=list(cfg.normalize_std)),
        ]
    )


def discover_tar_manifest(cfg: DataConfig) -> list[ManifestEntry]:
    candidates: list[str] = []
    if cfg.tar_glob:
        candidates.extend(glob.glob(cfg.tar_glob))
    candidates.extend(cfg.tar_paths)
    unique_paths = sorted({str(Path(path).resolve()) for path in candidates})
    if not unique_paths:
        raise FileNotFoundError(
            f"No tar files matched data.tar_glob={cfg.tar_glob!r} and data.tar_paths={cfg.tar_paths!r}"
        )
    if cfg.manifest_shuffle:
        rnd = random.Random(cfg.manifest_seed)
        rnd.shuffle(unique_paths)
    manifest: list[ManifestEntry] = []
    for path_str in unique_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Tar path does not exist: {path}")
        stat = path.stat()
        manifest.append(
            ManifestEntry(path=str(path), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))
        )
    return manifest


def manifest_metadata(manifest: Iterable[ManifestEntry]) -> list[dict[str, Any]]:
    return [
        {"path": entry.path, "size": entry.size, "mtime_ns": entry.mtime_ns}
        for entry in manifest
    ]


class TarImageStream(IterableDataset):
    """Deterministic single-process tar stream with resumable cursor semantics."""

    def __init__(
        self,
        *,
        manifest: list[ManifestEntry],
        transform: T.Compose,
        start_tar_index: int = 0,
        start_member_index: int = 0,
        cache_in_ram: bool = True,
        local_cache_dir: str | None = None,
        download_cmd: str = "cp",
        prefetch_depth: int = 2,
        prefetch_timeout_sec: float = 30.0,
        extensions: tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff"),
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.transform = transform
        self.start_tar_index = max(0, int(start_tar_index))
        self.start_member_index = max(0, int(start_member_index))
        self.cache_in_ram = cache_in_ram
        self.local_cache_dir = str(local_cache_dir) if local_cache_dir else None
        self.download_cmd = download_cmd.split()
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.prefetch_timeout_sec = float(prefetch_timeout_sec)
        self.extensions = tuple(ext.lower().lstrip(".") for ext in extensions)
        self.health = StreamHealth()
        self._ready_queue: queue.Queue[TarSource | None] = queue.Queue(maxsize=self.prefetch_depth)

        if self.local_cache_dir and not self.cache_in_ram:
            Path(self.local_cache_dir).mkdir(parents=True, exist_ok=True)

    def _prepare_source(self, tar_index: int, entry: ManifestEntry) -> TarSource:
        source_path = Path(entry.path)
        if self.cache_in_ram:
            with source_path.open("rb") as handle:
                return TarSource(
                    tar_index=tar_index,
                    entry=entry,
                    tar_bytes=handle.read(),
                    local_path=None,
                )

        if source_path.exists():
            return TarSource(
                tar_index=tar_index,
                entry=entry,
                tar_bytes=None,
                local_path=str(source_path),
            )

        if not self.local_cache_dir:
            raise FileNotFoundError(
                f"Cannot materialize tar source {source_path}; local_cache_dir is not configured"
            )
        local_path = Path(self.local_cache_dir) / source_path.name
        subprocess.run(
            self.download_cmd + [entry.path, str(local_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return TarSource(
            tar_index=tar_index,
            entry=entry,
            tar_bytes=None,
            local_path=str(local_path),
            cleanup_after=True,
        )

    def _prefetch_worker(self) -> None:
        try:
            for tar_index in range(self.start_tar_index, len(self.manifest)):
                source = self._prepare_source(tar_index, self.manifest[tar_index])
                self._ready_queue.put(source)
        finally:
            self._ready_queue.put(None)

    def __iter__(self):
        self.health = StreamHealth()
        downloader_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        downloader_thread.start()

        current_start_member_index = self.start_member_index
        while True:
            try:
                tar_source = self._ready_queue.get(timeout=self.prefetch_timeout_sec)
            except queue.Empty:
                self.health.prefetch_stalls += 1
                logger.warning(
                    "Tar prefetch stall detected after %.1f seconds.",
                    self.prefetch_timeout_sec,
                )
                continue

            if tar_source is None:
                self.health.end_of_stream = True
                break

            tar_index = tar_source.tar_index
            try:
                if tar_source.tar_bytes is not None:
                    tar_handle = tarfile.open(fileobj=BytesIO(tar_source.tar_bytes), mode="r:*")
                else:
                    tar_handle = tarfile.open(tar_source.local_path, "r")
            except Exception as exc:
                self.health.read_errors += 1
                logger.warning("Failed to open tar %s: %s", tar_source.entry.path, exc)
                if tar_source.cleanup_after and tar_source.local_path:
                    Path(tar_source.local_path).unlink(missing_ok=True)
                continue

            try:
                with tar_handle as archive:
                    for member_index, member in enumerate(archive):
                        if tar_index == self.start_tar_index and member_index < current_start_member_index:
                            continue
                        if not member.isfile():
                            self.health.skipped_samples += 1
                            continue
                        suffix = member.name.lower().rsplit(".", maxsplit=1)
                        extension = suffix[-1] if len(suffix) == 2 else ""
                        if extension not in self.extensions:
                            self.health.skipped_samples += 1
                            continue
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            self.health.read_errors += 1
                            continue
                        try:
                            with Image.open(BytesIO(extracted.read())) as img:
                                image = self.transform(img.convert("RGB"))
                        except Exception as exc:
                            self.health.corrupt_images += 1
                            logger.warning(
                                "Skipping corrupt image %s in %s: %s",
                                member.name,
                                tar_source.entry.path,
                                exc,
                            )
                            continue
                        yield {
                            "image": image,
                            "tar_path": tar_source.entry.path,
                            "member_name": member.name,
                            "tar_index": tar_index,
                            "member_index": member_index,
                            "next_cursor": {
                                "tar_index": tar_index,
                                "member_index": member_index + 1,
                            },
                        }
                current_start_member_index = 0
            except Exception as exc:
                self.health.read_errors += 1
                logger.warning("Failed while reading tar %s: %s", tar_source.entry.path, exc)
            finally:
                if tar_source.cleanup_after and tar_source.local_path:
                    Path(tar_source.local_path).unlink(missing_ok=True)


def collate_image_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    images = torch.stack([sample["image"] for sample in samples], dim=0)
    return {
        "images": images,
        "samples": samples,
        "last_cursor": samples[-1]["next_cursor"],
    }
