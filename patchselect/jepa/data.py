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
import torchvision.io as tv_io
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=list(cfg.normalize_mean), std=list(cfg.normalize_std)),
        ]
    )


def _decode_image_bytes(image_bytes: bytes, extension: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Decode an encoded image into a CHW uint8 tensor."""
    if extension in {"jpg", "jpeg"}:
        encoded = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        return tv_io.decode_jpeg(encoded, mode=tv_io.ImageReadMode.RGB, device=device)
    with Image.open(BytesIO(image_bytes)) as img:
        return TF.pil_to_tensor(img.convert("RGB"))


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
                            image_bytes = extracted.read()
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
                            "image_bytes": image_bytes,
                            "extension": extension,
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


def build_collate_image_samples(
    transform: T.Compose,
    *,
    decode_device: str | torch.device = "cpu",
) -> Any:
    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        images_by_index: list[torch.Tensor | None] = [None] * len(samples)
        jpeg_indices = [
            index for index, sample in enumerate(samples) if sample["extension"] in {"jpg", "jpeg"}
        ]
        if jpeg_indices:
            encoded_batch = [
                torch.frombuffer(bytearray(samples[index]["image_bytes"]), dtype=torch.uint8)
                for index in jpeg_indices
            ]
            decoded_batch = tv_io.decode_jpeg(
                encoded_batch,
                mode=tv_io.ImageReadMode.RGB,
                device=decode_device,
            )
            for index, image in zip(jpeg_indices, decoded_batch, strict=True):
                images_by_index[index] = transform(image)

        for index, sample in enumerate(samples):
            if images_by_index[index] is not None:
                continue
            decoded = _decode_image_bytes(
                sample["image_bytes"],
                sample["extension"],
                device="cpu",
            )
            image = transform(decoded)
            if decode_device != "cpu":
                image = image.to(decode_device)
            images_by_index[index] = image

        images = torch.stack([image for image in images_by_index if image is not None], dim=0)
        return {
            "images": images,
            "samples": samples,
            "last_cursor": samples[-1]["next_cursor"],
        }

    return _collate


_PIPELINE_SENTINEL = object()


class PipelinedBatchLoader:
    """Multi-threaded data loading pipeline for high GPU utilisation.

    Architecture (3 stages, fully overlapped):
        Stage 1 – *extraction*  (1 thread): iterates ``TarImageStream``
            sequentially to preserve deterministic resume ordering and puts
            raw ``{image_bytes, extension, cursor …}`` dicts into a bounded
            queue.
        Stage 2 – *decode + transform*  (``decode_threads`` threads via
            ``ThreadPoolExecutor``): each worker decodes a JPEG on the CPU
            (``torchvision.io.decode_jpeg`` releases the GIL) and applies the
            augmentation ``transform`` pipeline, producing a float32 CHW
            tensor.
        Stage 3 – *batch assembly + prefetch*  (coordinator thread): collects
            ``batch_size`` decoded tensors **in submission order**, stacks them
            into a single tensor (optionally pinned), and places the completed
            batch into a ready queue.

    The main-thread iterator simply pops from the ready queue, so ``next()``
    returns almost instantly as long as the pipeline keeps ahead of GPU
    compute.
    """

    def __init__(
        self,
        dataset: TarImageStream,
        *,
        batch_size: int,
        transform: T.Compose,
        decode_threads: int = 4,
        prefetch_batches: int = 4,
        pin_memory: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform
        self.decode_threads = max(1, int(decode_threads))
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.pin_memory = pin_memory

    # -- internal helpers ------------------------------------------------- #

    def _decode_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Decode image bytes and apply *transform*.  Thread-safe."""
        image_bytes = sample["image_bytes"]
        extension = sample["extension"]
        if extension in {"jpg", "jpeg"}:
            encoded = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
            image = tv_io.decode_jpeg(encoded, mode=tv_io.ImageReadMode.RGB)
        else:
            image = _decode_image_bytes(image_bytes, extension, device="cpu")
        image = self.transform(image)
        return {
            "image": image,
            "next_cursor": sample["next_cursor"],
        }

    def _extraction_worker(
        self,
        sample_q: queue.Queue,
    ) -> None:
        """Stage 1 – sequential tar extraction."""
        try:
            for sample in self.dataset:
                sample_q.put(sample)
        except Exception as exc:
            sample_q.put(exc)
        finally:
            sample_q.put(_PIPELINE_SENTINEL)

    def _assembly_worker(
        self,
        sample_q: queue.Queue,
        batch_q: queue.Queue,
    ) -> None:
        """Stage 2+3 – parallel decode, then ordered batch assembly."""
        from concurrent.futures import ThreadPoolExecutor

        try:
            with ThreadPoolExecutor(
                max_workers=self.decode_threads,
                thread_name_prefix="jepa_decode",
            ) as pool:
                exhausted = False
                while not exhausted:
                    # Collect one batch worth of samples -------------------- #
                    futures: list[Any] = []
                    samples_meta: list[dict[str, Any]] = []
                    for _ in range(self.batch_size):
                        item = sample_q.get()
                        if item is _PIPELINE_SENTINEL:
                            exhausted = True
                            break
                        if isinstance(item, Exception):
                            raise item
                        futures.append(pool.submit(self._decode_one, item))
                        samples_meta.append(item)

                    if not futures:
                        break

                    # Collect results in submission order ------------------- #
                    decoded = [f.result() for f in futures]
                    images = torch.stack([d["image"] for d in decoded], dim=0)
                    if self.pin_memory and images.device.type == "cpu":
                        images = images.pin_memory()
                    batch = {
                        "images": images,
                        "last_cursor": decoded[-1]["next_cursor"],
                        "samples": samples_meta,
                    }
                    batch_q.put(batch)
        except Exception as exc:
            batch_q.put(exc)
        finally:
            batch_q.put(_PIPELINE_SENTINEL)

    # -- public interface ------------------------------------------------- #

    def __iter__(self) -> "PipelinedBatchLoader":
        # bounded queues provide natural back-pressure
        self._sample_q: queue.Queue = queue.Queue(
            maxsize=self.batch_size * (self.prefetch_batches + 1),
        )
        self._batch_q: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        self._extract_thread = threading.Thread(
            target=self._extraction_worker,
            args=(self._sample_q,),
            daemon=True,
            name="jepa_extract",
        )
        self._assembly_thread = threading.Thread(
            target=self._assembly_worker,
            args=(self._sample_q, self._batch_q),
            daemon=True,
            name="jepa_assemble",
        )
        self._extract_thread.start()
        self._assembly_thread.start()
        return self

    def __next__(self) -> dict[str, Any]:
        item = self._batch_q.get()
        if item is _PIPELINE_SENTINEL:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item
