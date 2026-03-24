"""Worker helpers for multiprocessing local patch selection."""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict
from pathlib import Path

from patchselect.arrow_utils import open_rgb_image
from patchselect.config import PatchSelectionConfig
from patchselect.pipeline import save_selected_patch, select_patches_from_image

logger = logging.getLogger(__name__)


def config_to_payload(cfg: PatchSelectionConfig) -> dict:
    return asdict(cfg)


def config_from_payload(payload: dict) -> PatchSelectionConfig:
    return PatchSelectionConfig(**payload)


def set_gpu_device(gpu_id: int | None) -> None:
    if gpu_id is None:
        return
    try:
        import cupy as cp

        cp.cuda.Device(gpu_id).use()
    except Exception:
        return


def free_gpu_memory() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return


def is_oom_error(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "out of memory",
        "memoryerror",
        "outofmemory",
        "cuda_error_out_of_memory",
        "cudamemoryerror",
    )
    return any(marker in message for marker in markers)


def process_image_task(task: dict) -> dict:
    cfg = config_from_payload(task["cfg"])
    gpu_id = task.get("gpu_id")
    try:
        logger.debug(
            "Worker starting sample=%s source_index=%s backend=%s gpu=%s.",
            task["sample_id"],
            task["source_index"],
            cfg.descriptor_backend,
            gpu_id if gpu_id is not None else "cpu",
        )
        set_gpu_device(gpu_id)
        image = open_rgb_image(task["bytes_data"])
        rows, patch_records = select_patches_from_image(
            image=image,
            sample_id=task["sample_id"],
            metadata=task["metadata"],
            cfg=cfg,
            source_shard=task["source_shard"],
            source_index=task["source_index"],
        )
        if task.get("save_selected_patches"):
            output_dir = Path(task["selected_patch_dir"])
            for patch_rgb, record in patch_records:
                save_selected_patch(patch_rgb, record, output_dir, cfg)
        result = {
            "status": "ok",
            "sample_id": task["sample_id"],
            "source_index": task["source_index"],
            "rows": rows,
        }
        logger.debug(
            "Worker completed sample=%s with %d selected row(s).",
            task["sample_id"],
            len(rows),
        )
    except Exception as exc:
        result = {
            "status": "oom" if is_oom_error(exc) else "error",
            "sample_id": task["sample_id"],
            "source_index": task["source_index"],
            "message": f"{type(exc).__name__}: {exc}",
        }
        logger.exception(
            "Worker failed for sample=%s with status=%s.",
            task["sample_id"],
            result["status"],
        )
    finally:
        if cfg.descriptor_backend == "cucim":
            free_gpu_memory()
        gc.collect()
    return result
