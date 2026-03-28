"""Run per-image local patch selection over Arrow shards."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from patchselect.arrow_utils import (
    discover_arrow_files,
    extract_custom_metadata,
    load_arrow_shard,
)
from patchselect.config import PatchSelectionConfig
from patchselect.io_utils import write_json, write_rows_part
from patchselect.local_worker import config_to_payload, process_image_task
from patchselect.logging_utils import add_logging_args, configure_logging

logger = logging.getLogger(__name__)
RESUME_STATE_PATH = "resume_state.json"


def parse_magnification_factors(raw_value: str) -> tuple[float, ...]:
    factors = tuple(
        float(token.strip()) for token in raw_value.split(",") if token.strip()
    )
    if not factors:
        raise argparse.ArgumentTypeError(
            "magnification_factors must contain at least one positive float"
        )
    if any(factor <= 0 for factor in factors):
        raise argparse.ArgumentTypeError(
            "magnification_factors must contain only positive floats"
        )
    return factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local stain-aware patch selection on Arrow shards."
    )
    add_logging_args(parser)
    parser.add_argument(
        "--data_dir", default="Data", help="Directory containing .arrow shards"
    )
    parser.add_argument(
        "--output_dir",
        default="patchselect/out/local_selection",
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("all", "train", "valid", "test", "eval"),
        help="Which shard split to process",
    )
    parser.add_argument(
        "--limit_images",
        type=int,
        default=None,
        help="Optional maximum number of images to process",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size before descriptor downsampling",
    )
    parser.add_argument(
        "--patch_stride", type=int, default=256, help="Patch extraction stride"
    )
    parser.add_argument(
        "--magnification_factors",
        type=parse_magnification_factors,
        default=(1.0, 0.5, 0.25),
        help="Comma-separated image-scale factors used for local selection, for example 1.0,0.5,0.25",
    )
    parser.add_argument(
        "--descriptor_backend",
        default="cpu",
        choices=("cpu", "cucim"),
        help="Descriptor extraction backend",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for local selection",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Number of images to process per dispatch chunk",
    )
    parser.add_argument(
        "--auto_reduce_gpu_workers",
        action="store_true",
        help="When using the cucim backend, retry a chunk with fewer workers after GPU OOM",
    )
    parser.add_argument(
        "--gpu_ids",
        default=None,
        help="Comma-separated GPU ids for cucim workers; defaults to GPU 0",
    )
    parser.add_argument(
        "--downsample_size",
        type=int,
        default=None,
        help="Optional descriptor resolution; defaults to patch_size for full-resolution patch descriptors",
    )
    parser.add_argument(
        "--slide_stats_size",
        type=int,
        default=None,
        help="Optional slide-level stain-stat resolution; defaults to the full image",
    )
    parser.add_argument(
        "--disable_rle_mask",
        action="store_true",
        help="Ignore metadata rle_mask even when present",
    )
    parser.add_argument(
        "--rle_min_fraction",
        type=float,
        default=0.75,
        help="Minimum patch foreground overlap required when rle_mask is available",
    )
    parser.add_argument(
        "--local_keep_ratio",
        type=float,
        default=0.10,
        help="Fraction of valid patches to keep per image",
    )
    parser.add_argument(
        "--local_keep_min",
        type=int,
        default=1,
        help="Minimum selected patches per image",
    )
    parser.add_argument(
        "--local_keep_max",
        type=int,
        default=4,
        help="Maximum selected patches per image",
    )
    parser.add_argument(
        "--semantic_weight",
        type=float,
        default=0.50,
        help="Weight on semantic coverage in the local objective",
    )
    parser.add_argument(
        "--interface_weight",
        type=float,
        default=0.30,
        help="Weight on interface coverage in the local objective",
    )
    parser.add_argument(
        "--quality_weight",
        type=float,
        default=0.15,
        help="Weight on patch quality in the local objective",
    )
    parser.add_argument(
        "--redundancy_weight",
        type=float,
        default=0.20,
        help="Penalty weight on redundancy in the local objective",
    )
    parser.add_argument(
        "--nuisance_weight",
        type=float,
        default=0.35,
        help="Penalty weight on nuisance concentration in the local objective",
    )
    parser.add_argument(
        "--state_gain_bonus",
        type=float,
        default=0.35,
        help="Coverage bonus for selecting a locally unseen semantic state bin",
    )
    parser.add_argument(
        "--interface_gain_bonus",
        type=float,
        default=0.20,
        help="Coverage bonus for selecting a locally unseen interface bin",
    )
    parser.add_argument(
        "--flush_rows", type=int, default=5000, help="Rows per parquet flush"
    )
    parser.add_argument(
        "--save_selected_patches",
        action="store_true",
        help="Also save selected patch crops as image files for debugging",
    )
    parser.add_argument(
        "--selected_patch_dir",
        default=None,
        help="Directory for selected patch crops; defaults to <output_dir>/selected_patches",
    )
    parser.add_argument(
        "--image_format",
        default="jpg",
        choices=("jpg", "jpeg", "png"),
        help="Saved patch image format",
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=95, help="JPEG quality for saved patches"
    )
    return parser.parse_args()


def parse_gpu_ids(raw_value: str | None) -> list[int]:
    if not raw_value:
        return [0]
    return [int(token.strip()) for token in raw_value.split(",") if token.strip()]


def build_task(
    cfg: PatchSelectionConfig,
    bytes_data: bytes,
    metadata: dict,
    sample_id: str,
    source_shard: str,
    source_index: int,
    save_selected_patches: bool,
    selected_patch_dir: Path,
    gpu_id: int | None,
) -> dict:
    return {
        "cfg": config_to_payload(cfg),
        "bytes_data": bytes_data,
        "metadata": metadata,
        "sample_id": sample_id,
        "source_shard": source_shard,
        "source_index": source_index,
        "save_selected_patches": save_selected_patches,
        "selected_patch_dir": str(selected_patch_dir),
        "gpu_id": gpu_id,
    }


def run_tasks_once(tasks: list[dict], worker_count: int) -> dict:
    logger.debug(
        "Dispatching chunk with %d task(s) using %d worker(s).",
        len(tasks),
        worker_count,
    )
    if worker_count <= 1:
        results = [process_image_task(task) for task in tasks]
        for result in results:
            if result["status"] != "ok":
                return {"status": result["status"], "result": result}
        return {"status": "ok", "results": results}

    results = []
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(process_image_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result["status"] != "ok":
                    executor.shutdown(wait=False, cancel_futures=True)
                    return {"status": result["status"], "result": result}
                results.append(result)
    except OSError as exc:
        raise RuntimeError(
            "Failed to start worker processes. Retry with --num_workers 1, "
            "or run outside a restricted environment that blocks process spawning."
        ) from exc
    return {"status": "ok", "results": results}


def process_chunk_with_retries(
    tasks: list[dict],
    worker_count: int,
    allow_gpu_reduction: bool,
) -> tuple[list[dict], int, int]:
    current_workers = max(1, min(worker_count, len(tasks)))
    persistent_worker_count = max(1, worker_count)
    reductions = 0
    while True:
        logger.info(
            "Processing chunk of %d image(s) with %d worker(s)%s.",
            len(tasks),
            current_workers,
            " with GPU auto-reduction enabled" if allow_gpu_reduction else "",
        )
        outcome = run_tasks_once(tasks, current_workers)
        if outcome["status"] == "ok":
            logger.debug(
                "Chunk completed successfully with %d worker(s).", current_workers
            )
            return outcome["results"], persistent_worker_count, reductions
        if outcome["status"] == "oom" and allow_gpu_reduction and current_workers > 1:
            sample_id = outcome["result"].get("sample_id", "unknown")
            logger.warning(
                "OOM while processing sample %s. Retrying chunk with %d worker(s).",
                sample_id,
                current_workers - 1,
            )
            current_workers -= 1
            persistent_worker_count = current_workers
            reductions += 1
            continue
        result = outcome["result"]
        raise RuntimeError(
            f"Worker failed with status={result['status']} "
            f"for sample={result['sample_id']} source_index={result['source_index']}: {result.get('message', '')}"
        )


def flush_rows_if_needed(
    rows: list[dict], output_dir: Path, row_part_index: int, flush_rows: int
) -> int:
    if len(rows) < flush_rows:
        return row_part_index
    logger.info(
        "Flushing %d local candidate row(s) to parquet part %d.",
        len(rows),
        row_part_index,
    )
    row_part_index = write_rows_part(
        rows, output_dir / "candidates", "local_candidates", row_part_index
    )
    rows.clear()
    return row_part_index


def build_resume_signature(
    *,
    cfg: PatchSelectionConfig,
    files: list[Path],
    limit_images: int | None,
    save_selected_patches: bool,
    selected_patch_dir: Path,
) -> dict:
    return {
        "config": config_to_payload(cfg),
        "limit_images": limit_images,
        "save_selected_patches": save_selected_patches,
        "selected_patch_dir": str(selected_patch_dir.resolve()),
        "source_files": [str(path.resolve()) for path in files],
    }


def load_resume_state(output_dir: Path) -> dict | None:
    state_path = output_dir / RESUME_STATE_PATH
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_resume_state(output_dir: Path, state: dict) -> None:
    write_json(output_dir / RESUME_STATE_PATH, state)


def initialize_resume_state(output_dir: Path, signature: dict) -> dict:
    existing_state = load_resume_state(output_dir)
    if existing_state is None:
        state = {
            "completed": False,
            "signature": signature,
            "processed_images": 0,
            "skipped_images": 0,
            "selected_patches": 0,
            "candidate_parts": 0,
            "final_worker_count": 0,
            "gpu_worker_reductions": 0,
            "task_counter": 0,
            "cursor": {"shard_index": 0, "next_source_index": 0},
        }
        write_resume_state(output_dir, state)
        return state
    if existing_state.get("signature") != signature:
        raise RuntimeError(
            "Existing local-select resume state does not match the current extraction "
            "parameters. Use a new output directory or remove the previous outputs."
        )
    return existing_state


def checkpoint_progress(
    output_dir: Path,
    state: dict,
    *,
    processed_images: int,
    skipped_images: int,
    selected_patches: int,
    row_part_index: int,
    current_worker_count: int,
    worker_reductions: int,
    task_counter: int,
    shard_index: int,
    next_source_index: int,
    completed: bool = False,
) -> dict:
    state.update(
        {
            "completed": completed,
            "processed_images": processed_images,
            "skipped_images": skipped_images,
            "selected_patches": selected_patches,
            "candidate_parts": row_part_index,
            "final_worker_count": current_worker_count,
            "gpu_worker_reductions": worker_reductions,
            "task_counter": task_counter,
            "cursor": {
                "shard_index": shard_index,
                "next_source_index": next_source_index,
            },
        }
    )
    write_resume_state(output_dir, state)
    return state


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = (
        Path(args.selected_patch_dir)
        if args.selected_patch_dir
        else output_dir / "selected_patches"
    )
    gpu_ids = parse_gpu_ids(args.gpu_ids) if args.descriptor_backend == "cucim" else []

    cfg = PatchSelectionConfig(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        descriptor_backend=args.descriptor_backend,
        magnification_factors=args.magnification_factors,
        downsample_size=args.downsample_size,
        slide_stats_size=args.slide_stats_size,
        use_rle_mask=not args.disable_rle_mask,
        rle_min_fraction=args.rle_min_fraction,
        local_keep_ratio=args.local_keep_ratio,
        local_keep_min=args.local_keep_min,
        local_keep_max=args.local_keep_max,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        semantic_weight=args.semantic_weight,
        interface_weight=args.interface_weight,
        quality_weight=args.quality_weight,
        redundancy_weight=args.redundancy_weight,
        nuisance_weight=args.nuisance_weight,
        state_gain_bonus=args.state_gain_bonus,
        interface_gain_bonus=args.interface_gain_bonus,
    )

    files = discover_arrow_files(Path(args.data_dir), split=args.split)
    resume_signature = build_resume_signature(
        cfg=cfg,
        files=files,
        limit_images=args.limit_images,
        save_selected_patches=args.save_selected_patches,
        selected_patch_dir=patch_dir,
    )
    state = initialize_resume_state(output_dir, resume_signature)
    if state.get("completed"):
        logger.info(
            "Local selection already completed for %s; nothing to resume.",
            output_dir,
        )
        print(
            f"Processed {state['processed_images']} images, selected "
            f"{state['selected_patches']} patches, wrote {state['candidate_parts']} "
            f"parquet parts using {state['final_worker_count']} worker(s)."
        )
        return

    logger.info(
        "Starting local selection over %d shard(s) from %s with backend=%s, workers=%d, chunk_size=%d, log_level=%s.",
        len(files),
        args.data_dir,
        args.descriptor_backend,
        args.num_workers,
        args.chunk_size,
        args.log_level,
    )
    rows: list[dict] = []
    row_part_index = int(state.get("candidate_parts", 0))
    processed_images = int(state.get("processed_images", 0))
    skipped_images = int(state.get("skipped_images", 0))
    selected_patches = int(state.get("selected_patches", 0))
    current_worker_count = max(1, args.num_workers)
    worker_reductions = int(state.get("gpu_worker_reductions", 0))
    task_counter = int(state.get("task_counter", 0))
    cursor = state.get("cursor", {})
    start_shard_index = int(cursor.get("shard_index", 0))
    start_source_index = int(cursor.get("next_source_index", 0))

    for shard_index, shard_path in enumerate(files):
        if shard_index < start_shard_index:
            continue
        logger.info("Loading shard %s.", shard_path)
        dataset = load_arrow_shard(shard_path)
        progress = tqdm(desc=f"Processing {shard_path.name}", unit="img")
        chunk_tasks: list[dict] = []
        shard_start_source_index = start_source_index if shard_index == start_shard_index else 0
        for source_index, item in enumerate(dataset):
            if source_index < shard_start_source_index:
                continue
            if args.limit_images is not None and processed_images >= args.limit_images:
                break

            bytes_data = item.get("jpg", {}).get("bytes")
            if not bytes_data:
                logger.debug(
                    "Skipping %s[%d] because no jpg bytes are present.",
                    shard_path.name,
                    source_index,
                )
                skipped_images += 1
                processed_images += 1
                progress.update(1)
                state = checkpoint_progress(
                    output_dir,
                    state,
                    processed_images=processed_images,
                    skipped_images=skipped_images,
                    selected_patches=selected_patches,
                    row_part_index=row_part_index,
                    current_worker_count=current_worker_count,
                    worker_reductions=worker_reductions,
                    task_counter=task_counter,
                    shard_index=shard_index,
                    next_source_index=source_index + 1,
                )
                continue

            metadata = extract_custom_metadata(item)
            sample_id = str(metadata.get("md5") or f"{shard_path.stem}:{source_index}")
            gpu_id = None
            if gpu_ids:
                gpu_id = gpu_ids[task_counter % len(gpu_ids)]
            logger.debug(
                "Queueing sample %s from shard=%s index=%d gpu=%s.",
                sample_id,
                shard_path.name,
                source_index,
                gpu_id if gpu_id is not None else "cpu",
            )
            chunk_tasks.append(
                build_task(
                    cfg=cfg,
                    bytes_data=bytes_data,
                    metadata=metadata,
                    sample_id=sample_id,
                    source_shard=str(shard_path),
                    source_index=source_index,
                    save_selected_patches=args.save_selected_patches,
                    selected_patch_dir=patch_dir,
                    gpu_id=gpu_id,
                )
            )
            task_counter += 1

            if len(chunk_tasks) < args.chunk_size:
                continue

            results, current_worker_count, reductions = process_chunk_with_retries(
                chunk_tasks,
                worker_count=current_worker_count,
                allow_gpu_reduction=args.auto_reduce_gpu_workers
                and cfg.descriptor_backend == "cucim",
            )
            worker_reductions += reductions
            for result in results:
                processed_images += 1
                progress.update(1)
                if result["rows"]:
                    rows.extend(result["rows"])
                    selected_patches += len(result["rows"])
                    logger.debug(
                        "Sample %s produced %d selected patch row(s).",
                        result["sample_id"],
                        len(result["rows"]),
                    )
                else:
                    skipped_images += 1
                    logger.debug(
                        "Sample %s produced no retained patches.",
                        result["sample_id"],
                    )
            progress.set_postfix(
                {
                    "selected": selected_patches,
                    "processed": processed_images,
                    "workers": current_worker_count,
                }
            )
            row_part_index = flush_rows_if_needed(
                rows, output_dir, row_part_index, flush_rows=1
            )
            state = checkpoint_progress(
                output_dir,
                state,
                processed_images=processed_images,
                skipped_images=skipped_images,
                selected_patches=selected_patches,
                row_part_index=row_part_index,
                current_worker_count=current_worker_count,
                worker_reductions=worker_reductions,
                task_counter=task_counter,
                shard_index=shard_index,
                next_source_index=source_index + 1,
            )
            chunk_tasks.clear()

        if chunk_tasks and (
            args.limit_images is None or processed_images < args.limit_images
        ):
            results, current_worker_count, reductions = process_chunk_with_retries(
                chunk_tasks,
                worker_count=current_worker_count,
                allow_gpu_reduction=args.auto_reduce_gpu_workers
                and cfg.descriptor_backend == "cucim",
            )
            worker_reductions += reductions
            for result in results:
                processed_images += 1
                progress.update(1)
                if result["rows"]:
                    rows.extend(result["rows"])
                    selected_patches += len(result["rows"])
                else:
                    skipped_images += 1
            progress.set_postfix(
                {
                    "selected": selected_patches,
                    "processed": processed_images,
                    "workers": current_worker_count,
                }
            )
            row_part_index = flush_rows_if_needed(
                rows, output_dir, row_part_index, flush_rows=1
            )
            state = checkpoint_progress(
                output_dir,
                state,
                processed_images=processed_images,
                skipped_images=skipped_images,
                selected_patches=selected_patches,
                row_part_index=row_part_index,
                current_worker_count=current_worker_count,
                worker_reductions=worker_reductions,
                task_counter=task_counter,
                shard_index=shard_index,
                next_source_index=source_index + 1,
            )

        progress.close()
        logger.info(
            "Finished shard %s. Processed=%d skipped=%d selected_patches=%d.",
            shard_path.name,
            processed_images,
            skipped_images,
            selected_patches,
        )
        if args.limit_images is not None and processed_images >= args.limit_images:
            break

    if rows:
        logger.info(
            "Final flush of %d remaining local candidate row(s) to parquet part %d.",
            len(rows),
            row_part_index,
        )
        row_part_index = write_rows_part(
            rows, output_dir / "candidates", "local_candidates", row_part_index
        )

    summary = {
        "processed_images": processed_images,
        "skipped_images": skipped_images,
        "selected_patches": selected_patches,
        "candidate_parts": row_part_index,
        "source_files": [str(path) for path in files],
        "final_worker_count": current_worker_count,
        "gpu_worker_reductions": worker_reductions,
        "config": vars(args),
    }
    write_json(output_dir / "run_summary.json", summary)
    checkpoint_progress(
        output_dir,
        state,
        processed_images=processed_images,
        skipped_images=skipped_images,
        selected_patches=selected_patches,
        row_part_index=row_part_index,
        current_worker_count=current_worker_count,
        worker_reductions=worker_reductions,
        task_counter=task_counter,
        shard_index=len(files),
        next_source_index=0,
        completed=True,
    )
    logger.info("Wrote local selection summary to %s.", output_dir / "run_summary.json")
    print(
        f"Processed {processed_images} images, selected {selected_patches} patches, "
        f"wrote {row_part_index} parquet parts using {current_worker_count} worker(s)."
    )


if __name__ == "__main__":
    main()
