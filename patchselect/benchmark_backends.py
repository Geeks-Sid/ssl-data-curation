"""Benchmark descriptor backends for patchselect."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from patchselect.arrow_utils import (
    discover_arrow_files,
    extract_custom_metadata,
    load_arrow_shard,
    open_rgb_image,
)
from patchselect.config import PatchSelectionConfig
from patchselect.descriptor_backend import (
    available_descriptor_backends,
    backend_is_available,
)
from patchselect.io_utils import write_json
from patchselect.pipeline import select_patches_from_image
from patchselect.run_local_selection import (
    build_task,
    parse_gpu_ids,
    process_chunk_with_retries,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs cuCIM descriptor backends."
    )
    parser.add_argument(
        "--data_dir", default="Data", help="Directory containing .arrow shards"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("all", "train", "valid", "test", "eval"),
        help="Which shard split to sample",
    )
    parser.add_argument(
        "--limit_images", type=int, default=8, help="Number of benchmark images"
    )
    parser.add_argument(
        "--warmup_images", type=int, default=1, help="Warmup images per backend"
    )
    parser.add_argument(
        "--backend",
        default="both",
        choices=("cpu", "cucim", "both"),
        help="Which backend(s) to benchmark",
    )
    parser.add_argument(
        "--synthetic_images",
        type=int,
        default=0,
        help="Generate synthetic benchmark images instead of reading Arrow shards",
    )
    parser.add_argument(
        "--synthetic_size",
        type=int,
        default=3000,
        help="Side length for synthetic images",
    )
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    parser.add_argument("--patch_stride", type=int, default=256, help="Patch stride")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Worker processes for benchmark execution",
    )
    parser.add_argument(
        "--worker_counts",
        default=None,
        help="Optional comma-separated worker counts to sweep; defaults to --num_workers",
    )
    parser.add_argument(
        "--auto_reduce_gpu_workers",
        action="store_true",
        help="Retry cucim benchmark chunks with fewer workers after GPU OOM",
    )
    parser.add_argument(
        "--gpu_ids", default=None, help="Comma-separated GPU ids for cucim worker tasks"
    )
    parser.add_argument(
        "--downsample_size",
        type=int,
        default=None,
        help="Optional patch descriptor downsample size",
    )
    parser.add_argument(
        "--slide_stats_size",
        type=int,
        default=None,
        help="Optional slide-level stain-stat resolution",
    )
    parser.add_argument(
        "--rle_min_fraction",
        type=float,
        default=0.75,
        help="Minimum RLE foreground overlap",
    )
    parser.add_argument(
        "--output_json", default=None, help="Optional JSON file for benchmark results"
    )
    parser.add_argument(
        "--output_plot",
        default=None,
        help="Optional image path for worker-scaling plots",
    )
    return parser.parse_args()


def maybe_sync_gpu(backend: str) -> None:
    if backend != "cucim":
        return
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def build_synthetic_samples(
    count: int, size: int
) -> list[tuple[bytes, str, dict, str, int]]:
    rng = np.random.default_rng(7)
    samples: list[tuple[bytes, str, dict, str, int]] = []
    for index in range(count):
        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for shape_index in range(10):
            x0 = int(rng.integers(0, max(1, size - size // 3)))
            y0 = int(rng.integers(0, max(1, size - size // 3)))
            x1 = int(min(size, x0 + rng.integers(size // 8, size // 3)))
            y1 = int(min(size, y0 + rng.integers(size // 8, size // 3)))
            fill = tuple(int(value) for value in rng.integers(50, 190, size=3))
            if shape_index % 2 == 0:
                draw.rectangle([x0, y0, x1, y1], fill=fill)
            else:
                draw.ellipse([x0, y0, x1, y1], fill=fill)
        metadata = {
            "gene": f"SYNTH{index % 4}",
            "tissue": f"synthetic_tissue_{index % 3}",
            "cell_type": "synthetic",
            "diagnosis": "synthetic benchmark image",
        }
        samples.append(
            (
                image_to_jpeg_bytes(image),
                f"synthetic-{index}",
                metadata,
                "synthetic",
                index,
            )
        )
    return samples


def load_arrow_samples(
    data_dir: Path, split: str, limit_images: int
) -> list[tuple[bytes, str, dict, str, int]]:
    files = discover_arrow_files(data_dir, split=split)
    samples: list[tuple[bytes, str, dict, str, int]] = []
    for shard_path in files:
        dataset = load_arrow_shard(shard_path)
        for source_index, item in enumerate(dataset):
            bytes_data = item.get("jpg", {}).get("bytes")
            if not bytes_data:
                continue
            metadata = extract_custom_metadata(item)
            sample_id = str(metadata.get("md5") or f"{shard_path.stem}:{source_index}")
            samples.append(
                (bytes_data, sample_id, metadata, str(shard_path), source_index)
            )
            if len(samples) >= limit_images:
                return samples
    return samples


def load_samples(
    args: argparse.Namespace,
) -> tuple[list[tuple[bytes, str, dict, str, int]], str]:
    if args.synthetic_images > 0:
        return (
            build_synthetic_samples(args.synthetic_images, args.synthetic_size),
            "synthetic",
        )
    try:
        samples = load_arrow_samples(Path(args.data_dir), args.split, args.limit_images)
    except FileNotFoundError:
        samples = []
    if samples:
        return samples, "arrow"
    fallback_count = max(args.limit_images, 4)
    return (
        build_synthetic_samples(fallback_count, args.synthetic_size),
        "synthetic_fallback",
    )


def parse_worker_counts(args: argparse.Namespace) -> list[int]:
    if not args.worker_counts:
        return [max(1, args.num_workers)]
    counts = []
    for token in args.worker_counts.split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if count < 1:
            raise ValueError("--worker_counts values must be >= 1")
        counts.append(count)
    if not counts:
        raise ValueError("--worker_counts did not contain any valid worker counts")
    return sorted(set(counts))


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def run_backend(
    backend: str,
    samples: list[tuple[bytes, str, dict, str, int]],
    args: argparse.Namespace,
    worker_count: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "backend": backend,
        "requested_workers": worker_count,
        "available": backend_is_available(backend),
    }
    if not result["available"]:
        result["error"] = f"Backend '{backend}' is not available in this environment."
        return result

    cfg = PatchSelectionConfig(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        descriptor_backend=backend,
        downsample_size=args.downsample_size,
        slide_stats_size=args.slide_stats_size,
        rle_min_fraction=args.rle_min_fraction,
    )

    warmup = min(args.warmup_images, len(samples))
    try:
        for bytes_data, sample_id, metadata, source_shard, source_index in samples[
            :warmup
        ]:
            image = open_rgb_image(bytes_data)
            select_patches_from_image(
                image=image,
                sample_id=sample_id,
                metadata=metadata,
                cfg=cfg,
                source_shard=source_shard,
                source_index=source_index,
            )
    except Exception as exc:
        result["error"] = f"Backend '{backend}' failed during warmup: {exc}"
        return result
    maybe_sync_gpu(backend)

    measured = samples[warmup:] if warmup < len(samples) else samples
    if not measured:
        result["error"] = "No samples available for timing."
        return result

    total_selected_patches = 0
    total_valid_patches = 0
    start = time.perf_counter()
    try:
        if worker_count > 1:
            gpu_ids = parse_gpu_ids(args.gpu_ids) if backend == "cucim" else []
            tasks = []
            for task_index, (
                bytes_data,
                sample_id,
                metadata,
                source_shard,
                source_index,
            ) in enumerate(measured):
                gpu_id = gpu_ids[task_index % len(gpu_ids)] if gpu_ids else None
                tasks.append(
                    build_task(
                        cfg=cfg,
                        bytes_data=bytes_data,
                        metadata=metadata,
                        sample_id=sample_id,
                        source_shard=source_shard,
                        source_index=source_index,
                        save_selected_patches=False,
                        selected_patch_dir=Path("patchselect/out/benchmark_selected"),
                        gpu_id=gpu_id,
                    )
                )
            worker_results, effective_workers, reductions = process_chunk_with_retries(
                tasks,
                worker_count=worker_count,
                allow_gpu_reduction=args.auto_reduce_gpu_workers and backend == "cucim",
            )
            result["effective_workers"] = effective_workers
            result["gpu_worker_reductions"] = reductions
            for worker_result in worker_results:
                rows = worker_result["rows"]
                total_selected_patches += len(rows)
                if rows:
                    total_valid_patches += int(rows[0]["local_valid_patch_count"])
        else:
            for bytes_data, sample_id, metadata, source_shard, source_index in measured:
                image = open_rgb_image(bytes_data)
                rows, _ = select_patches_from_image(
                    image=image,
                    sample_id=sample_id,
                    metadata=metadata,
                    cfg=cfg,
                    source_shard=source_shard,
                    source_index=source_index,
                )
                total_selected_patches += len(rows)
                if rows:
                    total_valid_patches += int(rows[0]["local_valid_patch_count"])
    except Exception as exc:
        result["error"] = f"Backend '{backend}' failed during timed execution: {exc}"
        return result
    maybe_sync_gpu(backend)
    elapsed = time.perf_counter() - start

    image_count = len(measured)
    result.update(
        {
            "image_count": image_count,
            "total_seconds": elapsed,
            "seconds_per_image": elapsed / max(image_count, 1),
            "images_per_second": image_count / max(elapsed, 1e-6),
            "total_selected_patches": total_selected_patches,
            "selected_patches_per_second": total_selected_patches / max(elapsed, 1e-6),
            "total_valid_patches": total_valid_patches,
            "valid_patches_per_second": total_valid_patches / max(elapsed, 1e-6),
            "warmup_images": warmup,
        }
    )
    return result


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    by_backend: dict[str, list[dict[str, object]]] = {}
    for result in results:
        if "total_seconds" not in result:
            continue
        by_backend.setdefault(str(result["backend"]), []).append(result)

    optimal_by_backend: dict[str, dict[str, object]] = {}
    for backend, backend_results in by_backend.items():
        best = max(backend_results, key=lambda item: float(item["images_per_second"]))
        optimal_by_backend[backend] = {
            "requested_workers": int(best["requested_workers"]),
            "effective_workers": int(best.get("effective_workers", best["requested_workers"])),
            "images_per_second": float(best["images_per_second"]),
            "seconds_per_image": float(best["seconds_per_image"]),
        }

    speedup_by_workers: list[dict[str, float | int]] = []
    cpu_by_worker = {
        int(item["requested_workers"]): item
        for item in by_backend.get("cpu", [])
    }
    gpu_by_worker = {
        int(item["requested_workers"]): item
        for item in by_backend.get("cucim", [])
    }
    for worker_count in sorted(set(cpu_by_worker) & set(gpu_by_worker)):
        cpu_time = float(cpu_by_worker[worker_count]["total_seconds"])
        gpu_time = float(gpu_by_worker[worker_count]["total_seconds"])
        speedup_by_workers.append(
            {
                "requested_workers": worker_count,
                "cpu_over_cucim_speedup": cpu_time / max(gpu_time, 1e-6),
            }
        )

    return {
        "optimal_by_backend": optimal_by_backend,
        "cpu_over_cucim_speedup_by_workers": speedup_by_workers,
    }


def plot_worker_scaling(results: list[dict[str, object]], output_path: Path) -> None:
    timed_results = [result for result in results if "total_seconds" in result]
    if not timed_results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_specs = [
        ("images_per_second", "Images / second"),
        ("seconds_per_image", "Seconds / image"),
    ]
    styles = {
        "cpu": {"marker": "o", "label": "CPU"},
        "cucim": {"marker": "s", "label": "cuCIM"},
    }

    by_backend: dict[str, list[dict[str, object]]] = {}
    for result in timed_results:
        by_backend.setdefault(str(result["backend"]), []).append(result)

    for axis, (metric_key, metric_label) in zip(axes, metric_specs):
        for backend, backend_results in by_backend.items():
            ordered = sorted(
                backend_results, key=lambda item: int(item["requested_workers"])
            )
            style = styles.get(backend, {"marker": "o", "label": backend})
            x_values = [int(item["requested_workers"]) for item in ordered]
            y_values = [float(item[metric_key]) for item in ordered]
            axis.plot(
                x_values,
                y_values,
                marker=style["marker"],
                linewidth=2,
                label=style["label"],
            )

            if metric_key == "images_per_second":
                best = max(ordered, key=lambda item: float(item[metric_key]))
            else:
                best = min(ordered, key=lambda item: float(item[metric_key]))
            best_x = int(best["requested_workers"])
            best_y = float(best[metric_key])
            axis.scatter([best_x], [best_y], s=60, zorder=3)
            axis.annotate(
                f"best={best_x}",
                xy=(best_x, best_y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

        axis.set_xlabel("Requested workers")
        axis.set_ylabel(metric_label)
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle("Patchselect Backend Scaling by Worker Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.data_dir = normalize_optional_text(args.data_dir) or "Data"
    args.split = normalize_optional_text(args.split) or "train"
    args.backend = normalize_optional_text(args.backend) or "both"
    args.gpu_ids = normalize_optional_text(args.gpu_ids)
    args.output_json = normalize_optional_text(args.output_json)
    args.output_plot = normalize_optional_text(args.output_plot)
    args.worker_counts = normalize_optional_text(args.worker_counts)
    worker_counts = parse_worker_counts(args)
    samples, source = load_samples(args)
    backends = ["cpu", "cucim"] if args.backend == "both" else [args.backend]
    results = [
        run_backend(backend, samples, args, worker_count)
        for worker_count in worker_counts
        for backend in backends
    ]
    summary = summarize_results(results)

    payload = {
        "sample_source": source,
        "available_backends": available_descriptor_backends(),
        "sample_count": len(samples),
        "config": vars(args),
        "worker_counts": worker_counts,
        "results": results,
        "summary": summary,
    }

    if args.output_json:
        write_json(Path(args.output_json), payload)
    if args.output_plot:
        plot_worker_scaling(results, Path(args.output_plot))

    print(f"Benchmark source: {source} ({len(samples)} images)")
    for result in results:
        if "error" in result:
            print(
                f"- {result['backend']} @ {result['requested_workers']} worker(s): "
                f"{result['error']}"
            )
            continue
        print(
            f"- {result['backend']} @ {result['requested_workers']} worker(s)"
            f" (effective {result.get('effective_workers', result['requested_workers'])}): "
            f"{result['seconds_per_image']:.4f}s/img, "
            f"{result['images_per_second']:.3f} img/s, "
            f"{result['selected_patches_per_second']:.1f} selected patches/s"
        )
    for backend, best in summary["optimal_by_backend"].items():
        print(
            f"Optimal {backend}: requested={best['requested_workers']} worker(s), "
            f"effective={best['effective_workers']}, "
            f"{best['images_per_second']:.3f} img/s"
        )
    for item in summary["cpu_over_cucim_speedup_by_workers"]:
        print(
            f"CPU / cuCIM speedup @ {item['requested_workers']} worker(s): "
            f"{item['cpu_over_cucim_speedup']:.3f}x"
        )
    if args.output_plot:
        print(f"Saved worker-scaling plot to {args.output_plot}")


if __name__ == "__main__":
    main()
