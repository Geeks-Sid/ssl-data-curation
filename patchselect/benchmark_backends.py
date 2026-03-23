"""Benchmark descriptor backends for patchselect."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from patchselect.arrow_utils import discover_arrow_files, extract_custom_metadata, load_arrow_shard, open_rgb_image
from patchselect.config import PatchSelectionConfig
from patchselect.descriptor_backend import available_descriptor_backends, backend_is_available
from patchselect.io_utils import write_json
from patchselect.pipeline import select_patches_from_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs cuCIM descriptor backends.")
    parser.add_argument("--data_dir", default="Data", help="Directory containing .arrow shards")
    parser.add_argument(
        "--split",
        default="train",
        choices=("all", "train", "valid", "test", "eval"),
        help="Which shard split to sample",
    )
    parser.add_argument("--limit_images", type=int, default=8, help="Number of benchmark images")
    parser.add_argument("--warmup_images", type=int, default=1, help="Warmup images per backend")
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
    parser.add_argument("--synthetic_size", type=int, default=3000, help="Side length for synthetic images")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    parser.add_argument("--patch_stride", type=int, default=256, help="Patch stride")
    parser.add_argument("--downsample_size", type=int, default=None, help="Optional patch descriptor downsample size")
    parser.add_argument(
        "--slide_stats_size",
        type=int,
        default=None,
        help="Optional slide-level stain-stat resolution",
    )
    parser.add_argument("--rle_min_fraction", type=float, default=0.75, help="Minimum RLE foreground overlap")
    parser.add_argument("--output_json", default=None, help="Optional JSON file for benchmark results")
    return parser.parse_args()


def maybe_sync_gpu(backend: str) -> None:
    if backend != "cucim":
        return
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def build_synthetic_samples(count: int, size: int) -> list[tuple[Image.Image, str, dict, str, int]]:
    rng = np.random.default_rng(7)
    samples: list[tuple[Image.Image, str, dict, str, int]] = []
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
        samples.append((image, f"synthetic-{index}", metadata, "synthetic", index))
    return samples


def load_arrow_samples(data_dir: Path, split: str, limit_images: int) -> list[tuple[Image.Image, str, dict, str, int]]:
    files = discover_arrow_files(data_dir, split=split)
    samples: list[tuple[Image.Image, str, dict, str, int]] = []
    for shard_path in files:
        dataset = load_arrow_shard(shard_path)
        for source_index, item in enumerate(dataset):
            bytes_data = item.get("jpg", {}).get("bytes")
            if not bytes_data:
                continue
            metadata = extract_custom_metadata(item)
            image = open_rgb_image(bytes_data)
            sample_id = str(metadata.get("md5") or f"{shard_path.stem}:{source_index}")
            samples.append((image, sample_id, metadata, str(shard_path), source_index))
            if len(samples) >= limit_images:
                return samples
    return samples


def load_samples(args: argparse.Namespace) -> tuple[list[tuple[Image.Image, str, dict, str, int]], str]:
    if args.synthetic_images > 0:
        return build_synthetic_samples(args.synthetic_images, args.synthetic_size), "synthetic"
    try:
        samples = load_arrow_samples(Path(args.data_dir), args.split, args.limit_images)
    except FileNotFoundError:
        samples = []
    if samples:
        return samples, "arrow"
    fallback_count = max(args.limit_images, 4)
    return build_synthetic_samples(fallback_count, args.synthetic_size), "synthetic_fallback"


def run_backend(
    backend: str,
    samples: list[tuple[Image.Image, str, dict, str, int]],
    args: argparse.Namespace,
) -> dict[str, object]:
    result: dict[str, object] = {
        "backend": backend,
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
        for image, sample_id, metadata, source_shard, source_index in samples[:warmup]:
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
        for image, sample_id, metadata, source_shard, source_index in measured:
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


def main() -> None:
    args = parse_args()
    samples, source = load_samples(args)
    backends = ["cpu", "cucim"] if args.backend == "both" else [args.backend]
    results = [run_backend(backend, samples, args) for backend in backends]

    speedup = None
    by_name = {result["backend"]: result for result in results if "total_seconds" in result}
    if "cpu" in by_name and "cucim" in by_name:
        speedup = by_name["cpu"]["total_seconds"] / max(by_name["cucim"]["total_seconds"], 1e-6)

    payload = {
        "sample_source": source,
        "available_backends": available_descriptor_backends(),
        "sample_count": len(samples),
        "config": vars(args),
        "results": results,
        "cpu_over_cucim_speedup": speedup,
    }

    if args.output_json:
        write_json(Path(args.output_json), payload)

    print(f"Benchmark source: {source} ({len(samples)} images)")
    for result in results:
        if "error" in result:
            print(f"- {result['backend']}: {result['error']}")
            continue
        print(
            f"- {result['backend']}: {result['seconds_per_image']:.4f}s/img, "
            f"{result['images_per_second']:.3f} img/s, "
            f"{result['selected_patches_per_second']:.1f} selected patches/s"
        )
    if speedup is not None:
        print(f"CPU / cuCIM speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
