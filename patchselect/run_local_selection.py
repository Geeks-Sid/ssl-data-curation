"""Run per-image local patch selection over Arrow shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from patchselect.arrow_utils import discover_arrow_files, extract_custom_metadata, load_arrow_shard, open_rgb_image
from patchselect.config import PatchSelectionConfig
from patchselect.io_utils import write_json, write_rows_part
from patchselect.pipeline import save_selected_patch, select_patches_from_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local stain-aware patch selection on Arrow shards.")
    parser.add_argument("--data_dir", default="Data", help="Directory containing .arrow shards")
    parser.add_argument("--output_dir", default="patchselect/out/local_selection", help="Output directory")
    parser.add_argument(
        "--split",
        default="train",
        choices=("all", "train", "valid", "test", "eval"),
        help="Which shard split to process",
    )
    parser.add_argument("--limit_images", type=int, default=None, help="Optional maximum number of images to process")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size before descriptor downsampling")
    parser.add_argument("--patch_stride", type=int, default=256, help="Patch extraction stride")
    parser.add_argument(
        "--descriptor_backend",
        default="cpu",
        choices=("cpu", "cucim"),
        help="Descriptor extraction backend",
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
    parser.add_argument("--local_keep_ratio", type=float, default=0.10, help="Fraction of valid patches to keep per image")
    parser.add_argument("--local_keep_min", type=int, default=1, help="Minimum selected patches per image")
    parser.add_argument("--local_keep_max", type=int, default=4, help="Maximum selected patches per image")
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
    parser.add_argument("--flush_rows", type=int, default=5000, help="Rows per parquet flush")
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
    parser.add_argument("--image_format", default="jpg", choices=("jpg", "jpeg", "png"), help="Saved patch image format")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for saved patches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = Path(args.selected_patch_dir) if args.selected_patch_dir else output_dir / "selected_patches"

    cfg = PatchSelectionConfig(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        descriptor_backend=args.descriptor_backend,
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
    rows: list[dict] = []
    row_part_index = 0
    processed_images = 0
    skipped_images = 0
    selected_patches = 0

    for shard_path in files:
        dataset = load_arrow_shard(shard_path)
        progress = tqdm(dataset, desc=f"Processing {shard_path.name}", unit="img")
        for source_index, item in enumerate(progress):
            if args.limit_images is not None and processed_images >= args.limit_images:
                break

            bytes_data = item.get("jpg", {}).get("bytes")
            if not bytes_data:
                skipped_images += 1
                continue

            metadata = extract_custom_metadata(item)
            image = open_rgb_image(bytes_data)
            sample_id = metadata.get("md5") or f"{shard_path.stem}:{source_index}"
            selected_rows, selected_patch_records = select_patches_from_image(
                image=image,
                sample_id=str(sample_id),
                metadata=metadata,
                cfg=cfg,
                source_shard=str(shard_path),
                source_index=source_index,
            )
            if not selected_rows:
                skipped_images += 1
                processed_images += 1
                continue

            rows.extend(selected_rows)
            selected_patches += len(selected_rows)
            processed_images += 1
            progress.set_postfix({"selected": selected_patches, "processed": processed_images})

            if args.save_selected_patches:
                for patch_rgb, record in selected_patch_records:
                    save_selected_patch(patch_rgb, record, patch_dir, cfg)

            if len(rows) >= args.flush_rows:
                row_part_index = write_rows_part(rows, output_dir / "candidates", "local_candidates", row_part_index)
                rows.clear()

        if args.limit_images is not None and processed_images >= args.limit_images:
            break

    if rows:
        row_part_index = write_rows_part(rows, output_dir / "candidates", "local_candidates", row_part_index)

    summary = {
        "processed_images": processed_images,
        "skipped_images": skipped_images,
        "selected_patches": selected_patches,
        "candidate_parts": row_part_index,
        "source_files": [str(path) for path in files],
        "config": vars(args),
    }
    write_json(output_dir / "run_summary.json", summary)
    print(f"Processed {processed_images} images, selected {selected_patches} patches, wrote {row_part_index} parquet parts.")


if __name__ == "__main__":
    main()
