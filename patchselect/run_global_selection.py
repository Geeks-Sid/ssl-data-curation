"""Run global balancing over locally selected patch candidates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from patchselect.config import GlobalSelectionConfig
from patchselect.export_tars import export_selected_patches_to_tars
from patchselect.io_utils import write_json
from patchselect.logging_utils import add_logging_args, configure_logging
from patchselect.selection import run_global_selection

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global balancing over local patch candidates."
    )
    add_logging_args(parser)
    parser.add_argument(
        "--candidate_dir",
        default="patchselect/out/local_selection/candidates",
        help="Directory containing local candidate parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="patchselect/out/global_selection",
        help="Output directory",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        required=True,
        help="Target number of final selected patches",
    )
    parser.add_argument(
        "--bin_columns",
        default="tissue,is_cancer,state_bin,interface_bin",
        help="Comma-separated column list used for balancing",
    )
    parser.add_argument(
        "--bin_alpha",
        type=float,
        default=0.5,
        help="Tempering coefficient for bin quotas",
    )
    parser.add_argument(
        "--min_quota",
        type=int,
        default=0,
        help="Optional minimum quota per non-empty bin",
    )
    parser.add_argument(
        "--utility_column",
        default="objective_score",
        help="Column used for within-bin ranking",
    )
    parser.add_argument(
        "--dataframe_backend",
        default="auto",
        choices=("auto", "pandas", "cudf"),
        help="Tabular backend for global selection. 'auto' uses cuDF when available, otherwise pandas.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars for the global selection and tar export passes.",
    )
    parser.add_argument(
        "--export_tars",
        action="store_true",
        help="Pack final selected patches into tar archives grouped by source Arrow shard",
    )
    parser.add_argument(
        "--export_patches",
        dest="export_tars",
        action="store_true",
        help="Alias for --export_tars; enables post-selection patch materialization",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional Arrow shard directory used to resolve source_shard paths during tar export",
    )
    parser.add_argument(
        "--tar_output_dir",
        default=None,
        help="Optional output directory for tar archives and exporter working files; defaults to <output_dir>/final_selection_tars",
    )
    parser.add_argument(
        "--tar_output_mode",
        default="tar",
        choices=("tar", "files", "both"),
        help="Whether exported selected patches are written as tar archives, loose image files, or both",
    )
    parser.add_argument(
        "--tar_image_output_dir",
        default=None,
        help="Optional output directory for loose patch image files when --tar_output_mode is files/both",
    )
    parser.add_argument(
        "--tar_image_format",
        default="jpg",
        choices=("jpg", "jpeg", "png"),
        help="Patch encoding format for tar members and loose patch files",
    )
    parser.add_argument(
        "--tar_jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality used when tar members are written as jpg/jpeg",
    )
    parser.add_argument(
        "--tar_default_patch_size",
        type=int,
        default=256,
        help="Fallback crop size when patch_size is absent from a final manifest row",
    )
    parser.add_argument(
        "--tar_compression",
        default="none",
        choices=("none", "gz"),
        help="Tar compression mode; plain .tar is the default",
    )
    parser.add_argument(
        "--tar_num_workers",
        type=int,
        default=1,
        help="Number of worker processes for shard-level tar export parallelism",
    )
    parser.add_argument(
        "--tar_max_images_per_tar",
        type=int,
        default=None,
        help="Optional cap on image members per tar; large shards are split into numbered tar chunks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    candidate_dir = Path(args.candidate_dir)
    candidate_files = sorted(candidate_dir.glob("*.parquet"))
    if not candidate_files:
        raise FileNotFoundError(f"No parquet candidate files found in {candidate_dir}")
    logger.info(
        "Starting global selection from %d candidate parquet file(s) in %s with log_level=%s.",
        len(candidate_files),
        candidate_dir,
        args.log_level,
    )

    config = GlobalSelectionConfig(
        target_size=args.target_size,
        bin_columns=tuple(
            column.strip() for column in args.bin_columns.split(",") if column.strip()
        ),
        bin_alpha=args.bin_alpha,
        utility_column=args.utility_column,
        dataframe_backend=args.dataframe_backend,
        show_progress=not args.no_progress,
        per_bin_min_quota=args.min_quota,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_global_selection(candidate_files, output_dir, config)
    tar_result = None
    if args.export_tars:
        logger.info("Tar export requested; scanning final selection parquet files.")
        final_selection_dir = output_dir / "final_selection"
        final_selection_files = sorted(final_selection_dir.glob("*.parquet"))
        if final_selection_files:
            tar_output_dir = (
                Path(args.tar_output_dir)
                if args.tar_output_dir
                else output_dir / "final_selection_tars"
            )
            tar_result = export_selected_patches_to_tars(
                final_selection_files=final_selection_files,
                output_dir=tar_output_dir,
                data_dir=Path(args.data_dir) if args.data_dir else None,
                image_format=args.tar_image_format,
                jpeg_quality=args.tar_jpeg_quality,
                default_patch_size=args.tar_default_patch_size,
                compression=args.tar_compression,
                output_mode=args.tar_output_mode,
                image_output_dir=(
                    Path(args.tar_image_output_dir)
                    if args.tar_image_output_dir
                    else None
                ),
                num_workers=args.tar_num_workers,
                max_images_per_tar=args.tar_max_images_per_tar,
                show_progress=not args.no_progress,
            )
            logger.info(
                "Patch export completed with %d tar member(s), %d loose image file(s), across %d tar file(s) using %d worker(s).",
                tar_result["written_members"],
                tar_result["written_files"],
                tar_result["tar_count"],
                tar_result["num_workers"],
            )
        else:
            logger.warning(
                "Tar export requested but no final selection parquet files were produced."
            )
            tar_result = {
                "final_selection_files": 0,
                "tar_count": 0,
                "selected_rows": 0,
                "written_members": 0,
                "written_files": 0,
                "output_mode": args.tar_output_mode,
                "compression": args.tar_compression,
                "image_format": args.tar_image_format,
                "jpeg_quality": args.tar_jpeg_quality,
                "default_patch_size": args.tar_default_patch_size,
                "num_workers": args.tar_num_workers,
                "max_images_per_tar": args.tar_max_images_per_tar or 0,
                "image_output_dir": (
                    args.tar_image_output_dir if args.tar_image_output_dir else ""
                ),
            }
    summary = {
        "config": vars(args),
        **result,
    }
    if tar_result is not None:
        summary["tar_export"] = tar_result
    write_json(output_dir / "run_summary.json", summary)
    logger.info("Wrote global selection summary to %s.", output_dir / "run_summary.json")
    message = f"Global selection complete: {result['selected_rows']} rows selected across {result['bin_count']} bins."
    if tar_result is not None:
        message += (
            " Patch export wrote "
            f"{tar_result['written_members']} tar member(s)"
        )
        if tar_result["written_files"] > 0:
            message += f" and {tar_result['written_files']} loose image file(s)"
        message += f" across {tar_result['tar_count']} tar file(s)."
    print(message)


if __name__ == "__main__":
    main()
