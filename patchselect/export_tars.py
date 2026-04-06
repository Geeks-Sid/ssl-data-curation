"""Export globally selected patches into tar archives grouped by source Arrow shard."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import logging
import multiprocessing as mp
import sys
import tarfile
from pathlib import Path
from typing import Any
import uuid

import pandas as pd
import pyarrow.dataset as ds
from PIL import Image
from tqdm.auto import tqdm

from patchselect.arrow_utils import (
    discover_arrow_files,
    load_arrow_shard,
    open_rgb_image,
    slugify,
)
from patchselect.io_utils import write_dataframe_part, write_json

logger = logging.getLogger(__name__)


def maybe_make_progress(
    *,
    total: int,
    desc: str,
    unit: str,
    enabled: bool,
) -> Any:
    if not enabled or not sys.stderr.isatty():
        return None
    return tqdm(total=total, desc=desc, unit=unit)


def parse_image_format(value: str) -> str:
    normalized = value.strip().lower().lstrip(".")
    if normalized not in {"jpg", "jpeg", "png"}:
        raise argparse.ArgumentTypeError(
            f"invalid choice: {value!r} (choose from 'jpg', 'jpeg', 'png')"
        )
    return normalized


def shard_token(source_shard: str) -> str:
    path = Path(source_shard)
    stem = slugify(path.stem, "arrow_shard")
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def build_tar_name(
    source_shard: str, compression: str, chunk_index: int | None = None
) -> str:
    suffix = ".tar.gz" if compression == "gz" else ".tar"
    name = shard_token(source_shard)
    if chunk_index is not None:
        name = f"{name}_part{chunk_index:06d}"
    return f"{name}{suffix}"


def build_chunked_tar_path(base_tar_path: Path, chunk_index: int) -> Path:
    if base_tar_path.name.endswith(".tar.gz"):
        base_name = base_tar_path.name[: -len(".tar.gz")]
        suffix = ".tar.gz"
    else:
        base_name = base_tar_path.stem
        suffix = base_tar_path.suffix
    return base_tar_path.with_name(f"{base_name}_part{chunk_index:06d}{suffix}")


def build_patch_member_name(row: dict, image_format: str) -> str:
    suffix = image_format.lower().lstrip(".")
    sample_slug = slugify(row.get("sample_slug") or row.get("sample_id"), "sample")
    scale_token = ""
    scale_level = row.get("scale_level")
    if scale_level is not None and not pd.isna(scale_level):
        scale_token = f"__m{int(scale_level):02d}"
    return (
        f"{sample_slug}"
        f"__i{int(row['source_index']):07d}"
        f"__p{int(row['patch_index']):04d}"
        f"{scale_token}"
        f"__x{int(row['patch_x']):05d}"
        f"__y{int(row['patch_y']):05d}.{suffix}"
    )


def build_patch_output_path(row: dict, output_dir: Path, image_format: str) -> Path:
    shard_dir = output_dir / shard_token(str(row.get("source_shard", "arrow_shard")))
    return shard_dir / build_patch_member_name(row, image_format=image_format)


def resolve_shard_path(
    source_shard: str, data_dir: Path | None, lookup: dict[str, Path]
) -> Path:
    shard_path = Path(source_shard)
    if shard_path.exists():
        return shard_path
    fallback = lookup.get(shard_path.name.lower())
    if fallback is not None:
        return fallback
    location = f" or --data_dir {data_dir}" if data_dir is not None else ""
    raise FileNotFoundError(f"Could not resolve Arrow shard {source_shard!r}{location}")


def build_data_dir_lookup(data_dir: Path | None) -> dict[str, Path]:
    if data_dir is None:
        return {}
    try:
        files = discover_arrow_files(data_dir, split="all")
    except FileNotFoundError:
        logger.warning(
            "No .arrow files found under data_dir=%s; tar export will rely on source_shard paths stored in the manifest.",
            data_dir,
        )
        return {}
    return {path.name.lower(): path for path in files}


def encode_patch_image(
    patch: Image.Image, image_format: str, jpeg_quality: int
) -> bytes:
    buffer = io.BytesIO()
    if image_format.lower() in {"jpg", "jpeg"}:
        patch.save(buffer, format="JPEG", quality=jpeg_quality)
    else:
        patch.save(buffer, format=image_format.upper())
    return buffer.getvalue()


def resolve_patch_size(value: object, default_patch_size: int) -> int:
    if value is None or pd.isna(value):
        return default_patch_size
    return int(value)


def resolve_scaled_image_size(
    row: Any, current_image: Image.Image
) -> tuple[int, int]:
    scaled_width = getattr(row, "scaled_image_width", None)
    scaled_height = getattr(row, "scaled_image_height", None)
    if (
        scaled_width is not None
        and scaled_height is not None
        and not pd.isna(scaled_width)
        and not pd.isna(scaled_height)
    ):
        return int(scaled_width), int(scaled_height)

    scale_factor = getattr(row, "scale_factor", None)
    if scale_factor is None or pd.isna(scale_factor):
        return current_image.width, current_image.height

    factor = float(scale_factor)
    return (
        max(1, int(round(current_image.width * factor))),
        max(1, int(round(current_image.height * factor))),
    )


def partition_final_selection_by_shard(
    final_selection_files: list[Path],
    partition_root: Path,
) -> dict[str, str]:
    partition_root.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(
        [str(path) for path in final_selection_files], format="parquet"
    )
    part_index = 0
    token_to_shard: dict[str, str] = {}

    for batch in dataset.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        for source_shard, group in frame.groupby("source_shard", sort=False):
            token = shard_token(str(source_shard))
            token_to_shard[token] = str(source_shard)
            out_dir = partition_root / token
            out_dir.mkdir(parents=True, exist_ok=True)
            part_path = out_dir / f"part-{part_index:06d}.parquet"
            write_dataframe_part(group, part_path)
            part_index += 1

    mapping_rows = [
        {
            "partition": token,
            "source_shard": source_shard,
            "tar_name": build_tar_name(source_shard, compression="none"),
        }
        for token, source_shard in token_to_shard.items()
    ]
    if mapping_rows:
        pd.DataFrame(mapping_rows).to_parquet(
            partition_root / "shard_partition_map.parquet", index=False
        )
    return token_to_shard


def export_partition_assets(
    partition_files: list[Path],
    resolved_shard_path: Path,
    tar_path: Path | None,
    image_output_dir: Path | None,
    image_format: str,
    jpeg_quality: int,
    default_patch_size: int,
    compression: str,
    max_images_per_tar: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    if tar_path is None and image_output_dir is None:
        raise ValueError("At least one export target must be configured")
    if max_images_per_tar is not None and max_images_per_tar <= 0:
        raise ValueError(
            f"max_images_per_tar must be >= 1, got {max_images_per_tar}"
        )

    frame = (
        ds.dataset([str(path) for path in partition_files], format="parquet")
        .to_table()
        .to_pandas()
    )
    if frame.empty:
        return {
            "selected_rows": 0,
            "written_members": 0,
            "written_files": 0,
            "tar_count": 0,
            "tar_paths": [],
        }

    frame = frame.sort_values(
        ["source_index", "patch_index", "patch_y", "patch_x"]
    ).reset_index(drop=True)
    dataset = load_arrow_shard(resolved_shard_path)
    written_members = 0
    written_files = 0
    tar_paths: list[str] = []
    tar_mode = "w:gz" if compression == "gz" else "w"
    archive = None
    current_tar_members = 0
    current_tar_index = 0

    def advance_progress() -> None:
        if progress is not None:
            progress.update(1)

    def open_next_archive() -> None:
        nonlocal archive, current_tar_members, current_tar_index
        if tar_path is None:
            return
        if archive is not None:
            archive.close()
        current_tar_index += 1
        target_path = tar_path
        if max_images_per_tar is not None:
            target_path = build_chunked_tar_path(tar_path, current_tar_index)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        archive = tarfile.open(target_path, tar_mode)
        tar_paths.append(str(target_path))
        current_tar_members = 0

    try:
        current_source_index = None
        current_image = None
        scaled_image_cache: dict[tuple[int, int], Image.Image] = {}

        for row in frame.itertuples(index=False):
            if current_source_index != row.source_index:
                item = dataset[int(row.source_index)]
                bytes_data = item.get("jpg", {}).get("bytes")
                if not bytes_data:
                    advance_progress()
                    continue
                current_image = open_rgb_image(bytes_data)
                current_source_index = int(row.source_index)
                scaled_image_cache = {}

            if current_image is None:
                advance_progress()
                continue

            scaled_size = resolve_scaled_image_size(row, current_image)
            working_image = current_image
            if scaled_size != (current_image.width, current_image.height):
                working_image = scaled_image_cache.get(scaled_size)
                if working_image is None:
                    working_image = current_image.resize(
                        scaled_size, Image.Resampling.BILINEAR
                    )
                    scaled_image_cache[scaled_size] = working_image

            patch_size = resolve_patch_size(
                getattr(row, "patch_size", None), default_patch_size
            )
            left = int(row.patch_x)
            top = int(row.patch_y)
            right = min(left + patch_size, working_image.width)
            bottom = min(top + patch_size, working_image.height)
            if left >= right or top >= bottom:
                advance_progress()
                continue

            patch = working_image.crop((left, top, right, bottom))
            encoded = encode_patch_image(
                patch, image_format=image_format, jpeg_quality=jpeg_quality
            )
            row_dict = row._asdict()

            if archive is not None:
                if (
                    max_images_per_tar is not None
                    and current_tar_members >= max_images_per_tar
                ):
                    open_next_archive()
            elif tar_path is not None:
                open_next_archive()

            if archive is not None:
                member_name = build_patch_member_name(row_dict, image_format=image_format)
                info = tarfile.TarInfo(name=member_name)
                info.size = len(encoded)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(encoded))
                written_members += 1
                current_tar_members += 1

            if image_output_dir is not None:
                output_path = build_patch_output_path(
                    row_dict, output_dir=image_output_dir, image_format=image_format
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(encoded)
                written_files += 1
            advance_progress()
    finally:
        if archive is not None:
            archive.close()

    return {
        "selected_rows": int(len(frame)),
        "written_members": written_members,
        "written_files": written_files,
        "tar_count": len(tar_paths),
        "tar_paths": tar_paths,
    }


def export_shard_assets(task: dict[str, Any]) -> dict[str, Any]:
    shard_result = export_partition_assets(
        partition_files=list(task["partition_files"]),
        resolved_shard_path=Path(task["resolved_shard_path"]),
        tar_path=Path(task["tar_path"]) if task["tar_path"] else None,
        image_output_dir=(
            Path(task["image_output_dir"]) if task["image_output_dir"] else None
        ),
        image_format=str(task["image_format"]),
        jpeg_quality=int(task["jpeg_quality"]),
        default_patch_size=int(task["default_patch_size"]),
        compression=str(task["compression"]),
        max_images_per_tar=(
            int(task["max_images_per_tar"]) if task["max_images_per_tar"] else None
        ),
    )
    return {
        "order": int(task["order"]),
        "source_shard": str(task["source_shard"]),
        "resolved_source_shard": str(task["resolved_shard_path"]),
        "tar_path": str(task["tar_path"]) if task["tar_path"] else "",
        "image_output_dir": str(task["image_output_dir"]) if task["image_output_dir"] else "",
        **shard_result,
    }


def export_selected_patches_to_tars(
    final_selection_files: list[Path],
    output_dir: Path,
    data_dir: Path | None,
    image_format: str,
    jpeg_quality: int,
    default_patch_size: int,
    compression: str = "none",
    output_mode: str = "tar",
    image_output_dir: Path | None = None,
    num_workers: int = 1,
    max_images_per_tar: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    if not final_selection_files:
        raise FileNotFoundError(
            "No final selection parquet files provided for tar export"
        )
    if output_mode not in {"tar", "files", "both"}:
        raise ValueError(f"Unsupported output_mode={output_mode!r}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if max_images_per_tar is not None and max_images_per_tar <= 0:
        raise ValueError(
            f"max_images_per_tar must be >= 1, got {max_images_per_tar}"
        )

    partition_root = output_dir / "shard_partitions" / f"run_{uuid.uuid4().hex}"
    token_to_shard = partition_final_selection_by_shard(
        final_selection_files, partition_root
    )
    data_dir_lookup = build_data_dir_lookup(data_dir)
    if output_mode in {"files", "both"} and image_output_dir is None:
        image_output_dir = output_dir / "debug_images"
    total_selected_rows = 0
    if show_progress:
        total_selected_rows = int(
            ds.dataset([str(path) for path in final_selection_files], format="parquet")
            .count_rows()
        )

    tar_count = 0
    written_members = 0
    written_files = 0
    selected_rows = 0
    shard_summaries: list[dict[str, Any]] = []
    shard_tasks: list[dict[str, Any]] = []

    for order, (token, source_shard) in enumerate(sorted(token_to_shard.items())):
        partition_dir = partition_root / token
        partition_files = sorted(partition_dir.glob("*.parquet"))
        if not partition_files:
            continue
        resolved_shard_path = resolve_shard_path(
            source_shard, data_dir=data_dir, lookup=data_dir_lookup
        )
        tar_path = None
        if output_mode in {"tar", "both"}:
            tar_path = output_dir / build_tar_name(source_shard, compression=compression)
        shard_tasks.append(
            {
                "order": order,
                "source_shard": source_shard,
                "resolved_shard_path": resolved_shard_path,
                "partition_files": partition_files,
                "tar_path": tar_path,
                "image_output_dir": image_output_dir,
                "image_format": image_format,
                "jpeg_quality": jpeg_quality,
                "default_patch_size": default_patch_size,
                "compression": compression,
                "max_images_per_tar": max_images_per_tar,
            }
        )

    logger.info(
        "Exporting selected patches across %d shard(s) with %d worker(s).",
        len(shard_tasks),
        min(num_workers, max(1, len(shard_tasks))),
    )

    progress = maybe_make_progress(
        total=total_selected_rows,
        desc="Exporting selected patches",
        unit="patch",
        enabled=show_progress,
    )

    def record_shard_result(shard_info: dict[str, Any]) -> None:
        nonlocal tar_count, written_members, written_files, selected_rows
        if (
            int(shard_info["written_members"]) <= 0
            and int(shard_info["written_files"]) <= 0
        ):
            return
        tar_count += int(shard_info["tar_count"])
        written_members += int(shard_info["written_members"])
        written_files += int(shard_info["written_files"])
        selected_rows += int(shard_info["selected_rows"])
        shard_summaries.append(
            {
                "source_shard": str(shard_info["source_shard"]),
                "resolved_source_shard": str(shard_info["resolved_source_shard"]),
                "tar_path": str(shard_info["tar_path"]),
                "tar_paths": list(shard_info["tar_paths"]),
                "tar_count": int(shard_info["tar_count"]),
                "image_output_dir": str(shard_info["image_output_dir"]),
                "selected_rows": int(shard_info["selected_rows"]),
                "written_members": int(shard_info["written_members"]),
                "written_files": int(shard_info["written_files"]),
                "order": int(shard_info["order"]),
            }
        )

    try:
        if num_workers == 1 or len(shard_tasks) <= 1:
            for task in shard_tasks:
                shard_result = export_partition_assets(
                    partition_files=task["partition_files"],
                    resolved_shard_path=task["resolved_shard_path"],
                    tar_path=task["tar_path"],
                    image_output_dir=task["image_output_dir"],
                    image_format=task["image_format"],
                    jpeg_quality=task["jpeg_quality"],
                    default_patch_size=task["default_patch_size"],
                    compression=task["compression"],
                    max_images_per_tar=task["max_images_per_tar"],
                    progress=progress,
                )
                record_shard_result(
                    {
                        "order": task["order"],
                        "source_shard": task["source_shard"],
                        "resolved_source_shard": str(task["resolved_shard_path"]),
                        "tar_path": str(task["tar_path"]) if task["tar_path"] else "",
                        "image_output_dir": (
                            str(task["image_output_dir"])
                            if task["image_output_dir"]
                            else ""
                        ),
                        **shard_result,
                    }
                )
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=min(num_workers, len(shard_tasks)),
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    futures = [
                        executor.submit(export_shard_assets, task) for task in shard_tasks
                    ]
                    for future in as_completed(futures):
                        shard_info = future.result()
                        if progress is not None:
                            progress.update(int(shard_info["selected_rows"]))
                        record_shard_result(shard_info)
            except OSError as exc:
                raise RuntimeError(
                    "Failed to start tar-export worker processes. Retry with "
                    "num_workers=1 or run outside a restricted environment that "
                    "blocks process spawning."
                ) from exc
    finally:
        if progress is not None:
            progress.close()

    shard_summaries = [
        {key: value for key, value in item.items() if key != "order"}
        for item in sorted(shard_summaries, key=lambda item: int(item["order"]))
    ]

    summary = {
        "final_selection_files": len(final_selection_files),
        "output_mode": output_mode,
        "tar_count": tar_count,
        "selected_rows": selected_rows,
        "written_members": written_members,
        "written_files": written_files,
        "compression": compression,
        "image_format": image_format,
        "jpeg_quality": jpeg_quality,
        "default_patch_size": default_patch_size,
        "image_output_dir": str(image_output_dir) if image_output_dir else "",
        "num_workers": num_workers,
        "max_images_per_tar": max_images_per_tar or 0,
    }
    write_json(
        output_dir / "tar_export_summary.json", {**summary, "shards": shard_summaries}
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export globally selected patches grouped by source Arrow shard."
    )
    parser.add_argument(
        "--final_selection_dir",
        default="patchselect/out/global_selection/final_selection",
        help="Directory containing final selection parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="patchselect/out/global_selection/final_selection_tars",
        help="Output directory for tar archives and exporter working files",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional directory containing .arrow shards when manifest source paths need remapping",
    )
    parser.add_argument(
        "--image_format",
        type=parse_image_format,
        default="jpg",
        help="Patch encoding format for tar members and loose image exports",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality used when --image_format is jpg/jpeg",
    )
    parser.add_argument(
        "--default_patch_size",
        type=int,
        default=256,
        help="Fallback crop size when patch_size is absent from a manifest row",
    )
    parser.add_argument(
        "--compression",
        default="none",
        choices=("none", "gz"),
        help="Tar compression mode. Plain .tar is the default and works well for WebDataset-style loaders.",
    )
    parser.add_argument(
        "--output_mode",
        default="tar",
        choices=("tar", "files", "both"),
        help="Whether to write tar archives, loose patch image files, or both",
    )
    parser.add_argument(
        "--image_output_dir",
        default=None,
        help="Optional output directory for loose patch image files; defaults to <output_dir>/debug_images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for shard-level parallel export; each worker writes different source-shard outputs",
    )
    parser.add_argument(
        "--max_images_per_tar",
        type=int,
        default=None,
        help="Optional cap on image members per tar; large shards are split into numbered tar chunks",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars during export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_selection_dir = Path(args.final_selection_dir)
    final_selection_files = sorted(final_selection_dir.glob("*.parquet"))
    if not final_selection_files:
        raise FileNotFoundError(
            f"No final selection parquet files found in {final_selection_dir}"
        )
    output_dir = Path(args.output_dir)
    result = export_selected_patches_to_tars(
        final_selection_files=final_selection_files,
        output_dir=output_dir,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        default_patch_size=args.default_patch_size,
        compression=args.compression,
        output_mode=args.output_mode,
        image_output_dir=Path(args.image_output_dir) if args.image_output_dir else None,
        num_workers=args.num_workers,
        max_images_per_tar=args.max_images_per_tar,
        show_progress=not args.no_progress,
    )
    message = (
        "Patch export complete: "
        f"{result['selected_rows']} selected rows produced "
        f"{result['written_members']} tar member(s) across {result['tar_count']} tar file(s)"
    )
    if result["written_files"] > 0:
        message += f" and {result['written_files']} loose image file(s)"
    message += "."
    print(message)


if __name__ == "__main__":
    main()
