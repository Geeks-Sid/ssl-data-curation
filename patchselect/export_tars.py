"""Export globally selected patches into tar archives grouped by source Arrow shard."""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from PIL import Image

from patchselect.arrow_utils import discover_arrow_files, load_arrow_shard, open_rgb_image, slugify
from patchselect.io_utils import write_dataframe_part, write_json


def shard_token(source_shard: str) -> str:
    path = Path(source_shard)
    stem = slugify(path.stem, "arrow_shard")
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def build_tar_name(source_shard: str, compression: str) -> str:
    suffix = ".tar.gz" if compression == "gz" else ".tar"
    return f"{shard_token(source_shard)}{suffix}"


def build_patch_member_name(row: dict, image_format: str) -> str:
    suffix = image_format.lower().lstrip(".")
    sample_slug = slugify(row.get("sample_slug") or row.get("sample_id"), "sample")
    return (
        f"{sample_slug}"
        f"__i{int(row['source_index']):07d}"
        f"__p{int(row['patch_index']):04d}"
        f"__x{int(row['patch_x']):05d}"
        f"__y{int(row['patch_y']):05d}.{suffix}"
    )


def resolve_shard_path(source_shard: str, data_dir: Path | None, lookup: dict[str, Path]) -> Path:
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
    return {path.name.lower(): path for path in discover_arrow_files(data_dir, split="all")}


def encode_patch_image(patch: Image.Image, image_format: str, jpeg_quality: int) -> bytes:
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


def partition_final_selection_by_shard(
    final_selection_files: list[Path],
    partition_root: Path,
) -> dict[str, str]:
    partition_root.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset([str(path) for path in final_selection_files], format="parquet")
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
        pd.DataFrame(mapping_rows).to_parquet(partition_root / "shard_partition_map.parquet", index=False)
    return token_to_shard


def export_partition_to_tar(
    partition_files: list[Path],
    resolved_shard_path: Path,
    tar_path: Path,
    image_format: str,
    jpeg_quality: int,
    default_patch_size: int,
    compression: str,
) -> dict[str, int]:
    frame = ds.dataset([str(path) for path in partition_files], format="parquet").to_table().to_pandas()
    if frame.empty:
        return {"selected_rows": 0, "written_members": 0}

    frame = frame.sort_values(["source_index", "patch_index", "patch_y", "patch_x"]).reset_index(drop=True)
    dataset = load_arrow_shard(resolved_shard_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_mode = "w:gz" if compression == "gz" else "w"
    written_members = 0

    with tarfile.open(tar_path, tar_mode) as archive:
        current_source_index = None
        current_image = None

        for row in frame.itertuples(index=False):
            if current_source_index != row.source_index:
                item = dataset[int(row.source_index)]
                bytes_data = item.get("jpg", {}).get("bytes")
                if not bytes_data:
                    continue
                current_image = open_rgb_image(bytes_data)
                current_source_index = int(row.source_index)

            if current_image is None:
                continue

            patch_size = resolve_patch_size(getattr(row, "patch_size", None), default_patch_size)
            left = int(row.patch_x)
            top = int(row.patch_y)
            right = min(left + patch_size, current_image.width)
            bottom = min(top + patch_size, current_image.height)
            if left >= right or top >= bottom:
                continue

            patch = current_image.crop((left, top, right, bottom))
            encoded = encode_patch_image(patch, image_format=image_format, jpeg_quality=jpeg_quality)
            member_name = build_patch_member_name(row._asdict(), image_format=image_format)
            info = tarfile.TarInfo(name=member_name)
            info.size = len(encoded)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(encoded))
            written_members += 1

    return {"selected_rows": int(len(frame)), "written_members": written_members}


def export_selected_patches_to_tars(
    final_selection_files: list[Path],
    output_dir: Path,
    data_dir: Path | None,
    image_format: str,
    jpeg_quality: int,
    default_patch_size: int,
    compression: str = "none",
) -> dict[str, int]:
    if not final_selection_files:
        raise FileNotFoundError("No final selection parquet files provided for tar export")

    partition_root = output_dir / "shard_partitions"
    token_to_shard = partition_final_selection_by_shard(final_selection_files, partition_root)
    data_dir_lookup = build_data_dir_lookup(data_dir)

    tar_count = 0
    written_members = 0
    selected_rows = 0
    shard_summaries: list[dict[str, str | int]] = []

    for token, source_shard in sorted(token_to_shard.items()):
        partition_dir = partition_root / token
        partition_files = sorted(partition_dir.glob("*.parquet"))
        if not partition_files:
            continue
        resolved_shard_path = resolve_shard_path(source_shard, data_dir=data_dir, lookup=data_dir_lookup)
        tar_path = output_dir / build_tar_name(source_shard, compression=compression)
        shard_result = export_partition_to_tar(
            partition_files=partition_files,
            resolved_shard_path=resolved_shard_path,
            tar_path=tar_path,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            default_patch_size=default_patch_size,
            compression=compression,
        )
        if shard_result["written_members"] <= 0:
            continue
        tar_count += 1
        written_members += shard_result["written_members"]
        selected_rows += shard_result["selected_rows"]
        shard_summaries.append(
            {
                "source_shard": source_shard,
                "resolved_source_shard": str(resolved_shard_path),
                "tar_path": str(tar_path),
                "selected_rows": shard_result["selected_rows"],
                "written_members": shard_result["written_members"],
            }
        )

    summary = {
        "final_selection_files": len(final_selection_files),
        "tar_count": tar_count,
        "selected_rows": selected_rows,
        "written_members": written_members,
        "compression": compression,
        "image_format": image_format,
        "jpeg_quality": jpeg_quality,
        "default_patch_size": default_patch_size,
    }
    write_json(output_dir / "tar_export_summary.json", {**summary, "shards": shard_summaries})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack globally selected patches into tar archives grouped by source Arrow shard."
    )
    parser.add_argument(
        "--final_selection_dir",
        default="patchselect/out/global_selection/final_selection",
        help="Directory containing final selection parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="patchselect/out/global_selection/final_selection_tars",
        help="Output directory for tar archives",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional directory containing .arrow shards when manifest source paths need remapping",
    )
    parser.add_argument(
        "--image_format",
        default="jpg",
        choices=("jpg", "jpeg", "png"),
        help="Patch encoding format inside each tar archive",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_selection_dir = Path(args.final_selection_dir)
    final_selection_files = sorted(final_selection_dir.glob("*.parquet"))
    if not final_selection_files:
        raise FileNotFoundError(f"No final selection parquet files found in {final_selection_dir}")
    output_dir = Path(args.output_dir)
    result = export_selected_patches_to_tars(
        final_selection_files=final_selection_files,
        output_dir=output_dir,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        default_patch_size=args.default_patch_size,
        compression=args.compression,
    )
    print(
        f"Tar export complete: {result['written_members']} patches written across {result['tar_count']} tar files."
    )


if __name__ == "__main__":
    main()
