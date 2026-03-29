"""Export embedded patch images from a parquet file to PNG or JPEG files."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export patch images stored in a parquet file."
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Parquet file containing an image_bytes column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Dummy/exported_patches"),
        help="Directory where patch images will be written.",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        help="Output image format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of patches to export.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when --format=jpg.",
    )
    return parser.parse_args()


def build_name(task_id: object, index: int, suffix: str) -> str:
    task = str(task_id) if task_id is not None else "patches"
    return f"{task}_patch_{index:05d}.{suffix}"


def export_patches(
    parquet_path: Path,
    output_dir: Path,
    image_format: str,
    limit: int | None,
    jpeg_quality: int,
) -> int:
    frame = pd.read_parquet(parquet_path, columns=["task_id", "image_bytes"])
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        if limit is not None and exported >= limit:
            break
        image_bytes = row.image_bytes
        if not image_bytes:
            continue

        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            out_path = output_dir / build_name(row.task_id, index, image_format)
            if image_format == "jpg":
                rgb.save(out_path, format="JPEG", quality=jpeg_quality)
            else:
                rgb.save(out_path, format="PNG")
        exported += 1
    return exported


def main() -> None:
    args = parse_args()
    exported = export_patches(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        image_format=args.format,
        limit=args.limit,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Exported {exported} patches to {args.output_dir}")


if __name__ == "__main__":
    main()
