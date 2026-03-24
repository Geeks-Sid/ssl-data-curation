"""Export full images from Arrow shards for inspection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from patchselect.arrow_utils import (
    build_output_name,
    common_export_args,
    discover_arrow_files,
    load_arrow_shard,
    resolve_collision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export images from Arrow shards using filenames of the form "
            "'marker_tissue_celltype_index.ext'."
        )
    )
    common_export_args(parser)
    parser.add_argument(
        "--output_dir",
        default="out/exported_images",
        help="Directory where images are written",
    )
    parser.add_argument(
        "--split",
        default="eval",
        choices=("all", "train", "valid", "test", "eval"),
        help="Which shard split to export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_arrow_files(data_dir, split=args.split)
    collision_counts: Counter[str] = Counter()
    exported = 0

    for shard in files:
        dataset = load_arrow_shard(shard)
        for index, item in enumerate(dataset, start=1):
            bytes_data = item.get("jpg", {}).get("bytes")
            if not bytes_data:
                continue
            filename = build_output_name(item, index)
            output_path = resolve_collision(output_dir / filename, collision_counts)
            output_path.write_bytes(bytes_data)
            exported += 1
            if args.limit is not None and exported >= args.limit:
                break
        if args.limit is not None and exported >= args.limit:
            break

    print("Arrow files:")
    for path in files:
        print(f"  {path}")
    print(f"Exported {exported} images to {output_dir}")


if __name__ == "__main__":
    main()
