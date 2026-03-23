"""Run global balancing over locally selected patch candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from patchselect.config import GlobalSelectionConfig
from patchselect.io_utils import write_json
from patchselect.selection import run_global_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global balancing over local patch candidates.")
    parser.add_argument(
        "--candidate_dir",
        default="patchselect/out/local_selection/candidates",
        help="Directory containing local candidate parquet files",
    )
    parser.add_argument("--output_dir", default="patchselect/out/global_selection", help="Output directory")
    parser.add_argument("--target_size", type=int, required=True, help="Target number of final selected patches")
    parser.add_argument(
        "--bin_columns",
        default="tissue,cell_type,state_bin",
        help="Comma-separated column list used for balancing",
    )
    parser.add_argument("--bin_alpha", type=float, default=0.5, help="Tempering coefficient for bin quotas")
    parser.add_argument("--min_quota", type=int, default=0, help="Optional minimum quota per non-empty bin")
    parser.add_argument("--utility_column", default="utility", help="Column used for within-bin ranking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    candidate_files = sorted(candidate_dir.glob("*.parquet"))
    if not candidate_files:
        raise FileNotFoundError(f"No parquet candidate files found in {candidate_dir}")

    config = GlobalSelectionConfig(
        target_size=args.target_size,
        bin_columns=tuple(column.strip() for column in args.bin_columns.split(",") if column.strip()),
        bin_alpha=args.bin_alpha,
        utility_column=args.utility_column,
        per_bin_min_quota=args.min_quota,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_global_selection(candidate_files, output_dir, config)
    summary = {
        "config": vars(args),
        **result,
    }
    write_json(output_dir / "run_summary.json", summary)
    print(
        f"Global selection complete: {result['selected_rows']} rows selected across {result['bin_count']} bins."
    )


if __name__ == "__main__":
    main()
