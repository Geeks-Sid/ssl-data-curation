"""I/O helpers for parquet outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_dataframe_part(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_rows_part(rows: list[dict], output_dir: Path, prefix: str, part_index: int) -> int:
    if not rows:
        return part_index
    frame = pd.DataFrame(rows)
    out_path = output_dir / f"{prefix}_part-{part_index:06d}.parquet"
    write_dataframe_part(frame, out_path)
    return part_index + 1


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
