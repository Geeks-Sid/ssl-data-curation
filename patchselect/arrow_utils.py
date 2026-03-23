"""Arrow dataset helpers for patch selection."""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image


def discover_arrow_files(data_dir: Path, split: str = "all") -> list[Path]:
    files = sorted(data_dir.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No .arrow files found in {data_dir}")

    split = split.lower()
    if split == "all":
        return files

    keyword_map = {
        "train": ("train",),
        "valid": ("valid", "validation"),
        "test": ("test",),
        "eval": ("valid", "validation", "test"),
    }
    keywords = keyword_map.get(split)
    if keywords is None:
        raise ValueError(f"Unsupported split={split!r}")

    selected = [path for path in files if any(keyword in path.name.lower() for keyword in keywords)]
    if not selected:
        raise FileNotFoundError(
            f"No .arrow files matching split={split!r} found in {data_dir}"
        )
    return selected


def load_arrow_shard(path: Path):
    dataset = load_dataset(
        "arrow",
        data_files=[str(path)],
        split="train",
        keep_in_memory=False,
    )
    return dataset.cast_column("jpg", HFImage(decode=False))


def iter_arrow_records(paths: Iterable[Path]):
    for path in paths:
        dataset = load_arrow_shard(path)
        for index, item in enumerate(dataset):
            yield path, index, item


def slugify(value: object, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def decode_json_like(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_custom_metadata(item: dict) -> dict:
    payload = decode_json_like(item.get("json", {}))
    metadata = payload.get("custom_metadata", payload)
    metadata = decode_json_like(metadata)
    return metadata if isinstance(metadata, dict) else {}


def infer_malignancy(metadata: dict) -> int | None:
    for key in ("is_cancer", "malignancy", "malignancy_status", "cancer"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(bool(value))
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"cancer", "malignant", "tumor", "yes", "true", "1"}:
                return 1
            if lowered in {"normal", "benign", "no", "false", "0"}:
                return 0

    text_fields = " ".join(
        str(metadata.get(key, "")).lower()
        for key in ("tissue", "diagnosis", "snomed_text", "title")
    )
    cancer_terms = (
        "cancer",
        "carcinoma",
        "adenocarcinoma",
        "sarcoma",
        "lymphoma",
        "melanoma",
        "glioma",
        "tumor",
        "malignan",
    )
    if any(term in text_fields for term in cancer_terms):
        return 1
    return None


def normalize_metadata(metadata: dict) -> dict:
    return {
        "gene": str(metadata.get("gene") or metadata.get("marker") or metadata.get("antibody") or ""),
        "tissue": str(metadata.get("tissue") or metadata.get("organ") or metadata.get("tissue_name") or ""),
        "cell_type": str(
            metadata.get("cell_type")
            or metadata.get("celltype")
            or metadata.get("cell")
            or metadata.get("cell line")
            or ""
        ),
        "diagnosis": str(
            metadata.get("snomed_text")
            or metadata.get("diagnosis")
            or metadata.get("description")
            or ""
        ),
        "snomed_code": str(metadata.get("snomed_code") or metadata.get("snomed") or ""),
        "md5": str(metadata.get("md5") or metadata.get("image_md5") or ""),
        "is_cancer": infer_malignancy(metadata),
    }


def open_rgb_image(bytes_data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(bytes_data))
    return image.convert("RGB")


def build_output_name(item: dict, index: int) -> str:
    metadata = normalize_metadata(extract_custom_metadata(item))
    jpg_path = item.get("jpg", {}).get("path", "")
    suffix = Path(jpg_path).suffix.lower() or ".jpg"
    marker = slugify(metadata.get("gene"), "unknown_marker")
    tissue = slugify(metadata.get("tissue"), "unknown_tissue")
    cell_type = slugify(metadata.get("cell_type"), "unknown_celltype")
    return f"{marker}_{tissue}_{cell_type}_{index}{suffix}"


def resolve_collision(path: Path, counts: Counter[str]) -> Path:
    if not path.exists() and counts[path.name] == 0:
        counts[path.name] += 1
        return path

    stem = path.stem
    suffix = path.suffix
    while True:
        counts[path.name] += 1
        candidate = path.with_name(f"{stem}_{counts[path.name]}{suffix}")
        if not candidate.exists():
            return candidate


def common_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data_dir", default="Data", help="Directory containing .arrow shards")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to process")
    return parser
