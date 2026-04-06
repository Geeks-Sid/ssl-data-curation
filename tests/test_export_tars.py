import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from patchselect.arrow_utils import discover_arrow_files
from patchselect.export_tars import (
    build_data_dir_lookup,
    export_selected_patches_to_tars,
    shard_token,
)


def make_final_selection_row(
    *,
    sample_suffix: str,
    source_shard: str,
    source_index: int,
    patch_index: int,
) -> dict:
    return {
        "sample_id": f"sample-{sample_suffix}",
        "sample_slug": f"sample_{sample_suffix}",
        "source_shard": source_shard,
        "source_index": source_index,
        "patch_index": patch_index,
        "patch_size": 8,
        "patch_x": 0,
        "patch_y": 0,
        "scale_level": 0,
        "scale_factor": 1.0,
        "grid_row": 0,
        "grid_col": 0,
        "gene": "GENE",
        "tissue": "synthetic",
        "cell_type": "Tumor cells",
        "diagnosis": "Synthetic",
        "scaled_image_width": 8,
        "scaled_image_height": 8,
        "is_cancer": 1.0,
        "selection_role": "prototype",
        "descriptor_backend": "cpu",
        "rle_available": 1,
        "rle_patch_foreground_frac": 1.0,
        "rle_slide_foreground_frac": 1.0,
        "objective_score": 1.0 - (patch_index / 100.0),
        "utility": 1.0 - (patch_index / 100.0),
        "quality_score": 0.5,
        "nuisance_score": 0.0,
        "semantic_coverage_score": 0.5,
        "interface_score": 0.1,
        "redundancy_penalty": 0.0,
        "rarity_score": 0.2,
        "state_bin": 0,
        "interface_bin": 0,
    }


class ExportTarsPathResolutionTest(unittest.TestCase):
    def test_discover_arrow_files_accepts_direct_file_and_nested_dirs(self) -> None:
        root = Path(__file__).resolve().parents[1] / "out" / "_test_export_tars_paths"
        shutil.rmtree(root, ignore_errors=True)
        nested = root / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        arrow_path = nested / "sample.arrow"
        arrow_path.write_bytes(b"arrow")
        try:
            self.assertEqual(discover_arrow_files(arrow_path), [arrow_path])
            self.assertEqual(discover_arrow_files(root), [arrow_path])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_build_data_dir_lookup_returns_empty_for_missing_dir(self) -> None:
        missing = Path(__file__).resolve().parents[1] / "out" / "_missing_arrow_dir"
        shutil.rmtree(missing, ignore_errors=True)
        lookup = build_data_dir_lookup(missing)
        self.assertEqual(lookup, {})

    def test_export_selected_patches_ignores_stale_partition_rows(self) -> None:
        root = Path(__file__).resolve().parents[1] / "out" / "_test_export_tars_stale"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            final_selection_dir = root / "final_selection"
            final_selection_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "tar_output"
            shard_path = root / "synthetic.arrow"
            shard_path.write_bytes(b"placeholder")

            current_rows = [
                make_final_selection_row(
                    sample_suffix="current-a",
                    source_shard=str(shard_path),
                    source_index=0,
                    patch_index=0,
                ),
                make_final_selection_row(
                    sample_suffix="current-b",
                    source_shard=str(shard_path),
                    source_index=0,
                    patch_index=1,
                ),
            ]
            current_manifest = final_selection_dir / "final_selection_part-000000.parquet"
            pd.DataFrame(current_rows).to_parquet(current_manifest, index=False)

            stale_partition_dir = output_dir / "shard_partitions" / shard_token(str(shard_path))
            stale_partition_dir.mkdir(parents=True, exist_ok=True)
            stale_manifest = stale_partition_dir / "part-999999.parquet"
            pd.DataFrame(
                [
                    make_final_selection_row(
                        sample_suffix="stale",
                        source_shard=str(shard_path),
                        source_index=0,
                        patch_index=99,
                    )
                ]
            ).to_parquet(stale_manifest, index=False)

            image = Image.new("RGB", (8, 8), color=(255, 0, 0))
            with patch("patchselect.export_tars.load_arrow_shard", return_value=[{"jpg": {"bytes": image.tobytes()}}]), patch(
                "patchselect.export_tars.open_rgb_image",
                return_value=image,
            ):
                result = export_selected_patches_to_tars(
                    final_selection_files=[current_manifest],
                    output_dir=output_dir,
                    data_dir=None,
                    image_format="png",
                    jpeg_quality=95,
                    default_patch_size=8,
                    compression="none",
                    output_mode="tar",
                    image_output_dir=None,
                    num_workers=1,
                    max_images_per_tar=None,
                    show_progress=False,
                )

            self.assertEqual(result["selected_rows"], 2)
            self.assertEqual(result["written_members"], 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
