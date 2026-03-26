import unittest
from pathlib import Path
import shutil

import pandas as pd

from patchselect.config import GlobalSelectionConfig
from patchselect.selection import run_global_selection


def make_candidate_row(
    *,
    sample_suffix: str,
    source_index: int,
    patch_index: int,
    tissue: str,
    is_cancer: float | None,
    state_bin: int,
    interface_bin: int,
    objective_score: float,
) -> dict:
    return {
        "sample_id": f"sample-{sample_suffix}",
        "sample_slug": f"sample_{sample_suffix}",
        "source_shard": "dummy.arrow",
        "source_index": source_index,
        "patch_index": patch_index,
        "patch_size": 256,
        "patch_x": patch_index * 16,
        "patch_y": patch_index * 16,
        "grid_row": patch_index,
        "grid_col": patch_index,
        "gene": "GENE",
        "tissue": tissue,
        "cell_type": "Tumor cells",
        "diagnosis": "Synthetic",
        "is_cancer": is_cancer,
        "selection_role": "prototype",
        "descriptor_backend": "cpu",
        "rle_available": 1,
        "rle_patch_foreground_frac": 0.9,
        "rle_slide_foreground_frac": 0.8,
        "objective_score": objective_score,
        "utility": objective_score,
        "quality_score": 0.1,
        "nuisance_score": -0.1,
        "semantic_coverage_score": 0.2,
        "interface_score": 0.3,
        "redundancy_penalty": -0.2,
        "rarity_score": 0.4,
        "state_bin": state_bin,
        "interface_bin": interface_bin,
    }


class GlobalSelectionTest(unittest.TestCase):
    def test_global_selection_keeps_nan_bin_rows(self) -> None:
        frame = pd.DataFrame(
            [
                make_candidate_row(
                    sample_suffix="a",
                    source_index=0,
                    patch_index=0,
                    tissue="bone marrow",
                    is_cancer=None,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.9,
                ),
                make_candidate_row(
                    sample_suffix="b",
                    source_index=0,
                    patch_index=1,
                    tissue="bone marrow",
                    is_cancer=None,
                    state_bin=1,
                    interface_bin=0,
                    objective_score=0.8,
                ),
                make_candidate_row(
                    sample_suffix="c",
                    source_index=1,
                    patch_index=0,
                    tissue="prostate cancer",
                    is_cancer=1.0,
                    state_bin=0,
                    interface_bin=1,
                    objective_score=0.7,
                ),
                make_candidate_row(
                    sample_suffix="d",
                    source_index=1,
                    patch_index=1,
                    tissue="prostate cancer",
                    is_cancer=1.0,
                    state_bin=1,
                    interface_bin=1,
                    objective_score=0.6,
                ),
            ]
        )

        root = Path(__file__).resolve().parents[1] / "out" / "_test_global_selection"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            candidate_path = root / "candidates.parquet"
            output_dir = root / "out"
            frame.to_parquet(candidate_path, index=False)

            result = run_global_selection(
                [candidate_path],
                output_dir,
                GlobalSelectionConfig(target_size=4),
            )

            self.assertEqual(result["selected_rows"], 4)
            final_files = sorted((output_dir / "final_selection").glob("*.parquet"))
            self.assertEqual(len(final_files), 1)

            selected = pd.concat(
                [pd.read_parquet(path) for path in final_files], ignore_index=True
            )
            self.assertEqual(len(selected), 4)
            self.assertIn("sample_slug", selected.columns)
            self.assertTrue(selected["is_cancer"].isna().any())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
