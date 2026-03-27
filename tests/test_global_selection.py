import unittest
from pathlib import Path
import shutil

import pandas as pd

from patchselect.config import GlobalSelectionConfig
from patchselect.selection import cudf, resolve_dataframe_backend, run_global_selection


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
    def test_resolve_dataframe_backend_defaults_to_available_backend(self) -> None:
        expected = "cudf" if cudf is not None else "pandas"
        self.assertEqual(resolve_dataframe_backend("auto"), expected)

    def test_requesting_cudf_without_dependency_raises(self) -> None:
        if cudf is not None:
            self.skipTest("cudf is installed in this environment")
        with self.assertRaises(ImportError):
            resolve_dataframe_backend("cudf")

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

    def test_global_selection_supports_explicit_pandas_backend(self) -> None:
        frame = pd.DataFrame(
            [
                make_candidate_row(
                    sample_suffix="a",
                    source_index=0,
                    patch_index=0,
                    tissue="bone marrow",
                    is_cancer=0.0,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.9,
                ),
                make_candidate_row(
                    sample_suffix="b",
                    source_index=0,
                    patch_index=1,
                    tissue="bone marrow",
                    is_cancer=0.0,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.8,
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
                GlobalSelectionConfig(target_size=1, dataframe_backend="pandas"),
            )

            self.assertEqual(result["selected_rows"], 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_global_selection_keeps_global_topk_across_files(self) -> None:
        frame_a = pd.DataFrame(
            [
                make_candidate_row(
                    sample_suffix="a",
                    source_index=0,
                    patch_index=0,
                    tissue="bone marrow",
                    is_cancer=0.0,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.10,
                ),
                make_candidate_row(
                    sample_suffix="b",
                    source_index=0,
                    patch_index=1,
                    tissue="bone marrow",
                    is_cancer=0.0,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.90,
                ),
            ]
        )
        frame_b = pd.DataFrame(
            [
                make_candidate_row(
                    sample_suffix="c",
                    source_index=1,
                    patch_index=0,
                    tissue="bone marrow",
                    is_cancer=0.0,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.80,
                ),
                make_candidate_row(
                    sample_suffix="d",
                    source_index=1,
                    patch_index=1,
                    tissue="bone marrow",
                    is_cancer=0.0,
                    state_bin=0,
                    interface_bin=0,
                    objective_score=0.20,
                ),
            ]
        )

        root = Path(__file__).resolve().parents[1] / "out" / "_test_global_selection"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            candidate_path_a = root / "candidates_a.parquet"
            candidate_path_b = root / "candidates_b.parquet"
            output_dir = root / "out"
            frame_a.to_parquet(candidate_path_a, index=False)
            frame_b.to_parquet(candidate_path_b, index=False)

            result = run_global_selection(
                [candidate_path_a, candidate_path_b],
                output_dir,
                GlobalSelectionConfig(target_size=2, bin_alpha=1.0),
            )

            self.assertEqual(result["selected_rows"], 2)
            final_files = sorted((output_dir / "final_selection").glob("*.parquet"))
            self.assertEqual(len(final_files), 1)

            selected = pd.concat(
                [pd.read_parquet(path) for path in final_files], ignore_index=True
            )
            self.assertEqual(len(selected), 2)
            self.assertEqual(
                sorted(selected["objective_score"].tolist(), reverse=True),
                [0.9, 0.8],
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
