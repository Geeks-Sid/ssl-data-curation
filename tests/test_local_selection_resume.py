import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd

from patchselect import run_local_selection


class DummyTqdm:
    def __init__(self, *args, **kwargs) -> None:
        self.updated = 0

    def update(self, value: int) -> None:
        self.updated += value

    def set_postfix(self, *_args, **_kwargs) -> None:
        return

    def close(self) -> None:
        return


class LocalSelectionResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = (
            Path(__file__).resolve().parents[1]
            / "out"
            / f"_test_local_selection_resume_{self._testMethodName}_{uuid.uuid4().hex}"
        )
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_local_selection_resumes_when_signature_matches(self) -> None:
        output_dir = self.temp_dir / "out"
        shard_path = self.temp_dir / "train-000.arrow"
        shard_path.write_bytes(b"placeholder")
        dataset = [
            {"jpg": {"bytes": b"img-0"}, "meta": {"md5": "sample-0"}},
            {"jpg": {"bytes": b"img-1"}, "meta": {"md5": "sample-1"}},
            {"jpg": {"bytes": b"img-2"}, "meta": {"md5": "sample-2"}},
        ]
        processed_task_ids: list[list[str]] = []
        worker_counts: list[int] = []
        call_count = {"value": 0}

        def fake_process_chunk(tasks, worker_count, allow_gpu_reduction):
            processed_task_ids.append([task["sample_id"] for task in tasks])
            worker_counts.append(worker_count)
            call_count["value"] += 1
            if call_count["value"] == 2:
                raise RuntimeError("simulated crash")
            results = []
            for task in tasks:
                results.append(
                    {
                        "status": "ok",
                        "sample_id": task["sample_id"],
                        "source_index": task["source_index"],
                        "rows": [
                            {
                                "sample_id": task["sample_id"],
                                "source_index": task["source_index"],
                                "patch_index": 0,
                            }
                        ],
                    }
                )
            return results, worker_count, 0

        patches = [
            mock.patch.object(
                run_local_selection, "discover_arrow_files", return_value=[shard_path]
            ),
            mock.patch.object(
                run_local_selection, "load_arrow_shard", return_value=dataset
            ),
            mock.patch.object(
                run_local_selection,
                "extract_custom_metadata",
                side_effect=lambda item: item["meta"],
            ),
            mock.patch.object(
                run_local_selection, "process_chunk_with_retries", side_effect=fake_process_chunk
            ),
            mock.patch.object(run_local_selection, "tqdm", DummyTqdm),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with mock.patch(
                "sys.argv",
                [
                    "main.py",
                    "--data_dir",
                    str(self.temp_dir),
                    "--output_dir",
                    str(output_dir),
                    "--chunk_size",
                    "1",
                    "--num_workers",
                    "1",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    run_local_selection.main()

            with mock.patch(
                "sys.argv",
                [
                    "main.py",
                    "--data_dir",
                    str(self.temp_dir),
                    "--output_dir",
                    str(output_dir),
                    "--chunk_size",
                    "1",
                    "--num_workers",
                    "3",
                ],
            ):
                run_local_selection.main()

        self.assertEqual(processed_task_ids, [["sample-0"], ["sample-1"], ["sample-1"], ["sample-2"]])
        self.assertEqual(worker_counts, [1, 1, 3, 3])

        summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["processed_images"], 3)
        self.assertEqual(summary["selected_patches"], 3)

        state = json.loads(
            (output_dir / run_local_selection.RESUME_STATE_PATH).read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(state["completed"])
        self.assertEqual(state["processed_images"], 3)
        self.assertEqual(state["final_worker_count"], 3)

        parts = sorted((output_dir / "candidates").glob("*.parquet"))
        self.assertEqual(len(parts), 3)
        selected = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
        self.assertEqual(selected["sample_id"].tolist(), ["sample-0", "sample-1", "sample-2"])

    def test_local_selection_rejects_resume_when_signature_changes(self) -> None:
        output_dir = self.temp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = self.temp_dir / "train-000.arrow"
        shard_path.write_bytes(b"placeholder")

        existing_state = {
            "completed": False,
            "signature": {
                "config": {
                    "patch_size": 256,
                    "patch_stride": 256,
                    "descriptor_backend": "cpu",
                    "magnification_factors": [1.0, 0.5, 0.25],
                    "downsample_size": None,
                    "slide_stats_size": None,
                    "use_rle_mask": True,
                    "rle_min_fraction": 0.75,
                    "tissue_min_fraction": 0.05,
                    "od_tissue_threshold": 0.12,
                    "sat_tissue_threshold": 0.08,
                    "value_tissue_threshold": 0.95,
                    "nuclei_h_threshold": 0.35,
                    "dab_positive_threshold": 0.30,
                    "local_keep_ratio": 0.10,
                    "local_keep_min": 1,
                    "local_keep_max": 4,
                    "min_component_size": 16,
                    "border_width": 4,
                    "image_format": "jpg",
                    "jpeg_quality": 95,
                    "semantic_weight": 0.50,
                    "interface_weight": 0.30,
                    "quality_weight": 0.15,
                    "redundancy_weight": 0.20,
                    "nuisance_weight": 0.35,
                    "state_gain_bonus": 0.35,
                    "interface_gain_bonus": 0.20,
                },
                "limit_images": None,
                "save_selected_patches": False,
                "selected_patch_dir": str((output_dir / "selected_patches").resolve()),
                "source_files": [str(shard_path.resolve())],
            },
            "processed_images": 1,
            "skipped_images": 0,
            "selected_patches": 1,
            "candidate_parts": 1,
            "final_worker_count": 1,
            "gpu_worker_reductions": 0,
            "task_counter": 1,
            "cursor": {"shard_index": 0, "next_source_index": 1},
        }
        (output_dir / run_local_selection.RESUME_STATE_PATH).write_text(
            json.dumps(existing_state), encoding="utf-8"
        )

        with mock.patch.object(
            run_local_selection, "discover_arrow_files", return_value=[shard_path]
        ), mock.patch("sys.argv", ["main.py", "--output_dir", str(output_dir), "--patch_size", "128"]):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                run_local_selection.main()

    def test_short_chunk_does_not_reduce_future_worker_count(self) -> None:
        output_dir = self.temp_dir / "out"
        shard_a = self.temp_dir / "train-000.arrow"
        shard_b = self.temp_dir / "train-001.arrow"
        shard_a.write_bytes(b"placeholder-a")
        shard_b.write_bytes(b"placeholder-b")
        datasets = {
            shard_a: [
                {"jpg": {"bytes": b"img-a0"}, "meta": {"md5": "sample-a0"}},
                {"jpg": {"bytes": b"img-a1"}, "meta": {"md5": "sample-a1"}},
                {"jpg": {"bytes": b"img-a2"}, "meta": {"md5": "sample-a2"}},
            ],
            shard_b: [
                {"jpg": {"bytes": b"img-b0"}, "meta": {"md5": "sample-b0"}},
                {"jpg": {"bytes": b"img-b1"}, "meta": {"md5": "sample-b1"}},
                {"jpg": {"bytes": b"img-b2"}, "meta": {"md5": "sample-b2"}},
                {"jpg": {"bytes": b"img-b3"}, "meta": {"md5": "sample-b3"}},
            ],
        }
        worker_counts: list[int] = []
        task_batches: list[list[str]] = []

        def fake_process_chunk(tasks, worker_count, allow_gpu_reduction):
            task_batches.append([task["sample_id"] for task in tasks])
            worker_counts.append(worker_count)
            results = []
            for task in tasks:
                results.append(
                    {
                        "status": "ok",
                        "sample_id": task["sample_id"],
                        "source_index": task["source_index"],
                        "rows": [
                            {
                                "sample_id": task["sample_id"],
                                "source_index": task["source_index"],
                                "patch_index": 0,
                            }
                        ],
                    }
                )
            return results, worker_count, 0

        with mock.patch.object(
            run_local_selection,
            "discover_arrow_files",
            return_value=[shard_a, shard_b],
        ), mock.patch.object(
            run_local_selection,
            "load_arrow_shard",
            side_effect=lambda path: datasets[path],
        ), mock.patch.object(
            run_local_selection,
            "extract_custom_metadata",
            side_effect=lambda item: item["meta"],
        ), mock.patch.object(
            run_local_selection,
            "process_chunk_with_retries",
            side_effect=fake_process_chunk,
        ), mock.patch.object(run_local_selection, "tqdm", DummyTqdm), mock.patch(
            "sys.argv",
            [
                "main.py",
                "--data_dir",
                str(self.temp_dir),
                "--output_dir",
                str(output_dir),
                "--chunk_size",
                "2",
                "--num_workers",
                "4",
            ],
        ):
            run_local_selection.main()

        self.assertEqual(
            task_batches,
            [
                ["sample-a0", "sample-a1"],
                ["sample-a2"],
                ["sample-b0", "sample-b1"],
                ["sample-b2", "sample-b3"],
            ],
        )
        self.assertEqual(worker_counts, [4, 4, 4, 4])

        state = json.loads(
            (output_dir / run_local_selection.RESUME_STATE_PATH).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(state["final_worker_count"], 4)


if __name__ == "__main__":
    unittest.main()
