import io
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

from patchselect import export_tars
from patchselect.config import PatchSelectionConfig
from patchselect.constants import BASE_FEATURE_NAMES, NEIGHBOR_FEATURE_NAMES
from patchselect.pipeline import select_patches_from_image
from patchselect.selection import add_neighborhood_features


def make_test_image(size: int = 4) -> Image.Image:
    values = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    return Image.fromarray(values, mode="RGB")


def make_image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fake_descriptors(
    patches: list[np.ndarray], _slide_stats: object, _cfg: PatchSelectionConfig, **_kwargs
) -> list[np.ndarray]:
    descriptors: list[np.ndarray] = []
    for index, patch in enumerate(patches):
        descriptor = np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float32)
        descriptor[0] = 1.0
        descriptor[4] = float(patch.mean()) / 255.0
        descriptor[9] = float(index + 1) / 10.0
        descriptor[12] = float(index + 1) / 10.0
        descriptor[20 + (index % 4)] = 1.0
        descriptor[24] = float(index + 1) / 10.0
        descriptor[25] = 0.2 + 0.05 * index
        descriptor[29 + (index % 3)] = 0.1 + 0.05 * index
        descriptor[34] = float(index + 1)
        descriptor[36] = float(index + 1)
        descriptor[41] = 0.1 * index
        descriptors.append(descriptor)
    return descriptors


class LocalSelectionMultiscaleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = (
            Path(__file__).resolve().parents[1]
            / "out"
            / f"_test_local_selection_multiscale_{uuid.uuid4().hex}"
        )
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiscale_local_selection_pools_candidates_across_scales(self) -> None:
        image = make_test_image()
        cfg = PatchSelectionConfig(
            patch_size=2,
            patch_stride=2,
            magnification_factors=(1.0, 0.5),
            use_rle_mask=False,
            local_keep_ratio=1.0,
            local_keep_min=1,
            local_keep_max=10,
        )

        with mock.patch(
            "patchselect.pipeline.compute_slide_stats_for_backend",
            return_value=object(),
        ), mock.patch(
            "patchselect.pipeline.compute_patch_descriptors_for_backend",
            side_effect=fake_descriptors,
        ):
            rows, patch_records = select_patches_from_image(
                image=image,
                sample_id="sample-1",
                metadata={},
                cfg=cfg,
                source_shard="dummy.arrow",
                source_index=0,
            )

        self.assertEqual(len(rows), 5)
        self.assertEqual(len(patch_records), 5)
        self.assertEqual(sorted(row["scale_level"] for row in rows), [0, 0, 0, 0, 1])
        self.assertEqual(
            sorted((row["scaled_image_width"], row["scaled_image_height"]) for row in rows),
            [(2, 2), (4, 4), (4, 4), (4, 4), (4, 4)],
        )
        self.assertTrue(all(row["image_width"] == 4 for row in rows))
        self.assertTrue(all(row["image_height"] == 4 for row in rows))

    def test_multiscale_local_selection_respects_max_patches_per_image(self) -> None:
        image = make_test_image()
        cfg = PatchSelectionConfig(
            patch_size=2,
            patch_stride=2,
            magnification_factors=(1.0, 0.5),
            use_rle_mask=False,
            local_keep_ratio=1.0,
            local_keep_min=1,
            local_keep_max=3,
        )

        with mock.patch(
            "patchselect.pipeline.compute_slide_stats_for_backend",
            return_value=object(),
        ), mock.patch(
            "patchselect.pipeline.compute_patch_descriptors_for_backend",
            side_effect=fake_descriptors,
        ):
            rows, _patch_records = select_patches_from_image(
                image=image,
                sample_id="sample-1",
                metadata={},
                cfg=cfg,
                source_shard="dummy.arrow",
                source_index=0,
            )

        self.assertEqual(len(rows), 3)

    def test_neighborhood_features_do_not_cross_scales(self) -> None:
        base_descriptor = np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float32)
        records = [
            {
                "scale_level": 0,
                "grid_row": 0,
                "grid_col": 0,
                "descriptor_base": base_descriptor.copy(),
            },
            {
                "scale_level": 1,
                "grid_row": 0,
                "grid_col": 0,
                "descriptor_base": base_descriptor.copy(),
            },
        ]

        add_neighborhood_features(records)

        for record in records:
            np.testing.assert_allclose(
                record["descriptor"][-len(NEIGHBOR_FEATURE_NAMES) :], 0.0
            )

    def test_export_partition_assets_reconstructs_scaled_crop(self) -> None:
        image = make_test_image()
        image_bytes = make_image_bytes(image)
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "sample-1",
                    "sample_slug": "sample_1",
                    "source_shard": "dummy.arrow",
                    "source_index": 0,
                    "patch_index": 0,
                    "patch_size": 1,
                    "patch_x": 1,
                    "patch_y": 0,
                    "scale_level": 1,
                    "scale_factor": 0.5,
                    "scaled_image_width": 2,
                    "scaled_image_height": 2,
                }
            ]
        )
        partition_path = self.temp_dir / "partition.parquet"
        output_dir = self.temp_dir / "images"
        frame.to_parquet(partition_path, index=False)

        with mock.patch.object(
            export_tars, "load_arrow_shard", return_value=[{"jpg": {"bytes": image_bytes}}]
        ):
            result = export_tars.export_partition_assets(
                partition_files=[partition_path],
                resolved_shard_path=Path("dummy.arrow"),
                tar_path=None,
                image_output_dir=output_dir,
                image_format="png",
                jpeg_quality=95,
                default_patch_size=1,
                compression="none",
            )

        self.assertEqual(result["written_files"], 1)
        written_files = list(output_dir.rglob("*.png"))
        self.assertEqual(len(written_files), 1)

        exported_patch = np.asarray(Image.open(written_files[0]).convert("RGB"))
        resized_image = image.resize((2, 2), Image.Resampling.BILINEAR)
        expected_patch = np.asarray(resized_image.crop((1, 0, 2, 1)).convert("RGB"))
        np.testing.assert_array_equal(exported_patch, expected_patch)


if __name__ == "__main__":
    unittest.main()
