import io
import json
import random
import shutil
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from patchselect.jepa import checkpointing
from patchselect.jepa.cli import main as jepa_main
from patchselect.jepa.config import load_config
from patchselect.jepa.model import TimmPathologySpatialJEPA, compute_losses
from patchselect.jepa.runner import _handle_nonfinite_grad_norm, build_run_paths


class DummyPatchEmbed(torch.nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.grid_size = (14, 14)
        self.num_patches = 14 * 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DummyEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = DummyPatchEmbed(embed_dim)
        self.num_prefix_tokens = 1
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.norm = torch.nn.LayerNorm(embed_dim)


class DummyWandbRun:
    def __init__(self, run_id: str = "dummy-run") -> None:
        self.id = run_id
        self.project = "unit-tests"
        self.artifacts = []

    def log_artifact(self, artifact, aliases=None):
        self.artifacts.append((artifact, aliases or []))

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class DummyArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class DummyWandbModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.logged = []
        self.runs = []
        self.api = types.SimpleNamespace(api_key="test-key")

    def init(self, **kwargs):
        run_id = kwargs.get("id") or f"run-{len(self.runs)}"
        run = DummyWandbRun(run_id=run_id)
        self.runs.append(run)
        return run

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))

    def Artifact(self, name, type):
        return DummyArtifact(name, type)


class DummyGradScaler:
    def __init__(self, *, enabled: bool, scale: float = 65536.0) -> None:
        self._enabled = enabled
        self._scale = scale
        self.step_calls = 0
        self.update_calls = 0

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale

    def step(self, optimizer) -> None:
        self.step_calls += 1

    def update(self) -> None:
        self.update_calls += 1


class JepaTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="jepa_test_"))
        self.tar_dir = self.temp_dir / "tars"
        self.tar_dir.mkdir(parents=True, exist_ok=True)
        self.output_root = self.temp_dir / "runs"
        self._write_tar(self.tar_dir / "shard-000.tar", [(255, 0, 0), (0, 255, 0)])
        self._write_tar(self.tar_dir / "shard-001.tar", [(0, 0, 255), (255, 255, 0)])

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_tar(
        self,
        tar_path: Path,
        rgb_values: list[tuple[int, int, int]],
        *,
        corrupt: bool = False,
    ) -> None:
        with tarfile.open(tar_path, "w") as archive:
            for index, rgb in enumerate(rgb_values):
                image = Image.new("RGB", (224, 224), rgb)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
                info = tarfile.TarInfo(name=f"sample_{index}.png")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            if corrupt:
                bad_bytes = b"not-a-real-image"
                bad_info = tarfile.TarInfo(name="broken.png")
                bad_info.size = len(bad_bytes)
                archive.addfile(bad_info, io.BytesIO(bad_bytes))

    def _write_config(self, filename: str, extra: dict) -> Path:
        cfg = {
            "data": {
                "tar_glob": str(self.tar_dir / "*.tar"),
                "cache_in_ram": True,
            },
            "runtime": {
                "experiment_name": "tests",
                "run_name": filename.replace(".yaml", ""),
                "output_root": str(self.output_root),
                "device": "cpu",
                "precision": "32",
                "batch_size": 2,
                "max_steps": 3,
            },
            "logging": {
                "log_every_steps": 1,
                "avg_window": 3,
            },
            "checkpoint": {
                "save_every_steps": 1,
                "keep_last_k": 2,
                "resume": "none",
            },
            "wandb": {
                "mode": "offline",
                "project": "unit-tests",
                "log_checkpoints": True,
            },
            "model": {
                "family": "lewm",
                "pred_depth": 2,
                "pred_num_heads": 4,
            },
            "projector": {
                "kind": "mlp_ln",
                "hidden_dim": 64,
                "out_dim": 32,
            },
            "regularizer": {
                "name": "gaussian_sketch",
                "weight": 0.1,
                "num_projections": 32,
            },
        }
        merged = OmegaConf.merge(cfg, extra)
        path = self.temp_dir / filename
        OmegaConf.save(config=OmegaConf.create(merged), f=str(path))
        return path

    def _run_main(self, config_path: Path | list[Path], dummy_wandb: DummyWandbModule) -> None:
        config_paths = [config_path] if isinstance(config_path, Path) else config_path
        argv = ["train_jepa.py"]
        for path in config_paths:
            argv.extend(["--config_file", str(path)])
        with mock.patch(
            "timm.create_model",
            side_effect=lambda *args, **kwargs: DummyEncoder(embed_dim=32),
        ), mock.patch.dict(
            "sys.modules",
            {"wandb": dummy_wandb},
        ), mock.patch(
            "sys.argv",
            argv,
        ):
            jepa_main()

    def test_cpu_smoke_run_writes_artifacts(self) -> None:
        config_path = self._write_config("smoke.yaml", {"wandb": {"mode": "offline"}})
        dummy_wandb = DummyWandbModule()
        self._run_main(config_path, dummy_wandb)

        run_dir = self.output_root / "tests" / "smoke"
        self.assertTrue((run_dir / "config.resolved.yaml").exists())
        self.assertTrue((run_dir / "logs" / "train.log").exists())
        self.assertTrue((run_dir / "metrics.jsonl").exists())
        self.assertTrue((run_dir / "checkpoints" / "latest.pt").exists())
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "completed")
        self.assertGreaterEqual(state["global_step"], 1)

    def test_checkpoint_roundtrip_restores_state(self) -> None:
        config_path = self._write_config("roundtrip.yaml", {})
        cfg, resolved, _ = load_config(str(config_path), [])
        run_dir = build_run_paths(cfg).run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": {"weight": torch.tensor([1.0])},
            "model_training_state": {"mask_sampler_step": 3, "last_mask_metadata": {}},
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"last_epoch": 3},
            "scaler": {},
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
            "config_hash": checkpointing.config_hash(resolved),
            "resume_signature": checkpointing.compute_resume_signature(resolved, []),
            "global_step": 7,
            "images_seen": 14,
            "batches_seen": 7,
            "data_cursor": {"tar_index": 1, "member_index": 4},
            "meter_state": {"window": 3, "meters": {}},
            "wandb_run_id": "resume-me",
            "family": "lewm",
        }
        step_path, latest_path = checkpointing.save_checkpoint_bundle(
            checkpoint_dir=run_dir / "checkpoints",
            payload=payload,
            global_step=7,
            keep_last_k=2,
        )
        self.assertTrue(step_path.exists())
        self.assertTrue(latest_path.exists())
        loaded = checkpointing.load_checkpoint(latest_path, device="cpu")
        self.assertEqual(loaded["global_step"], 7)
        self.assertEqual(loaded["images_seen"], 14)
        self.assertEqual(loaded["data_cursor"]["tar_index"], 1)
        self.assertEqual(loaded["wandb_run_id"], "resume-me")

    def test_resume_reuses_wandb_run_id(self) -> None:
        config_path = self._write_config("resume.yaml", {})
        dummy_wandb = DummyWandbModule()
        self._run_main(config_path, dummy_wandb)
        run_dir = self.output_root / "tests" / "resume"
        first_state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        run_id = first_state["wandb_run_id"]

        resume_cfg = self._write_config(
            "resume_second.yaml",
            {
                "runtime": {"run_name": "resume"},
                "checkpoint": {"resume": "auto"},
            },
        )
        self._run_main(resume_cfg, dummy_wandb)
        second_state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(second_state["wandb_run_id"], run_id)

    def test_signature_mismatch_rejects_resume(self) -> None:
        config_path = self._write_config("sig_a.yaml", {})
        dummy_wandb = DummyWandbModule()
        self._run_main(config_path, dummy_wandb)

        mismatch_cfg = self._write_config(
            "sig_b.yaml",
            {
                "runtime": {"run_name": "sig_a"},
                "checkpoint": {"resume": "auto"},
                "augment": {"image_size": 128},
            },
        )
        with mock.patch(
            "timm.create_model",
            side_effect=lambda *args, **kwargs: DummyEncoder(embed_dim=32),
        ), mock.patch.dict("sys.modules", {"wandb": dummy_wandb}), mock.patch(
            "sys.argv",
            ["train_jepa.py", "--config_file", str(mismatch_cfg)],
        ):
            with self.assertRaisesRegex(RuntimeError, "signature"):
                jepa_main()

    def test_corrupt_image_is_tracked_and_training_continues(self) -> None:
        self._write_tar(self.tar_dir / "shard-002.tar", [(12, 12, 12)], corrupt=True)
        config_path = self._write_config("corrupt.yaml", {})
        dummy_wandb = DummyWandbModule()
        self._run_main(config_path, dummy_wandb)

        state = json.loads(
            (self.output_root / "tests" / "corrupt" / "state.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(state["health"]["corrupt_images"], 1)
        self.assertEqual(state["status"], "completed")

    def test_invalid_family_target_encoder_combination_rejected(self) -> None:
        config_path = self._write_config(
            "invalid_family.yaml",
            {
                "model": {"family": "ijepa"},
                "target_encoder": {"kind": "shared"},
            },
        )
        with self.assertRaisesRegex(ValueError, "ijepa requires target_encoder.kind=ema"):
            load_config(str(config_path), [])

    def test_block_mask_metadata_and_gaussian_regularizer_are_finite(self) -> None:
        config_path = self._write_config(
            "mask_block.yaml",
            {
                "masking": {
                    "strategy": "block_targets",
                    "num_targets": 2,
                    "target_scale_range": [0.1, 0.15],
                    "aspect_ratio_range": [0.8, 1.2],
                    "context_min_keep": 16,
                },
            },
        )
        cfg, _, _ = load_config(str(config_path), [])
        with mock.patch(
            "timm.create_model",
            side_effect=lambda *args, **kwargs: DummyEncoder(embed_dim=32),
        ):
            model = TimmPathologySpatialJEPA(cfg, mask_seed=123)
            images = torch.randn(2, 3, 224, 224)
            output = model(images)
            loss, metrics = compute_losses(output, cfg.loss, cfg.regularizer)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("train/gaussian_sketch", metrics)
        self.assertGreaterEqual(output.mask_metadata["num_targets_used"], 1)
        self.assertGreater(output.mask_metadata["target_token_count"], 0)

    def test_nonfinite_grad_norm_is_skipped_when_amp_scaler_is_enabled(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter.grad = torch.full_like(parameter, float("inf"))
        scaler = DummyGradScaler(enabled=True)

        should_skip = _handle_nonfinite_grad_norm(
            grad_norm=float("nan"),
            optimizer=optimizer,
            scaler=scaler,
            global_step=1,
        )

        self.assertTrue(should_skip)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)
        self.assertIsNone(parameter.grad)

    def test_nonfinite_grad_norm_raises_without_amp_scaler(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = DummyGradScaler(enabled=False)

        with self.assertRaisesRegex(RuntimeError, "Non-finite gradient norm at step 2: nan"):
            _handle_nonfinite_grad_norm(
                grad_norm=float("nan"),
                optimizer=optimizer,
                scaler=scaler,
                global_step=1,
            )

    def test_ijepa_uses_ema_target_and_supports_resume(self) -> None:
        config_path = self._write_config(
            "ijepa.yaml",
            {
                "model": {"family": "ijepa"},
                "target_encoder": {"kind": "ema", "ema_momentum": 0.99},
                "regularizer": {"name": "none", "weight": 0.0},
            },
        )
        dummy_wandb = DummyWandbModule()
        self._run_main(config_path, dummy_wandb)

        run_dir = self.output_root / "tests" / "ijepa"
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["family"], "ijepa")
        self.assertEqual(state["target_encoder_kind"], "ema")
        checkpoint = checkpointing.load_checkpoint(run_dir / "checkpoints" / "latest.pt", device="cpu")
        self.assertIn("model_training_state", checkpoint)
        self.assertEqual(checkpoint["family"], "ijepa")

        resume_cfg = self._write_config(
            "ijepa_resume.yaml",
            {
                "runtime": {"run_name": "ijepa"},
                "model": {"family": "ijepa"},
                "target_encoder": {"kind": "ema", "ema_momentum": 0.99},
                "regularizer": {"name": "none", "weight": 0.0},
                "checkpoint": {"resume": "auto"},
            },
        )
        self._run_main(resume_cfg, dummy_wandb)
        resumed_state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(resumed_state["wandb_run_id"], state["wandb_run_id"])

    def test_checkpoint_retention_keeps_latest_and_recent_steps(self) -> None:
        checkpoint_dir = self.temp_dir / "ckpts"
        payload = {
            "model": {},
            "model_training_state": {"mask_sampler_step": 0, "last_mask_metadata": {}},
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": None,
            "scaler": {},
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
            "config_hash": "abc",
            "resume_signature": {"hash": "def", "payload": {}},
            "global_step": 0,
            "images_seen": 0,
            "batches_seen": 0,
            "data_cursor": {"tar_index": 0, "member_index": 0},
            "meter_state": {"window": 3, "meters": {}},
            "wandb_run_id": None,
            "family": "lewm",
        }
        for step in range(1, 6):
            payload["global_step"] = step
            checkpointing.save_checkpoint_bundle(
                checkpoint_dir=checkpoint_dir,
                payload=payload,
                global_step=step,
                keep_last_k=2,
            )
        numbered = sorted(path.name for path in checkpoint_dir.glob("step_*.pt"))
        self.assertEqual(numbered, ["step_00000004.pt", "step_00000005.pt"])
        self.assertTrue((checkpoint_dir / "latest.pt").exists())
