import os
import unittest
from pathlib import Path
from unittest import mock

from patchselect.jepa.launch import build_command, build_run_name, main
from patchselect.jepa.presets import resolve_preset, resolve_sweep


class JepaLaunchTest(unittest.TestCase):
    def test_preset_resolution_returns_expected_stack(self) -> None:
        config_stack = resolve_preset("lewm_default")
        self.assertEqual(Path(config_stack[0]), Path("configs/jepa/families/lewm.yaml"))
        self.assertIn(
            Path("configs/jepa/ablations/regularizer_gaussian_sketch.yaml"),
            [Path(entry) for entry in config_stack],
        )

    def test_sweep_resolution_is_ordered(self) -> None:
        self.assertEqual(
            resolve_sweep("core"),
            (
                "lewm_default",
                "ijepa_default",
                "lewm_tokenmask_ablation",
                "ijepa_sigreg_ablation",
                "lewm_heavy_projector_ablation",
                "ijepa_legacy_aug_ablation",
            ),
        )

    def test_build_command_uses_all_config_files_and_run_name(self) -> None:
        with mock.patch.dict(os.environ, {"RUN_SUFFIX": "trial"}, clear=False):
            command = build_command("ijepa_default")
        self.assertIn("--config_file", command)
        self.assertIn(str(Path("configs/jepa/families/ijepa.yaml")), command)
        self.assertIn("runtime.experiment_name=ijepa_default", command)
        self.assertIn("runtime.run_name=ijepa_default-trial", command)

    def test_build_run_name_without_suffix(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(build_run_name("lewm_default"), "lewm_default")

    def test_single_launcher_invokes_one_subprocess(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=completed) as run_mock:
            exit_code = main(["single", "lewm_default"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(run_mock.call_count, 1)

    def test_sweep_stops_on_first_failure_when_not_continuing(self) -> None:
        with mock.patch("subprocess.run", side_effect=[mock.Mock(returncode=3), mock.Mock(returncode=0)]) as run_mock:
            with mock.patch.dict(os.environ, {"CONTINUE_ON_ERROR": "0"}, clear=False):
                exit_code = main(["sweep", "masking"])
        self.assertEqual(exit_code, 3)
        self.assertEqual(run_mock.call_count, 1)
