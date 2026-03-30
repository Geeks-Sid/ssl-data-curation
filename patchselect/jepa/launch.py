"""Launcher utilities for named JEPA presets and curated sweeps."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from patchselect.jepa.presets import resolve_preset, resolve_sweep


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_executable() -> str:
    return os.environ.get("PYTHON", sys.executable or "python")


def build_run_name(preset: str) -> str:
    suffix = os.environ.get("RUN_SUFFIX", "").strip()
    return f"{preset}-{suffix}" if suffix else preset


def build_command(preset: str) -> list[str]:
    command = [_python_executable(), "-m", "patchselect.jepa.cli"]
    for config_path in resolve_preset(preset):
        command.extend(["--config_file", config_path])
    command.extend(
        [
            f"runtime.experiment_name={preset}",
            f"runtime.run_name={build_run_name(preset)}",
        ]
    )
    resume = os.environ.get("RESUME")
    resume_mode = resume.strip() if resume is not None else "auto"
    command.extend(["--resume", resume_mode or "auto"])
    return command


def run_preset(preset: str) -> int:
    command = build_command(preset)
    print(shlex.join(command))
    completed = subprocess.run(command, cwd=str(_repo_root()), check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch curated JEPA presets or sweeps.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Run a single named preset")
    single.add_argument("name")

    sweep = subparsers.add_parser("sweep", help="Run a curated sweep group")
    sweep.add_argument("group")

    args = parser.parse_args(argv)
    continue_on_error = os.environ.get("CONTINUE_ON_ERROR", "0") == "1"

    if args.mode == "single":
        return run_preset(args.name)

    exit_code = 0
    for preset in resolve_sweep(args.group):
        exit_code = run_preset(preset)
        if exit_code != 0 and not continue_on_error:
            return exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
