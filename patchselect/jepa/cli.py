"""CLI entrypoint for JEPA training."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from patchselect.jepa.config import load_config
from patchselect.jepa.logging_utils import configure_logging
from patchselect.jepa.runner import build_run_paths, ensure_run_directory, train


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train JEPA on curated tar streams.")
    parser.add_argument(
        "--config_file",
        default=None,
        help="Optional YAML config file merged on top of configs/jepa/base.yaml",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the resolved config before training starts",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional shortcut for checkpoint.resume=none|auto|<path>",
    )
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_args()
    if args.resume is not None:
        overrides.append(f"checkpoint.resume={args.resume}")
    cfg, resolved_config, omega_cfg = load_config(args.config_file, overrides)
    paths = build_run_paths(cfg)
    ensure_run_directory(paths, cfg)
    configure_logging(paths.log_path, cfg.logging.log_level)
    if args.print_config:
        print(OmegaConf.to_yaml(omega_cfg))
    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=omega_cfg, f=str(paths.config_path))
    train(
        cfg,
        resolved_config,
        paths=paths,
        repo_root=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
