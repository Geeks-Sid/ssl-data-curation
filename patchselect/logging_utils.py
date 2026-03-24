"""Logging helpers for patchselect command modules."""

from __future__ import annotations

import argparse
import logging

LOG_LEVELS: dict[str, int] = {
    "quiet": logging.WARNING,
    "progress": logging.INFO,
    "detail": logging.DEBUG,
}

DEFAULT_LOG_LEVEL = "detail"


def add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--log_level",
        default=DEFAULT_LOG_LEVEL,
        choices=tuple(LOG_LEVELS.keys()),
        help=(
            "Logging verbosity: quiet shows warnings/errors, progress shows stage "
            "updates, detail shows per-image/per-bin tracing."
        ),
    )
    return parser


def configure_logging(level_name: str) -> None:
    level = LOG_LEVELS[level_name]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
