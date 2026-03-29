"""JEPA training package."""


def main() -> None:
    from patchselect.jepa.cli import main as cli_main

    cli_main()


__all__ = ["main"]
