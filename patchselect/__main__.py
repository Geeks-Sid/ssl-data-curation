"""Command dispatcher for patchselect."""

from __future__ import annotations

import sys

from patchselect.export_images import main as export_images_main
from patchselect.run_global_selection import main as global_main
from patchselect.run_local_selection import main as local_main


HELP_TEXT = """usage: patchselect {export-images,local-select,global-select} [args]

Patch selection tools for Arrow-backed IHC datasets.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(HELP_TEXT.strip())
        return

    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "export-images":
        export_images_main()
    elif command == "local-select":
        local_main()
    elif command == "global-select":
        global_main()
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
