import shutil
import unittest
from pathlib import Path

from patchselect.arrow_utils import discover_arrow_files
from patchselect.export_tars import build_data_dir_lookup


class ExportTarsPathResolutionTest(unittest.TestCase):
    def test_discover_arrow_files_accepts_direct_file_and_nested_dirs(self) -> None:
        root = Path(__file__).resolve().parents[1] / "out" / "_test_export_tars"
        shutil.rmtree(root, ignore_errors=True)
        nested = root / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        arrow_path = nested / "sample.arrow"
        arrow_path.write_bytes(b"arrow")
        try:
            self.assertEqual(discover_arrow_files(arrow_path), [arrow_path])
            self.assertEqual(discover_arrow_files(root), [arrow_path])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_build_data_dir_lookup_returns_empty_for_missing_dir(self) -> None:
        missing = Path(__file__).resolve().parents[1] / "out" / "_missing_arrow_dir"
        shutil.rmtree(missing, ignore_errors=True)
        lookup = build_data_dir_lookup(missing)
        self.assertEqual(lookup, {})


if __name__ == "__main__":
    unittest.main()
