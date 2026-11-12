"""Synthetic inventory checks; no AI-Hub data required."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_dataset_validation import InventoryError, inventory_dataset, main


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, names):
        path = self.root / "sample.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                for name in names:
                    archive.writestr(name, "private synthetic payload")
        return path

    def test_empty(self):
        report = inventory_dataset(self.root).summary()
        self.assertEqual(report["files"], 0)
        self.assertEqual(report["ctc_compatibility"], "not_established")

    def test_categories_do_not_claim_keypoints(self):
        self.archive(["morpheme/00/synthetic_morpheme.json", "keypoints/sample.json",
                      "x.npy", "x.mp4", "folder/", "x.txt"])
        report = inventory_dataset(self.root).summary()
        self.assertEqual(report["archive_members"], 5)
        self.assertEqual(report["archive_categories"], {
            "morpheme_json": 1, "unclassified_data_candidate": 2,
            "video_candidate": 1, "other": 1})
        self.assertEqual(report["keypoint_schema"], "not_inspected")

    def test_loose_files_and_uppercase(self):
        (self.root / "a.JSON").write_text("{}")
        (self.root / "x.MP4").touch()
        self.assertEqual(dict(inventory_dataset(self.root).loose_categories),
                         {"unclassified_data_candidate": 1, "video_candidate": 1})

    def test_loose_and_archive_relative_paths_match(self):
        relative = "morpheme/00/sample.json"
        path = self.root / relative
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        self.archive([relative])
        report = inventory_dataset(self.root)
        self.assertEqual(report.loose_categories, report.archive_categories)

    def test_root_link_rejected(self):
        for method in ("is_symlink", "is_junction"):
            with self.subTest(method=method), patch.object(Path, method, return_value=True):
                with self.assertRaisesRegex(InventoryError, "^invalid_root$"):
                    inventory_dataset(self.root)

    def test_child_link_rejected(self):
        (self.root / "child").touch()
        original = Path.is_junction
        with patch.object(Path, "is_junction", lambda p: p.name == "child" or original(p)):
            with self.assertRaisesRegex(InventoryError, "^linked_path$"):
                inventory_dataset(self.root)

    def test_walk_error_sanitized(self):
        def failed_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("private-path"))
            return iter(())
        with patch("fingerspelling.crowd_dataset_validation.os.walk", failed_walk):
            with self.assertRaisesRegex(InventoryError, "^directory_read$"):
                inventory_dataset(self.root)

    def test_does_not_read_zip_payload(self):
        self.archive(["sample.json"])
        with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("payload read")):
            self.assertEqual(inventory_dataset(self.root).archive_members, 1)

    def test_duplicates(self):
        self.archive(["a.json"] * 3)
        self.assertEqual(inventory_dataset(self.root).duplicate_members, 2)

    def test_duplicate_names_scoped_per_archive(self):
        self.archive(["a.json"])
        with zipfile.ZipFile(self.root / "second.zip", "w") as archive:
            archive.writestr("a.json", "{}")
        self.assertEqual(inventory_dataset(self.root).duplicate_members, 0)

    def test_source_unchanged_and_not_extracted(self):
        path = self.archive(["../escape.json", "C:/escape.json", "folder/file.json"])
        before = hashlib.sha256(path.read_bytes()).digest()
        inventory_dataset(self.root)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_bad_archive_sanitized(self):
        (self.root / "private-id.zip").write_bytes(b"invalid")
        with self.assertRaisesRegex(InventoryError, "^inventory_read$"):
            inventory_dataset(self.root)

    def test_missing_root(self):
        with self.assertRaisesRegex(InventoryError, "^invalid_root$"):
            inventory_dataset(self.root / "absent")

    def test_file_root(self):
        path = self.archive([])
        with self.assertRaises(InventoryError):
            inventory_dataset(path)

    def test_invalid_limits(self):
        for value in (True, 0, -1, 1.5, None):
            for option in ("max_files", "max_members"):
                with self.subTest(value=value, option=option):
                    with self.assertRaisesRegex(InventoryError, "^invalid_limit$"):
                        inventory_dataset(self.root, **{option: value})

    def test_file_limit(self):
        (self.root / "a").touch()
        (self.root / "b").touch()
        with self.assertRaisesRegex(InventoryError, "^file_limit$"):
            inventory_dataset(self.root, max_files=1)

    def test_total_member_limit(self):
        self.archive(["a", "b"])
        with zipfile.ZipFile(self.root / "second.zip", "w") as archive:
            archive.writestr("c", "")
        with self.assertRaisesRegex(InventoryError, "^member_limit$"):
            inventory_dataset(self.root, max_members=2)

    def test_large_inventory(self):
        self.archive([f"keypoints/synthetic{i}.json" for i in range(10000)])
        report = inventory_dataset(self.root, max_members=10000)
        self.assertEqual(report.archive_members, 10000)

    def test_cli_only_aggregates(self):
        self.archive(["private-synthetic-id.json"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.root)]), 0)
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn(str(self.root), output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["archives"], 1)

    def test_cli_failure(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.root / "absent")]), 2)
        self.assertEqual(json.loads(output.getvalue()), {"error": "invalid_root"})


if __name__ == "__main__":
    unittest.main()
