"""Aggregate audit regressions on synthetic ZIPs."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_crowd_keypoints import audit_archive, main
from test_crowd_keypoints import document, member


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.zip"

    def write(self, entries):
        with zipfile.ZipFile(self.path, "w") as archive:
            for name, raw in entries:
                archive.writestr(name, raw)

    def test_counts_and_label_coverage(self):
        raw = json.dumps(document())
        self.write([(member(0), raw), (member(2), raw)])
        before = hashlib.sha256(self.path.read_bytes()).digest()
        result = audit_archive(self.path, expected_video_ids=[
            "NIA_SL_FS0001_CROWD00_F.mp4", "NIA_SL_FS0002_CROWD00_F.mp4"])
        self.assertEqual(result["json_total"], 2)
        self.assertEqual(result["json_valid"], 2)
        self.assertEqual(result["missing_frame_indices"], 1)
        self.assertEqual(result["label_video_matches"], 1)
        self.assertEqual(result["label_without_keypoint"], 1)
        self.assertEqual(result["verified_conversion_success"], 0)
        self.assertEqual(result["conversion_blocked_sequences"], 1)
        self.assertTrue(result["source_hash_unchanged"])
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).digest())
        self.assertNotIn("NIA", json.dumps(result))

    def test_errors_continue(self):
        self.write([(member(), "invalid"), (member(1), json.dumps(document()))])
        result = audit_archive(self.path)
        self.assertEqual(result["json_total"], 2)
        self.assertEqual(result["json_valid"], 1)
        self.assertEqual(result["errors"], {"json_document": 1})
        self.assertEqual(result["valid_sequences"], 0)

    def test_empty_archive(self):
        self.write([])
        result = audit_archive(self.path)
        self.assertEqual(result["sequences"], 0)
        self.assertIsNone(result["confidence_zero_ratio"])

    def test_no_observed_coordinates(self):
        doc = document()
        for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
            doc["people"][key] = [0, 0, 0] * (len(doc["people"][key]) // 3)
        self.write([(member(), json.dumps(doc))])
        result = audit_archive(self.path)
        # A valid all-missing frame remains unsuitable for model training.
        self.assertEqual(result["confidence_zero_ratio"], 1)
        json.dumps(result, allow_nan=False)

    def test_cli_bad_zip(self):
        self.path.write_bytes(b"invalid")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.path)]), 2)
        self.assertEqual(json.loads(output.getvalue()), {"error": "archive_read"})

    def test_cli_missing_source(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.path)]), 2)
        self.assertEqual(json.loads(output.getvalue()), {"error": "archive_read"})

    def test_cli_invalid_inventory_root(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.path), "--inventory-root", str(self.path / "absent")]), 2)
        self.assertEqual(json.loads(output.getvalue()), {"error": "invalid_root"})


if __name__ == "__main__":
    unittest.main()
