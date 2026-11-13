"""Synthetic source diagnosis tests, independent of private CROWD payloads."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_keypoints import KeypointError, parse_keypoint_json
from fingerspelling.crowd_source_validation import ConfidenceDiagnostics, audit_source
from test_crowd_keypoints import document, member


class SourceTests(unittest.TestCase):
    def test_existing_eye_slots_follow_official_pose_enum(self):
        import ast
        from mediapipe.python.solutions.pose import PoseLandmark
        path = Path(__file__).resolve().parents[1] / "src" / "landmarks.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "POSE_IDX"
                                  for target in node.targets))
        indices = ast.literal_eval(assignment.value)
        self.assertEqual([PoseLandmark(index).name for index in indices[1:3]],
                         ["LEFT_EYE", "RIGHT_EYE"])
        self.assertEqual((PoseLandmark.LEFT_EYE_INNER.value,
                          PoseLandmark.RIGHT_EYE_INNER.value), (1, 4))

    def test_raw_confidence_measured_without_training_bypass(self):
        doc = document()
        doc["people"]["face_keypoints_2d"][2] = -.000175198
        doc["people"]["hand_right_keypoints_2d"][5] = 1.20531
        raw = json.dumps(doc).encode()
        diagnostic = ConfidenceDiagnostics()
        self.assertEqual(diagnostic.inspect(raw), "confidence")
        self.assertEqual(diagnostic.groups["face_keypoints_2d"]["negative"], 1)
        right = diagnostic.groups["hand_right_keypoints_2d"]
        self.assertEqual(right["maximum"], 1.20531)
        self.assertEqual(right["invalid_by_joint"], [0, 1] + [0] * 19)
        with self.assertRaisesRegex(KeypointError, "^confidence$"):
            parse_keypoint_json(raw)

    def test_boundary_values_and_empty_hand(self):
        doc = document()
        doc["people"]["pose_keypoints_2d"] = [0, 0, 0] * 24 + [1, 2, 1]
        doc["people"]["hand_left_keypoints_2d"] = []
        stats = ConfidenceDiagnostics()
        self.assertIsNone(stats.inspect(json.dumps(doc).encode()))
        self.assertEqual(stats.groups["pose_keypoints_2d"]["zero"], 24)
        self.assertEqual(stats.groups["hand_left_keypoints_2d"]["empty_frames"], 1)

    def test_structure_and_nonfinite_fail_before_aggregation(self):
        for value, code in ((True, "coordinate_type"), ("0.5", "coordinate_type"),
                            (float("nan"), "nonfinite"), (float("inf"), "nonfinite"),
                            (10**400, "coordinate_value")):
            with self.subTest(code=code, type=type(value).__name__):
                doc = document()
                doc["people"]["hand_right_keypoints_2d"][2] = value
                stats = ConfidenceDiagnostics()
                with self.assertRaisesRegex(KeypointError, "^" + code + "$"):
                    stats.inspect(json.dumps(doc).encode())
                self.assertEqual(stats.groups["pose_keypoints_2d"]["joints"], 0)

    def test_missing_and_wrong_joint_count(self):
        for remove in (True, False):
            doc = document()
            if remove:
                del doc["people"]["hand_left_keypoints_2d"]
            else:
                doc["people"]["hand_left_keypoints_2d"].pop()
            with self.assertRaisesRegex(KeypointError, "^joint_count$"):
                ConfidenceDiagnostics().inspect(json.dumps(doc).encode())

    def test_full_scan_continues_and_preserves_zip(self):
        bad = document()
        bad["people"]["face_keypoints_2d"][2] = 1.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(member(0), json.dumps(bad))
                archive.writestr(member(1), "{")
                archive.writestr(member(2), json.dumps(document()))
                archive.writestr(member(0, fs="0002"), json.dumps(document()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = audit_source(path, expected_video_ids=["NIA_SL_FS0001_CROWD00_F.mp4"])
            self.assertEqual(report["errors"], {"confidence": 1, "json_document": 1})
            self.assertEqual(report["strict_frames"], 2)
            self.assertEqual(report["strict_sequences"], 1)
            self.assertEqual(report["confidence_invalid_frames_per_sequence"], {0: 1, 1: 1})
            self.assertEqual(report["label_video_matches"], 1)
            self.assertEqual(report["keypoint_without_label"], 1)
            self.assertEqual(report["source_sha256"], digest)
            self.assertTrue(report["source_hash_unchanged"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(report["ctc_usable_samples"], 0)
            self.assertIsNone(report["missing_joint_ratio"])
            self.assertNotIn("NIA_SL", json.dumps(report))

    def test_duplicate_frame_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(member(), json.dumps(document()))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(member(), json.dumps(document()))
            with self.assertRaisesRegex(KeypointError, "^duplicate_frame$"):
                audit_source(path)

    def test_training_and_inference_share_missing_eye_policy(self):
        import numpy as np
        from fingerspelling.crowd_keypoint_adapter import (
            adapt_crowd_sequence, adapt_mediapipe53, to_ctc53,
        )
        from test_crowd_keypoint_adapter import SPEC, VIDEO
        frame = parse_keypoint_json(json.dumps(document()).encode())
        common = adapt_crowd_sequence([frame, frame], [0, 2], source=SPEC,
                                      video_id=VIDEO, fps=30)
        training = to_ctc53(common, allow_missing_inner_eyes=True)
        camera_xy = training["xy"].copy()
        camera_mask = training["mask"].copy()
        camera_xy[:, 1:3] = .75
        camera_mask[:, 1:3] = True
        inference = to_ctc53(adapt_mediapipe53(
            camera_xy, camera_mask, timestamps=training["timestamps"]),
            allow_missing_inner_eyes=True)
        for key in ("xy", "mask", "timestamps"):
            np.testing.assert_array_equal(training[key], inference[key])
        self.assertFalse(inference["mask"][:, 1:3].any())
        np.testing.assert_allclose(training["timestamps"], [0, 2 / 30])


if __name__ == "__main__":
    unittest.main()
