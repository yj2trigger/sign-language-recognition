"""Synthetic examples of the observed schema, with independent expected counts."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_keypoints import CrowdKeypointArchive, KeypointError, parse_keypoint_json


def document():
    return {"version": 1.3, "people": {"person_id": -1,
        "pose_keypoints_2d": [1., 2., .8] * 25,
        "face_keypoints_2d": [1., 2., .8] * 70,
        "hand_left_keypoints_2d": [1., 2., .8] * 21,
        "hand_right_keypoints_2d": [1., 2., .8] * 21,
        "pose_keypoints_3d": [], "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []},
        "camparam": {}}


def member(frame=0, group="00", fs="0001"):
    base = f"NIA_SL_FS{fs}_CROWD{group}_F"
    return f"keypoint/{group}/{base}/{base}_{frame:012d}_keypoints.json"


class ParserTests(unittest.TestCase):
    def parse(self, doc):
        return parse_keypoint_json(json.dumps(doc).encode())

    def test_observed_schema(self):
        doc = document()
        before = copy.deepcopy(doc)
        frame = self.parse(doc)
        self.assertEqual([a.shape for a in (frame.pose, frame.face, frame.left_hand, frame.right_hand)],
                         [(25, 3), (70, 3), (21, 3), (21, 3)])
        np.testing.assert_array_equal(frame.pose[0], [1, 2, .8])
        self.assertEqual(doc, before)
        self.assertFalse(frame.pose.flags.writeable)

    def test_missing_field(self):
        doc = document()
        del doc["people"]["hand_left_keypoints_2d"]
        with self.assertRaisesRegex(KeypointError, "^joint_count$"):
            self.parse(doc)

    def test_joint_count(self):
        doc = document()
        doc["people"]["pose_keypoints_2d"].pop()
        with self.assertRaisesRegex(KeypointError, "^joint_count$"):
            self.parse(doc)

    def test_empty_group(self):
        doc = document()
        doc["people"]["hand_left_keypoints_2d"] = []
        self.assertEqual(self.parse(doc).left_hand.shape, (0, 3))

    def test_numeric_types(self):
        for value in (True, "1", None, {}, []):
            doc = document()
            doc["people"]["pose_keypoints_2d"][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(KeypointError, "^coordinate_type$"):
                self.parse(doc)

    def test_nonfinite(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            doc = document()
            doc["people"]["pose_keypoints_2d"][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(KeypointError, "^nonfinite$"):
                self.parse(doc)

    def test_confidence(self):
        for value in (-.1, 1.01):
            doc = document()
            doc["people"]["pose_keypoints_2d"][2] = value
            with self.assertRaisesRegex(KeypointError, "^confidence$"):
                self.parse(doc)

    def test_people_schema(self):
        doc = document()
        doc["people"] = [doc["people"]]
        with self.assertRaisesRegex(KeypointError, "^people$"):
            self.parse(doc)

    def test_unsupported_3d(self):
        doc = document()
        doc["people"]["pose_keypoints_3d"] = [1, 2, 3, 1]
        with self.assertRaisesRegex(KeypointError, "^unsupported_3d$"):
            self.parse(doc)

    def test_version(self):
        doc = document()
        doc["version"] = 2
        with self.assertRaisesRegex(KeypointError, "^version$"):
            self.parse(doc)

    def test_invalid_json(self):
        for raw in (b"", b"{", b"\xff", b"[]"):
            with self.subTest(raw=raw), self.assertRaises(KeypointError):
                parse_keypoint_json(raw)

    def test_duplicate_key(self):
        with self.assertRaisesRegex(KeypointError, "^duplicate_json_key$"):
            parse_keypoint_json(b'{"version":1.3,"version":1.3}')

    def test_size_limit(self):
        with self.assertRaisesRegex(KeypointError, "^json_size$"):
            parse_keypoint_json(b"{}", max_bytes=1)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.zip"

    def write(self, entries, compression=zipfile.ZIP_STORED):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "w", compression=compression) as archive:
                for name, raw in entries:
                    archive.writestr(name, raw)

    def test_order_gaps_crc_and_immutability(self):
        raw = json.dumps(document()).encode()
        self.write([(member(2), raw), (member(0), raw)])
        before = hashlib.sha256(self.path.read_bytes()).digest()
        with CrowdKeypointArchive(self.path) as archive:
            sequence = archive.sequences[0]
            self.assertEqual(sequence.frame_indices, (0, 2))
            self.assertEqual(sequence.missing_frames, 1)
            self.assertEqual(len(list(archive.read_frames(sequence))), 2)
            self.assertNotIn("NIA", repr(sequence))
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).digest())
        self.assertEqual(len(list(self.path.parent.iterdir())), 1)

    def test_empty_archive(self):
        self.write([])
        with CrowdKeypointArchive(self.path) as archive:
            self.assertEqual(archive.sequences, ())

    def test_duplicate_frame(self):
        self.write([(member(), b"{}"), (member(), b"{}")])
        with self.assertRaisesRegex(KeypointError, "^duplicate_frame$"):
            with CrowdKeypointArchive(self.path):
                pass

    def test_invalid_member_identity(self):
        for name in (member().replace("keypoint/00/", "keypoint/01/"),
                     "../" + member(), member().replace("000000000000", "0"),
                     member().replace("_000000000000_", "_bad_")):
            self.write([(name, b"{}")])
            with self.subTest(name=name), self.assertRaisesRegex(KeypointError, "^member_name$"):
                with CrowdKeypointArchive(self.path):
                    pass

    def test_video_folder_filename_mismatch(self):
        name = member().replace("/NIA_SL_FS0001_CROWD00_F_", "/NIA_SL_FS0002_CROWD00_F_")
        self.write([(name, b"{}")])
        with self.assertRaisesRegex(KeypointError, "^member_name$"):
            with CrowdKeypointArchive(self.path):
                pass

    def test_limits(self):
        self.write([(member(), b"x" * 100), (member(1), b"x" * 100)])
        for options, code in (({"max_members": 1}, "member_limit"),
                              ({"max_json_bytes": 99}, "member_size"),
                              ({"max_total_bytes": 199}, "total_size")):
            with self.subTest(code=code), self.assertRaisesRegex(KeypointError, "^" + code + "$"):
                with CrowdKeypointArchive(self.path, **options):
                    pass

    def test_compression_ratio(self):
        self.write([(member(), b"0" * 10000)], zipfile.ZIP_DEFLATED)
        with self.assertRaisesRegex(KeypointError, "^compression_ratio$"):
            with CrowdKeypointArchive(self.path, max_ratio=10):
                pass

    def test_bad_archive(self):
        self.path.write_bytes(b"invalid")
        with self.assertRaisesRegex(KeypointError, "^archive_read$"):
            with CrowdKeypointArchive(self.path):
                pass

    def test_crc_error_and_continue(self):
        raw = json.dumps(document()).encode()
        self.write([(member(), raw), (member(1), raw)])
        content = bytearray(self.path.read_bytes())
        name_len, extra_len = struct.unpack_from("<HH", content, 26)
        content[30 + name_len + extra_len] ^= 1
        self.path.write_bytes(content)
        with CrowdKeypointArchive(self.path) as archive:
            results = list(archive.iter_frame_results(archive.sequences[0]))
            self.assertEqual(results[0], (None, "member_read"))
            self.assertIsNone(results[1][1])

    def test_closed_archive(self):
        self.write([(member(), json.dumps(document()))])
        with CrowdKeypointArchive(self.path) as archive:
            sequence = archive.sequences[0]
        with self.assertRaisesRegex(KeypointError, "^archive_closed$"):
            list(archive.read_frames(sequence))

    def test_invalid_limit(self):
        with self.assertRaisesRegex(KeypointError, "^invalid_limit$"):
            CrowdKeypointArchive(self.path, max_members=True)


if __name__ == "__main__":
    unittest.main()
