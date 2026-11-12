"""Hand-written expectations for common51 and explicit masked CTC53 output."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_keypoints import KeypointError, parse_keypoint_json
from fingerspelling.crowd_keypoint_adapter import (
    SourceSpec, adapt_crowd_sequence, adapt_labeled_sequence, adapt_mediapipe53,
    conversion_blockers, to_ctc53,
)
from test_crowd_keypoints import document

VIDEO = "NIA_SL_FS0001_CROWD00_F.mp4"
SPEC = SourceSpec("openpose_body25_hand21", "pixels_top_left",
                  "anatomical_unmirrored", (160, 90), "synthetic verified fixture")


def frame():
    doc = document()
    for name in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        count = len(doc["people"][name]) // 3
        doc["people"][name] = [value for i in range(count) for value in (float(i), 10., .8)]
    return parse_keypoint_json(json.dumps(doc).encode())


class AdapterTests(unittest.TestCase):
    def adapt(self, frames=None, indices=None, **options):
        if frames is None:
            frames = [frame()]
        if indices is None:
            indices = [0]
        return adapt_crowd_sequence(frames, indices, video_id=VIDEO,
                                    **{"source": SPEC, "fps": 30, **options})

    def test_joint_mapping_independent_expected(self):
        common = self.adapt()
        # BODY_25 L/R ear=18/17, shoulders=5/2, elbows=6/3, wrists=7/4.
        expected_body_x = np.array([0, 18, 17, 5, 2, 6, 3, 7, 4]) / 160
        np.testing.assert_allclose(common.xy[0, :9, 0], expected_body_x)
        np.testing.assert_allclose(common.xy[0, 9:30, 0], np.arange(21) / 160)
        self.assertEqual(common.xy.shape, (1, 51, 2))
        self.assertTrue(common.mask.all())

    def test_left_right_no_swap(self):
        doc = document()
        doc["people"]["hand_left_keypoints_2d"] = [10, 20, 1] * 21
        doc["people"]["hand_right_keypoints_2d"] = [30, 40, 1] * 21
        common = self.adapt([parse_keypoint_json(json.dumps(doc).encode())])
        np.testing.assert_allclose(common.xy[0, 9], [10 / 160, 20 / 90])
        np.testing.assert_allclose(common.xy[0, 30], [30 / 160, 40 / 90])

    def test_portrait_padding_matches_integer_geometry(self):
        doc = document()
        doc["people"]["pose_keypoints_2d"] = [45, 80, 1] * 25
        common = self.adapt([parse_keypoint_json(json.dumps(doc).encode())],
                            source=replace(SPEC, image_size=(90, 160)))
        # round(160*16/9)=284, left padding=(284-90)//2=97.
        np.testing.assert_allclose(common.xy[0, 0], [(45 + 97) / 284, .5])

    def test_wide_padding(self):
        common = self.adapt(source=replace(SPEC, image_size=(160, 50)))
        np.testing.assert_allclose(common.xy[0, 0], [0, (10 + 20) / 90])

    def test_normalized_coordinates(self):
        doc = document()
        for name in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
            count = len(doc["people"][name]) // 3
            doc["people"][name] = [.25, .5, 1] * count
        common = self.adapt([parse_keypoint_json(json.dumps(doc).encode())],
                            source=replace(SPEC, coordinates="normalized_top_left"))
        np.testing.assert_array_equal(common.xy[0, 0], [.25, .5])

    def test_unverified_metadata(self):
        variants = [None, replace(SPEC, layout="unknown"),
                    replace(SPEC, handedness="anatomical_mirrored"),
                    replace(SPEC, coordinates="unknown"), replace(SPEC, image_size=None),
                    replace(SPEC, image_size=(True, 90)), replace(SPEC, evidence="")]
        for spec in variants:
            with self.subTest(spec=spec), self.assertRaises(KeypointError):
                self.adapt(source=spec)
        self.assertEqual(conversion_blockers(None), ("source_spec_missing",))

    def test_confidence_zero_missing_even_at_threshold_zero(self):
        doc = document()
        doc["people"]["pose_keypoints_2d"][:3] = [0, 0, 0]
        doc["people"]["hand_left_keypoints_2d"] = []
        common = self.adapt([parse_keypoint_json(json.dumps(doc).encode())], confidence_threshold=0)
        self.assertFalse(common.mask[0, 0])
        self.assertFalse(common.mask[0, 9:30].any())
        self.assertEqual(np.count_nonzero(common.xy[~common.mask]), 0)

    def test_threshold_boundary_and_zero_coordinate(self):
        doc = document()
        doc["people"]["pose_keypoints_2d"][:3] = [0, 0, .5]
        common = self.adapt([parse_keypoint_json(json.dumps(doc).encode())])
        self.assertTrue(common.mask[0, 0])
        np.testing.assert_array_equal(common.xy[0, 0], [0, 0])

    def test_coordinate_bounds(self):
        doc = document()
        doc["people"]["pose_keypoints_2d"][0] = 161
        with self.assertRaisesRegex(KeypointError, "^coordinate_bounds$"):
            self.adapt([parse_keypoint_json(json.dumps(doc).encode())])

    def test_frame_index_gaps_preserve_time(self):
        common = self.adapt([frame(), frame(), frame()], [0, 2, 5], fps=10)
        np.testing.assert_allclose(common.timestamps, [0, .2, .5])

    def test_explicit_irregular_timestamps(self):
        common = self.adapt([frame(), frame()], [10, 12], fps=None, timestamps=[.1, .8])
        np.testing.assert_array_equal(common.timestamps, [.1, .8])

    def test_invalid_time(self):
        options = [{"fps": None}, {"fps": True}, {"fps": 0}, {"fps": float("nan")},
                   {"fps": 10**1000}, {"timestamps": [False, 1]},
                   {"timestamps": [1, 0]}, {"timestamps": [0, 0]},
                   {"timestamps": [0, float("inf")]}, {"timestamps": [False, True]},
                   {"fps": 1e-320}, {"timestamps": [-1, 1]}]
        for option in options:
            with self.subTest(option=option), self.assertRaises(KeypointError):
                self.adapt([frame(), frame()], [0, 1], **option)

    def test_invalid_frame_indices(self):
        for indices in ([0], [1, 0], [0, 0], [False, True], [False, 1], [0., 1.], [-1, 1]):
            with self.subTest(indices=indices), self.assertRaises(KeypointError):
                self.adapt([frame(), frame()], indices)

    def test_video_name_and_threshold_validation(self):
        with self.assertRaisesRegex(KeypointError, "^video_id$"):
            adapt_crowd_sequence([frame()], [0], video_id="different.mp4", source=SPEC, fps=30)
        for threshold in (True, float("nan"), float("inf"), 10**1000, -.1):
            with self.subTest(threshold=str(threshold)[:30]), self.assertRaises(KeypointError):
                self.adapt(confidence_threshold=threshold)

    def test_ragged_inputs_sanitized(self):
        with self.assertRaisesRegex(KeypointError, "^frame_indices$"):
            self.adapt([frame(), frame()], [0, [1, 2]])
        with self.assertRaisesRegex(KeypointError, "^timestamps$"):
            self.adapt([frame(), frame()], [0, 1], timestamps=[0, [1, 2]])

    def test_empty_and_frame_count(self):
        with self.assertRaisesRegex(KeypointError, "^empty_sequence$"):
            self.adapt([], [])
        with self.assertRaisesRegex(KeypointError, "^frame_count$"):
            self.adapt(expected_frame_count=2)

    def test_frame_limit(self):
        with self.assertRaisesRegex(KeypointError, "^frame_limit$"):
            self.adapt([frame(), frame()], [0, 1], max_frames=1)

    def test_ctc53_requires_explicit_policy(self):
        common = self.adapt()
        with self.assertRaisesRegex(KeypointError, "^inner_eyes_unavailable$"):
            to_ctc53(common)
        result = to_ctc53(common, allow_missing_inner_eyes=True)
        self.assertEqual(result["xy"].shape, (1, 53, 2))
        self.assertFalse(result["mask"][:, 1:3].any())
        self.assertEqual(np.count_nonzero(result["xy"][:, 1:3]), 0)
        np.testing.assert_allclose(result["xy"][0, 5, 0], 5 / 160)
        np.testing.assert_allclose(result["xy"][0, 6, 0], 2 / 160)

    def test_mediapipe_common_roundtrip(self):
        xy = np.arange(106, dtype=np.float32).reshape(1, 53, 2) / 106
        mask = np.ones((1, 53), bool)
        before = xy.copy()
        common = adapt_mediapipe53(xy, mask, timestamps=[0])
        np.testing.assert_array_equal(common.xy[0, :9], xy[0, [0, 3, 4, 5, 6, 7, 8, 9, 10]])
        result = to_ctc53(common, allow_missing_inner_eyes=True)
        np.testing.assert_array_equal(result["xy"][:, 3:], xy[:, 3:])
        np.testing.assert_array_equal(before, xy)

    def test_mediapipe_nonfinite_and_mask(self):
        xy = np.zeros((1, 53, 2))
        mask = np.ones((1, 53), bool)
        for value in (float("nan"), float("inf"), 1e300):
            xy[0, 0, 0] = value
            with self.assertRaises(KeypointError):
                adapt_mediapipe53(xy, mask, timestamps=[0])
        with self.assertRaisesRegex(KeypointError, "^mask_shape$"):
            adapt_mediapipe53(np.zeros((1, 53, 2)), mask.astype(int), timestamps=[0])

    def test_label_missing_blocks_before_read(self):
        class Sequence:
            video_id = VIDEO
        with self.assertRaisesRegex(KeypointError, "^label_missing$"):
            adapt_labeled_sequence(None, Sequence(), label_lookup=lambda video: None)

    def test_exact_video_label_and_source(self):
        class Sequence:
            video_id = VIDEO
            frame_indices = [0]
        class Archive:
            def read_frames(self, sequence):
                return iter([frame()])
        lookups = []
        def lookup(video):
            lookups.append(video)
            return "synthetic"
        common, text = adapt_labeled_sequence(Archive(), Sequence(), label_lookup=lookup,
                                              source=SPEC, fps=30)
        self.assertEqual(lookups, [VIDEO])
        self.assertEqual(text, "synthetic")
        self.assertEqual(common.video_id, VIDEO)

    def test_reproducible_and_source_unchanged(self):
        source = frame()
        before = source.pose.copy()
        a, b = self.adapt([source]), self.adapt([source])
        np.testing.assert_array_equal(a.xy, b.xy)
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_array_equal(source.pose, before)

    def test_long_sequence(self):
        source = frame()
        common = self.adapt((source for _ in range(10000)), np.arange(10000))
        self.assertEqual(common.xy.shape, (10000, 51, 2))
        self.assertAlmostEqual(common.timestamps[-1], 9999 / 30)


if __name__ == "__main__":
    unittest.main()
