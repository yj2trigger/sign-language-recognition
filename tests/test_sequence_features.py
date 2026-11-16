import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.sequence_features import build_sequence_features


class FeatureTests(unittest.TestCase):
    def test_velocity_fps_and_layout(self):
        xy, mask = np.zeros((3, 53, 2)), np.ones((3, 53), bool)
        xy[:, 0] = [[0, 0], [1, 2], [2, 4]]
        for fps in [15, 30]:
            result = build_sequence_features(xy, mask, fps=fps)
            self.assertEqual(result.values.shape, (3, 424))
            self.assertEqual(result.values.dtype, np.float32)
            self.assertEqual(result.channels, ("x", "y", "observed", "interpolated", "available", "vx", "vy", "motion_valid"))
            np.testing.assert_allclose(result.values[:, 5:8], [[0, 0, 0], [fps, fps * 2, 1], [fps, fps * 2, 1]])

    def test_irregular_time_and_gaps(self):
        xy, mask = np.zeros((4, 53, 2)), np.ones((4, 53), bool)
        xy[:, 0, 0] = [0, 1, 5, 7]
        mask[2, 1] = False
        result = build_sequence_features(xy, mask, fps=15, timestamps=[0, .5, 1.5, 2])
        np.testing.assert_allclose(result.values[:, 5], [0, 2, 4, 4])
        view = result.values.reshape(4, 53, 8)
        self.assertFalse(view[2:, 1, 5:].any())

    def test_no_timing_for_delta_and_no_motion(self):
        xy, mask = np.zeros((2, 53, 2)), np.ones((2, 53), bool)
        xy[1, 0] = [3, 4]
        np.testing.assert_array_equal(build_sequence_features(xy, mask, motion="delta").values[1, 5:8], [3, 4, 1])
        self.assertEqual(build_sequence_features(xy, mask, motion="none").values.shape, (2, 265))
        with self.assertRaises(ValueError):
            build_sequence_features(xy, mask)

    def test_single_and_motion_overflow(self):
        xy, mask = np.zeros((1, 53, 2)), np.ones((1, 53), bool)
        self.assertFalse(build_sequence_features(xy, mask, fps=60).values[:, 5:8].any())
        xy = np.zeros((2, 53, 2))
        mask = np.ones((2, 53), bool)
        xy[1] = 1
        with self.assertRaises(ValueError):
            build_sequence_features(xy, mask, timestamps=[0, 1e-100])

    def test_extreme_times_constant_coordinates_and_nan_prevention(self):
        xy, mask = np.ones((3, 53, 2)), np.ones((3, 53), bool)
        for motion in ["none", "delta", "velocity"]:
            with self.subTest(motion=motion), self.assertRaises(ValueError):
                build_sequence_features(xy, mask, motion=motion, normalization="center", aspect=1e-320, fps=30)
        # Very large but representable time intervals and constant position: v=0.
        result = build_sequence_features(xy, mask, timestamps=[0, 1e100, 2e100])
        self.assertTrue(np.isfinite(result.values).all())
        view = result.values.reshape(3, 53, 8)
        self.assertFalse(view[..., 5:7].any())
        self.assertTrue(view[1:, :, 7].all())
        xy[1:] = 2
        with np.errstate(all="raise"), self.assertRaisesRegex(ValueError, "motion overflow"):
            build_sequence_features(xy, mask, timestamps=[0, 1e-320, 2e-320])


if __name__ == "__main__":
    unittest.main()
