import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.sequence_preprocessing import missing_runs, prepare_sequence


class PreprocessingTests(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(missing_runs(np.array([False, False, True, False, True, False])),
                         [(0, 2), (3, 4), (5, 6)])
        self.assertEqual(missing_runs(np.ones(5, bool)), [])

    def test_interpolation_time_provenance_and_unresolved(self):
        xy, mask = np.zeros((7, 53, 2)), np.ones((7, 53), bool)
        xy[0, 0] = (0, 2)
        xy[3, 0] = (8, 10)
        mask[1:3, 0] = False
        mask[:2, 1] = False
        mask[4:, 2] = False
        mask[1:5, 3] = False
        mask[:, 4] = False
        before = xy.copy(), mask.copy()
        result = prepare_sequence(xy, mask, max_gap=2, timestamps=[0, 1, 3, 4, 5, 6, 7])
        np.testing.assert_allclose(result.xy[1:3, 0], [[2, 4], [6, 8]])
        self.assertFalse(result.observed[1, 0])
        self.assertTrue(result.interpolated[1, 0])
        self.assertTrue(result.available[1, 0])
        for joint, sl in [(1, slice(0, 2)), (2, slice(4, None)), (3, slice(1, 5)), (4, slice(None))]:
            self.assertFalse(result.available[sl, joint].any())
            self.assertFalse(result.xy[sl, joint].any())
        np.testing.assert_array_equal(xy, before[0])
        np.testing.assert_array_equal(mask, before[1])

    def test_shoulder_geometry_aspect_and_translation_scale(self):
        xy, mask = np.full((3, 53, 2), .5), np.ones((3, 53), bool)
        xy[:, 5], xy[:, 6], xy[:, 11] = (.6, .5), (.4, .5), (.7, .9)
        result = prepare_sequence(xy, mask, normalization="shoulder", aspect=2)
        np.testing.assert_allclose(result.xy[:, 5], [[.5, 0]] * 3, atol=1e-6)
        np.testing.assert_allclose(result.xy[:, 11], [[1, 1]] * 3, atol=1e-6)
        moved = prepare_sequence(xy * 3 + 2, mask, normalization="shoulder", aspect=2)
        np.testing.assert_allclose(moved.xy, result.xy, atol=2e-6)

    def test_reference_missing_and_zero_width(self):
        xy, mask = np.ones((3, 53, 2)), np.ones((3, 53), bool)
        xy[2, 5, 0] = 2
        mask[1, 5] = False
        result = prepare_sequence(xy, mask, normalization="shoulder")
        np.testing.assert_array_equal(result.normalization_valid, [False, False, True])
        self.assertFalse(result.available[:2].any())
        self.assertTrue(result.observed[0].all())
        self.assertTrue(np.isfinite(result.xy).all())

    def test_centroid_ignores_missing(self):
        xy, mask = np.full((1, 53, 2), 100), np.zeros((1, 53), bool)
        xy[0, :2] = [(2, 2), (4, 6)]
        mask[0, :2] = True
        result = prepare_sequence(xy, mask, normalization="center", aspect=2)
        np.testing.assert_array_equal(result.xy[0, :2], [(-1, -1), (1, 1)])
        self.assertFalse(result.xy[0, 2:].any())

    def test_sanitize_nan_and_all_missing(self):
        xy, mask = np.full((1, 53, 2), np.nan), np.ones((1, 53), bool)
        result = prepare_sequence(xy, mask, invalid_policy="mask", allow_all_missing=True)
        self.assertTrue(result.all_missing)
        self.assertFalse(result.available.any())
        self.assertFalse(result.xy.any())

    def test_options(self):
        xy, mask = np.ones((1, 53, 2)), np.ones((1, 53), bool)
        for options in [{"max_gap": -1}, {"max_gap": True}, {"max_gap": 1.5},
                        {"normalization": "wrist"}, {"aspect": 0}, {"epsilon": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_sequence(xy, mask, **options)

    def test_overflowing_interpolation_span_rejected(self):
        xy, mask = np.zeros((3, 53, 2)), np.ones((3, 53), bool)
        xy[2, 0] = 1
        mask[1, 0] = False
        # Midpoint must be 0.5 if computed; rejecting unrepresentable span is valid.
        with np.errstate(all="raise"), self.assertRaisesRegex(ValueError, "video_id='span'"):
            prepare_sequence(xy, mask, timestamps=[-1e308, 0, 1e308], max_gap=1, video_id="span")

    def test_normalization_overflow_and_direct_timing_validation(self):
        xy, mask = np.ones((1, 53, 2)), np.ones((1, 53), bool)
        with np.errstate(all="raise"):
            for mode in ["center", "shoulder"]:
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "overflow"):
                    prepare_sequence(xy, mask, normalization=mode, aspect=1e-320)
            # Even with max_gap=0 the public preprocessing API must validate metadata.
            with self.assertRaises(ValueError):
                prepare_sequence(xy, mask, fps=0)
        for aspect in [True, "2", 1j]:
            with self.subTest(aspect=aspect), self.assertRaises(ValueError):
                prepare_sequence(xy, mask, aspect=aspect)

    def test_long_gap_and_readonly_strided_input(self):
        backing = np.full((60000, 53, 2), .25, np.float32)
        xy = backing[::2]
        mask_backing = np.ones((60000, 53), bool)
        mask = mask_backing[::2]
        mask[1:-1, 0] = False
        xy.flags.writeable = False
        mask.flags.writeable = False
        result = prepare_sequence(xy, mask, max_gap=100, fps=30)
        self.assertEqual(result.xy.shape, (30000, 53, 2))
        self.assertFalse(result.available[1:-1, 0].any())
        self.assertFalse(result.interpolated.any())
        result.xy[:] = 9
        np.testing.assert_array_equal(backing, .25)
        self.assertFalse(np.shares_memory(result.observed, mask_backing))
        self.assertEqual(int((~mask).sum()), 29998)


if __name__ == "__main__":
    unittest.main()
