"""Synthetic raw-input contracts. Run with unittest; no extra dependencies."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.sequence_validation import SequenceError, sequence_times, validate_sequence


class ValidationTests(unittest.TestCase):
    def test_normal_and_partial(self):
        xy = np.full((4, 53, 2), .5, np.float32)
        mask = np.ones((4, 53), bool)
        mask[0, 0] = False
        report = validate_sequence(xy, mask)
        self.assertAlmostEqual(report.missing_rate, 1 / 212)
        self.assertFalse(report.observed[0, 0])
        np.testing.assert_array_equal(xy, .5)

    def test_shapes_types_empty(self):
        for xy, mask in [(np.zeros((3, 53)), np.ones((3, 53), bool)),
                         (np.zeros((3, 52, 2)), np.ones((3, 52), bool)),
                         (np.zeros((3, 53, 3)), np.ones((3, 53), bool)),
                         (np.zeros((3, 53, 2)), np.ones((2, 53), bool)),
                         (np.zeros((3, 53, 2)), np.ones((3, 53))),
                         (np.zeros((0, 53, 2)), np.ones((0, 53), bool)),
                         (np.zeros((1, 53, 2), complex), np.ones((1, 53), bool))]:
            with self.subTest(shape=xy.shape, dtype=xy.dtype):
                with self.assertRaises(SequenceError):
                    validate_sequence(xy, mask)

    def test_invalid_coordinates(self):
        for value in [np.nan, np.inf, -np.inf, 1e100]:
            for observed in [True, False]:
                xy = np.zeros((2, 53, 2), np.float64)
                mask = np.ones((2, 53), bool)
                xy[0, 0, 0], mask[0, 0] = value, observed
                with self.assertRaisesRegex(SequenceError, "video_id='bad'"):
                    validate_sequence(xy, mask, video_id="bad")
                report = validate_sequence(xy, mask, invalid_policy="mask")
                self.assertFalse(report.observed[0, 0])
                self.assertTrue(report.invalid_coordinates[0, 0])

    def test_offscreen_policy(self):
        xy, mask = np.full((1, 53, 2), 1.2), np.ones((1, 53), bool)
        self.assertFalse(validate_sequence(xy, mask).all_missing)
        with self.assertRaises(SequenceError):
            validate_sequence(xy, mask, coordinate_bounds=(0, 1))
        self.assertTrue(validate_sequence(xy, mask, coordinate_bounds=(0, 1),
                        invalid_policy="mask", allow_all_missing=True).all_missing)

    def test_all_missing_and_single_frame(self):
        xy, mask = np.zeros((1, 53, 2)), np.zeros((1, 53), bool)
        with self.assertRaises(SequenceError):
            validate_sequence(xy, mask)
        self.assertEqual(validate_sequence(xy, mask, allow_all_missing=True).missing_rate, 1)
        mask[0, 52] = True
        self.assertFalse(validate_sequence(xy, mask).all_missing)

    def test_time_validation(self):
        np.testing.assert_allclose(sequence_times(3, fps=2), [0, .5, 1])
        np.testing.assert_allclose(sequence_times(3, fps=15, timestamps=[4, 4.1, 4.3]), [4, 4.1, 4.3])
        for options in [{}, {"fps": 0}, {"fps": -1}, {"fps": True}, {"fps": np.nan},
                        {"timestamps": [0, 0, 1]}, {"timestamps": [0, 2, 1]},
                        {"timestamps": [0, np.inf, 2]}, {"timestamps": [0]}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                sequence_times(3, **options)

    def test_time_overflow_and_numeric_types(self):
        cases = [{"fps": 1e-320}, {"timestamps": [-1e308, 0, 1e308]},
                 {"timestamps": ["0", "1", "2"]}, {"timestamps": [False, True, True]},
                 {"timestamps": [0j, 1j, 2j]}, {"fps": 1j}, {"fps": "30"}]
        with np.errstate(all="raise"):
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    sequence_times(3, **options)
            with self.assertRaises(ValueError):
                sequence_times(2, timestamps=[-1e308, 1e308])
        np.testing.assert_array_equal(sequence_times(1, fps=1e-320), [0])
        for length in [0, -1, 1.5, True]:
            with self.subTest(length=length), self.assertRaises(ValueError):
                sequence_times(length, fps=30)

    def test_abnormal_masks_and_integer_coordinates(self):
        xy = np.ones((2, 53, 2), dtype=np.int64)
        self.assertTrue(validate_sequence(xy, np.ones((2, 53), bool)).observed.all())
        for mask in [np.ones((2, 53), int), np.full((2, 53), np.nan),
                     np.full((2, 53), "True", dtype=object)]:
            with self.subTest(dtype=mask.dtype), self.assertRaises(SequenceError):
                validate_sequence(xy, mask)


if __name__ == "__main__":
    unittest.main()
