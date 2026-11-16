import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_batching import (collate_ctc_batch, ctc_minimum_length,
    iter_ctc_batches, iter_ctc_samples, prepare_ctc_sample, transform_input_lengths, validate_targets)


def sample(length, targets=(1,), **options):
    return prepare_ctc_sample(str(length), np.full((length, 53, 2), .5, np.float32),
                              np.ones((length, 53), bool), targets, blank_id=0,
                              fps=30, **options)


class BatchingTests(unittest.TestCase):
    def test_padding_lengths_targets_and_signers(self):
        samples = [sample(2, (1, 2)), sample(5, (3, 3), signer_id="confirmed")]
        originals = [s.features.values.copy() for s in samples]
        batch = collate_ctc_batch(samples, blank_id=0)
        self.assertEqual(batch.features.shape, (2, 5, 424))
        np.testing.assert_array_equal(batch.input_lengths, [2, 5])
        np.testing.assert_array_equal(batch.padding_mask, [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
        self.assertFalse(batch.features[0, 2:].any())
        np.testing.assert_array_equal(batch.targets, [1, 2, 3, 3])
        np.testing.assert_array_equal(batch.target_lengths, [2, 2])
        self.assertIsNone(batch.output_lengths)
        np.testing.assert_array_equal(batch.signer_verified, [False, True])
        for s, before in zip(samples, originals):
            np.testing.assert_array_equal(s.features.values, before)

    def test_batch_one_empty_targets(self):
        batch = collate_ctc_batch([sample(1, ())], blank_id=0)
        self.assertEqual(batch.targets.dtype, np.int64)
        self.assertEqual(batch.targets.size, 0)
        self.assertEqual(batch.features.shape, (1, 1, 424))

    def test_bad_targets(self):
        for ids in [[0], [-1], [1.5], [True], [True, 2], [[1]], [np.nan], [2**64 - 1], [5]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_targets(ids, blank_id=0, num_classes=5)
        np.testing.assert_array_equal(validate_targets([0, 1], blank_id=4, num_classes=5), [0, 1])

    def test_ctc_repeat_condition_and_downsampling(self):
        self.assertEqual(ctc_minimum_length([1, 1, 2, 2, 2]), 8)
        with self.assertRaises(ValueError):
            sample(2, (1, 1))
        good = sample(5, (1, 1))
        with self.assertRaisesRegex(ValueError, "model time steps"):
            collate_ctc_batch([good], blank_id=0, length_transform=lambda n: n // 2)
        batch = collate_ctc_batch([good], blank_id=0, length_transform=lambda n: (n + 1) // 2)
        np.testing.assert_array_equal(batch.output_lengths, [3])

    def test_bad_transforms_and_mutation(self):
        lengths = np.array([3, 5], np.int64)
        for transform in [lambda n: [1.5, 2.5], lambda n: [0, 2], lambda n: [4, 6], lambda n: [1]]:
            with self.assertRaises(ValueError):
                transform_input_lengths(lengths, transform)
        def mutate(n):
            n[:] = 1
            return n
        transform_input_lengths(lengths, mutate)
        np.testing.assert_array_equal(lengths, [3, 5])

    def test_batch_guards(self):
        for options in [{"max_batch_size": 0}, {"max_batch_size": True},
                        {"max_padded_bytes": 2}, {"max_padded_bytes": -1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                collate_ctc_batch([sample(2)], blank_id=0, **options)
        with self.assertRaises(ValueError):
            collate_ctc_batch([], blank_id=0)
        with self.assertRaises(ValueError):
            collate_ctc_batch([sample(2), sample(3)], blank_id=0, max_batch_size=1)
        for size in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                list(iter_ctc_batches([], size, blank_id=0))

    def test_lazy_errors_and_final_batch(self):
        consumed, errors = [], []
        def records():
            for i in range(5):
                consumed.append(i)
                yield {"video_id": str(i), "xy": np.ones((3, 53, 2)),
                       "mask": np.ones((3, 53), bool), "target_ids": [0] if i == 1 else [1], "fps": 15}
        stream = iter_ctc_samples(records(), blank_id=0, on_error=lambda vid, err: errors.append((vid, err)))
        batches = iter_ctc_batches(stream, 3, blank_id=0)
        self.assertEqual(consumed, [])
        self.assertEqual(next(batches).features.shape[0], 3)
        self.assertEqual(consumed, [0, 1, 2, 3])
        self.assertEqual(len(list(batches)), 1)
        self.assertEqual(errors[0][0], "1")
        self.assertEqual(errors[0][1].video_id, "1")

    def test_raw_input_immutable_reproducible_and_long(self):
        xy, mask = np.full((10000, 53, 2), .5, np.float32), np.ones((10000, 53), bool)
        before = xy.copy(), mask.copy()
        a = prepare_ctc_sample("long", xy, mask, [1], blank_id=0, fps=15)
        b = prepare_ctc_sample("long", xy, mask, [1], blank_id=0, fps=15)
        np.testing.assert_array_equal(a.features.values, b.features.values)
        np.testing.assert_array_equal(xy, before[0])
        np.testing.assert_array_equal(mask, before[1])
        self.assertEqual(a.features.values.nbytes, 10000 * 424 * 4)
        self.assertFalse(np.shares_memory(a.features.values, xy))
        with self.assertRaises(ValueError):
            collate_ctc_batch([a, sample(1)], blank_id=0, max_padded_bytes=100)

    def test_all_missing_training_sample_rejected(self):
        with self.assertRaisesRegex(ValueError, "video_id='missing'"):
            prepare_ctc_sample("missing", np.zeros((2, 53, 2)), np.zeros((2, 53), bool),
                               [1], blank_id=0, fps=15, allow_all_missing=True)

    def test_synthetic_npz_both_storage_contracts(self):
        # Match build_dataset's clip/full metadata without reading real videos.
        xy = np.full((3, 53, 2), .5, np.float32)
        xy[:, 5], xy[:, 6] = (.6, .5), (.4, .5)
        mask = np.ones((3, 53), bool)
        import preprocess
        legacy = preprocess.normalize(xy, mask)
        for full in (False, True):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "synthetic.npz"
                extra = {"t": np.array([0, .04, .1]), "src_fps": 50., "full": True} if full else {}
                np.savez(path, xy=xy, mask=mask, fps=15., signer=123, **extra)
                with np.load(path, allow_pickle=False) as data:
                    record = {"video_id": "synthetic", "xy": data["xy"], "mask": data["mask"],
                              "target_ids": [1], "fps": float(data["fps"])}
                    if "t" in data:
                        record["timestamps"] = data["t"]
                    result = next(iter_ctc_samples([record], blank_id=0, normalization="shoulder", aspect=1))
                np.testing.assert_allclose(result.features.prepared.xy, legacy, atol=1e-6)
                self.assertEqual(result.input_length, 3)
                self.assertIsNone(result.signer_id)  # Stored signer is not automatically trusted.

    def test_missing_record_field_and_incompatible_channels(self):
        errors = []
        self.assertEqual(list(iter_ctc_samples([{"video_id": "no-xy"}], blank_id=0,
                         on_error=lambda vid, error: errors.append(error))), [])
        self.assertEqual(errors[0].video_id, "no-xy")
        with self.assertRaises(ValueError):
            list(iter_ctc_samples([{"video_id": "no-xy"}], blank_id=0))
        a, b = sample(3), sample(3, motion="delta")
        with self.assertRaises(ValueError):
            collate_ctc_batch([a, b], blank_id=0)

    def test_preprocessing_masks_and_targets_unchanged(self):
        xy, mask = np.full((5, 53, 2), .5), np.ones((5, 53), bool)
        targets = np.array([1, 2], dtype=np.int64)
        mask[2, 0] = False
        xy[2, 0] = np.nan
        before = xy.copy(), mask.copy(), targets.copy()
        result = prepare_ctc_sample("gaps", xy, mask, targets, blank_id=0,
                                    fps=15, max_gap=1, invalid_policy="mask")
        batch = collate_ctc_batch([result], blank_id=0, length_transform=lambda n: n)
        self.assertTrue(result.features.prepared.interpolated[2, 0])
        self.assertFalse(result.features.prepared.observed[2, 0])
        batch.features[:] = 0
        batch.targets[:] = 9
        np.testing.assert_array_equal(xy, before[0])
        np.testing.assert_array_equal(mask, before[1])
        np.testing.assert_array_equal(targets, before[2])
        np.testing.assert_array_equal(result.target_ids, [1, 2])

    def test_repeat_minimum_exact_boundary_and_uint64_targets(self):
        # 20 identical labels need 20 emissions plus 19 blank separators = 39.
        repeated = np.full(20, 7, np.int64)
        self.assertEqual(ctc_minimum_length(repeated), 39)
        exact = sample(39, repeated)
        batch = collate_ctc_batch([exact], blank_id=0, length_transform=lambda n: n)
        np.testing.assert_array_equal(batch.output_lengths, [39])
        with self.assertRaises(ValueError):
            sample(38, repeated)
        with self.assertRaises(ValueError):
            collate_ctc_batch([sample(78, repeated)], blank_id=0, length_transform=lambda n: n // 2 - 1)
        with self.assertRaises(ValueError):
            validate_targets(np.array([2**63], np.uint64), blank_id=0)
        np.testing.assert_array_equal(validate_targets(np.array([2**63 - 1], np.uint64), blank_id=0), [2**63 - 1])

    def test_large_padding_allocation_exact_limit_and_alias(self):
        long, short = sample(30000), sample(1)
        # B=2, T=30000, F=424 float32 plus one bool per frame.
        required = 2 * 30000 * (424 * 4 + 1)
        with self.assertRaisesRegex(ValueError, str(required)):
            collate_ctc_batch([long, short], blank_id=0, max_padded_bytes=required - 1)
        batch = collate_ctc_batch([long, short], blank_id=0, max_padded_bytes=required)
        self.assertEqual(batch.features.nbytes + batch.padding_mask.nbytes, required)
        np.testing.assert_array_equal(batch.input_lengths, [30000, 1])
        self.assertEqual(int(batch.padding_mask.sum()), 30001)
        self.assertFalse(batch.features[1, 1:].any())
        batch.features[:] = 0
        self.assertEqual(float(long.features.values[0, 0]), .5)
        self.assertEqual(float(short.features.values[0, 0]), .5)

    def test_numerical_errors_are_tracked_per_record(self):
        records = [{"video_id": name, "xy": np.ones((3, 53, 2)), "mask": np.ones((3, 53), bool),
                    "target_ids": [1], "fps": fps} for name, fps in [("bad-fps", 1e-320), ("good", 30)]]
        errors = []
        result = list(iter_ctc_samples(records, blank_id=0, on_error=lambda vid, err: errors.append((vid, err))))
        self.assertEqual([s.video_id for s in result], ["good"])
        self.assertEqual(errors[0][0], "bad-fps")
        self.assertEqual(errors[0][1].video_id, "bad-fps")


if __name__ == "__main__":
    unittest.main()
