"""External-target CTC samples, lazy preparation and batch-local padding."""
from dataclasses import dataclass

import numpy as np

from .sequence_features import build_sequence_features
from .sequence_validation import SequenceError


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_targets(target_ids, *, blank_id, num_classes=None):
    """Validate external token IDs; never construct vocabulary or insert blank."""
    blank_id = _integer(blank_id, "blank_id")
    if num_classes is not None:
        num_classes = _integer(num_classes, "num_classes", 1)
        if blank_id >= num_classes:
            raise ValueError("blank_id outside vocabulary")
    if isinstance(target_ids, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in target_ids):
        raise ValueError("target_ids must not contain booleans")
    ids = np.asarray(target_ids)
    if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
        raise ValueError("target_ids must be a one-dimensional integer sequence")
    if ids.size and ((ids < 0).any() or (ids > np.iinfo(np.int64).max).any()):
        raise ValueError("target_ids outside int64 range")
    ids = ids.astype(np.int64, copy=True)
    if (ids == blank_id).any() or (num_classes is not None and (ids >= num_classes).any()):
        raise ValueError("target_ids contain blank or out-of-vocabulary IDs")
    return ids


def ctc_minimum_length(target_ids):
    """Targets plus one separating step for every adjacent repeated label."""
    ids = np.asarray(target_ids)
    return len(ids) + int(np.count_nonzero(ids[1:] == ids[:-1]))


def transform_input_lengths(input_lengths, length_transform):
    """Invoke an explicit model-supplied transform on an isolated int64 copy."""
    lengths = np.asarray(input_lengths)
    if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or (lengths <= 0).any():
        raise ValueError("input_lengths must be positive integers")
    if (lengths > np.iinfo(np.int64).max).any():
        raise ValueError("input_lengths exceed int64")
    output = np.asarray(length_transform(lengths.astype(np.int64, copy=True)))
    if output.shape != lengths.shape or output.dtype.kind not in "iu":
        raise ValueError("length_transform must return integer lengths of the same shape")
    if (output <= 0).any() or (output > lengths).any():
        raise ValueError("downsampled lengths must lie between 1 and input length")
    return output.astype(np.int64, copy=True)


@dataclass(frozen=True)
class CTCSample:
    video_id: str
    features: object
    target_ids: np.ndarray
    signer_id: object = None

    @property
    def input_length(self):
        return len(self.features.values)

    @property
    def signer_verified(self):
        # Caller must supply only identifiers whose provenance was verified.
        return self.signer_id is not None


@dataclass(frozen=True)
class CTCBatch:
    features: np.ndarray
    padding_mask: np.ndarray
    input_lengths: np.ndarray
    output_lengths: object
    targets: np.ndarray
    target_lengths: np.ndarray
    video_ids: tuple
    signer_ids: tuple
    signer_verified: np.ndarray


def prepare_ctc_sample(video_id, xy, mask, target_ids, *, blank_id,
                       num_classes=None, signer_id=None, **feature_options):
    """One video at a time. Unknown signer remains None; filename is never parsed."""
    try:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be a nonempty string")
        if signer_id is not None and (isinstance(signer_id, (bool, np.bool_)) or
                                      not isinstance(signer_id, (str, int, np.integer)) or
                                      isinstance(signer_id, str) and not signer_id.strip()):
            raise ValueError("signer_id must be a confirmed string/integer or None")
        ids = validate_targets(target_ids, blank_id=blank_id, num_classes=num_classes)
        features = build_sequence_features(xy, mask, video_id=video_id, **feature_options)
        if not features.prepared.available.any():
            raise ValueError("no usable coordinates after preprocessing")
        if len(features.values) < ctc_minimum_length(ids):
            raise ValueError("insufficient CTC input length (including adjacent repeats)")
        return CTCSample(video_id, features, ids, signer_id)
    except SequenceError:
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise SequenceError(str(exc), video_id) from exc


def iter_ctc_samples(records, *, on_error=None, **options):
    """Lazy adapter for mapping records; errors are raised unless a callback is given.

    Callback receives (video_id, SequenceError); only that record is skipped.
    Records supply video_id,xy,mask,target_ids and optional signer_id,fps,timestamps.
    """
    for record in records:
        video_id = record.get("video_id")
        try:
            timing = {k: record[k] for k in ("fps", "timestamps") if k in record}
            sample_options = {**options, **timing}
            yield prepare_ctc_sample(video_id, record["xy"], record["mask"], record["target_ids"],
                                     signer_id=record.get("signer_id"), **sample_options)
        except (SequenceError, KeyError) as exc:
            error = exc if isinstance(exc, SequenceError) else SequenceError(f"missing field {exc}", video_id)
            if on_error is None:
                raise error
            on_error(video_id, error)


def collate_ctc_batch(samples, *, blank_id, num_classes=None, length_transform=None,
                      max_batch_size=None, max_padded_bytes=None):
    """Pad only to this batch's maximum T; True padding_mask means real frame.

    Return packed one-dimensional targets. Unknown model output lengths stay
    None; pass an explicit identity transform for a model without downsampling.
    Memory limit covers the output feature tensor and frame mask before allocation.
    """
    if max_batch_size is not None:
        _integer(max_batch_size, "max_batch_size", 1)
    if max_padded_bytes is not None:
        _integer(max_padded_bytes, "max_padded_bytes", 1)
    if not hasattr(samples, "__len__") or not len(samples):
        raise ValueError("samples must be a nonempty sized batch")
    if max_batch_size is not None and len(samples) > max_batch_size:
        raise ValueError("batch exceeds max_batch_size")
    lengths, targets = [], []
    channels = samples[0].features.channels
    width = len(channels) * 53
    for sample in samples:
        values = sample.features.values
        if (values.ndim != 2 or not len(values) or values.shape[1] != width or
                sample.features.channels != channels or values.dtype != np.float32 or
                not np.isfinite(values).all()):
            raise SequenceError("inconsistent, empty or invalid features", sample.video_id)
        try:
            ids = validate_targets(sample.target_ids, blank_id=blank_id, num_classes=num_classes)
        except ValueError as exc:
            raise SequenceError(str(exc), sample.video_id) from exc
        lengths.append(len(values))
        targets.append(ids)
    lengths = np.array(lengths, dtype=np.int64)
    output = None if length_transform is None else transform_input_lengths(lengths, length_transform)
    for sample, ids, length in zip(samples, targets, lengths if output is None else output):
        if length < ctc_minimum_length(ids):
            raise SequenceError("insufficient CTC model time steps", sample.video_id)
    batch_size, longest = len(samples), int(lengths.max())
    required_bytes = batch_size * longest * (width * 4 + 1)
    if max_padded_bytes is not None and required_bytes > max_padded_bytes:
        raise ValueError(f"padded allocation requires {required_bytes} bytes")
    features = np.zeros((batch_size, longest, width), dtype=np.float32)
    padding_mask = np.zeros((batch_size, longest), dtype=bool)
    for index, sample in enumerate(samples):
        features[index, :lengths[index]] = sample.features.values
        padding_mask[index, :lengths[index]] = True
    return CTCBatch(features, padding_mask, lengths, output, np.concatenate(targets),
                    np.array([len(t) for t in targets], dtype=np.int64),
                    tuple(s.video_id for s in samples), tuple(s.signer_id for s in samples),
                    np.array([s.signer_verified for s in samples], dtype=bool))


def iter_ctc_batches(samples, batch_size, **collate_options):
    """Bounded-memory batching; caller may bucket by length before this iterator."""
    batch_size = _integer(batch_size, "batch_size", 1)
    pending = []
    for sample in samples:
        pending.append(sample)
        if len(pending) == batch_size:
            yield collate_ctc_batch(pending, **collate_options)
            pending = []
    if pending:
        yield collate_ctc_batch(pending, **collate_options)
