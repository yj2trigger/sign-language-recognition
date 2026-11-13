"""Validation for raw landmark sequences; no MediaPipe or token-table imports."""
from dataclasses import dataclass

import numpy as np


class SequenceError(ValueError):
    """A data error that retains the caller-supplied video identifier."""

    def __init__(self, message, video_id=None):
        self.video_id = video_id
        super().__init__(f"video_id={video_id!r}: {message}")


@dataclass(frozen=True)
class ValidationReport:
    observed: np.ndarray
    invalid_coordinates: np.ndarray
    missing_rate: float
    all_missing: bool


def validate_sequence(xy, mask, *, video_id=None, coordinate_bounds=None,
                      invalid_policy="raise", allow_all_missing=False):
    """Check (T,53,2) real coordinates and a strict boolean (T,53) mask.

    Nonfinite values are invalid even in masked slots. ``mask`` policy sanitizes
    them downstream instead of rejecting. Optional inclusive bounds are a data
    policy: MediaPipe can legitimately return off-screen coordinates.
    """
    def fail(message):
        raise SequenceError(message, video_id)

    if invalid_policy not in ("raise", "mask"):
        fail("invalid_policy must be 'raise' or 'mask'")
    xy, mask = np.asarray(xy), np.asarray(mask)
    if xy.ndim != 3 or xy.shape[1:] != (53, 2):
        fail("xy must have shape (T,53,2)")
    if not len(xy):
        fail("empty sequence")
    if mask.shape != xy.shape[:2] or mask.dtype != np.bool_:
        fail("mask must be boolean with shape (T,53)")
    if xy.dtype.kind not in "fiu":
        fail("xy must contain real numeric coordinates")
    # All outputs use float32. Reject finite but unrepresentable magnitudes too.
    invalid = ~np.isfinite(xy).all(axis=-1)
    limit = np.finfo(np.float32).max
    invalid |= ((xy < -limit) | (xy > limit)).any(axis=-1)
    if coordinate_bounds is not None:
        bounds = np.asarray(coordinate_bounds, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            fail("coordinate_bounds must be two finite increasing values")
        invalid |= ((xy < bounds[0]) | (xy > bounds[1])).any(axis=-1)
    if invalid.any() and invalid_policy == "raise":
        fail(f"invalid coordinates at {int(invalid.sum())} joint/frame slots")
    observed = mask & ~invalid
    all_missing = not observed.any()
    if all_missing and not allow_all_missing:
        fail("all joints missing")
    return ValidationReport(observed, invalid, float(1 - observed.mean()), all_missing)


def _positive_scalar(value, name):
    """Convert a real scalar to finite positive float64 without silent coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def sequence_times(length, *, fps=None, timestamps=None):
    """Seconds with finite positive intervals and representable total duration."""
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)) or length <= 0:
        raise ValueError("length must be a positive integer")
    if fps is not None:
        fps = _positive_scalar(fps, "fps")
    if timestamps is not None:
        raw = np.asarray(timestamps)
        if raw.shape != (length,) or raw.dtype.kind not in "fiu":
            raise ValueError("timestamps must be real numeric values with shape (T,)")
        with np.errstate(over="ignore", invalid="ignore"):
            times = raw.astype(np.float64, copy=True)
    else:
        if fps is None:
            raise ValueError("provide fps or timestamps explicitly")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            times = np.arange(length, dtype=np.float64) / fps
    with np.errstate(over="ignore", invalid="ignore"):
        intervals = np.diff(times)
        duration = times[-1] - times[0]
    if (not np.isfinite(times).all() or not np.isfinite(intervals).all() or
            not np.isfinite(duration) or (intervals <= 0).any()):
        raise ValueError("times and duration must be finite with strictly increasing timestamps")
    return times
