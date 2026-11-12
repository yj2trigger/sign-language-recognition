"""Missingness-preserving preprocessing without temporal resampling."""
from dataclasses import dataclass

import numpy as np

from .sequence_validation import SequenceError, _positive_scalar, sequence_times, validate_sequence


@dataclass(frozen=True)
class PreparedSequence:
    xy: np.ndarray
    observed: np.ndarray
    interpolated: np.ndarray
    available: np.ndarray
    normalization_valid: np.ndarray
    missing_rate: float
    all_missing: bool


def missing_runs(valid):
    """Return half-open missing runs in a one-dimensional boolean mask."""
    valid = np.asarray(valid)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError("valid must be a one-dimensional boolean array")
    edges = np.diff(np.r_[False, ~valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def prepare_sequence(xy, mask, *, normalization="none", max_gap=0, fps=None,
                     timestamps=None, aspect=16 / 9, epsilon=1e-6,
                     video_id=None, coordinate_bounds=None, invalid_policy="raise",
                     allow_all_missing=False):
    """Interpolate bounded short gaps, then normalize using observed joints only.

    Modes: none (image coordinates), center (observed centroid), shoulder
    (observed shoulder midpoint/width, indices 5 and 6). Shoulder distances use
    image-width units (y/aspect). Missing reference frames stay unavailable;
    no future/previous reference is silently borrowed.
    """
    if normalization not in ("none", "center", "shoulder"):
        raise SequenceError("unknown normalization", video_id)
    if isinstance(max_gap, (bool, np.bool_)) or not isinstance(max_gap, (int, np.integer)) or max_gap < 0:
        raise SequenceError("max_gap must be a nonnegative integer", video_id)
    try:
        aspect = _positive_scalar(aspect, "aspect")
        epsilon = _positive_scalar(epsilon, "epsilon")
    except ValueError as exc:
        raise SequenceError(str(exc), video_id) from exc
    report = validate_sequence(xy, mask, video_id=video_id,
                               coordinate_bounds=coordinate_bounds,
                               invalid_policy=invalid_policy, allow_all_missing=allow_all_missing)
    observed = report.observed
    out = np.zeros(np.shape(xy), dtype=np.float32)
    out[observed] = np.asarray(xy)[observed]
    interpolated = np.zeros_like(observed)
    times = None
    if max_gap or fps is not None or timestamps is not None:
        try:
            times = sequence_times(len(out), fps=fps, timestamps=timestamps)
        except ValueError as exc:
            raise SequenceError(str(exc), video_id) from exc
    if max_gap:
        for joint in range(53):
            for start, stop in missing_runs(observed[:, joint]):
                if start == 0 or stop == len(out) or stop - start > max_gap:
                    continue
                weight = ((times[start:stop] - times[start - 1]) /
                          (times[stop] - times[start - 1]))[:, None]
                out[start:stop, joint] = ((1 - weight) * out[start - 1, joint] +
                                           weight * out[stop, joint])
                interpolated[start:stop, joint] = True
    available = observed | interpolated
    norm_valid = np.ones(len(out), dtype=bool)
    if normalization != "none":
        work = out.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            work[..., 1] /= aspect
        if not np.isfinite(work).all():
            raise SequenceError("aspect conversion overflow", video_id)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if normalization == "center":
                count = observed.sum(axis=1)
                norm_valid = count > 0
                center = np.where(observed[..., None], work, 0).sum(axis=1) / np.maximum(count, 1)[:, None]
                scale = np.ones(len(out))
            else:
                center = (work[:, 5] + work[:, 6]) / 2
                difference = work[:, 5] - work[:, 6]
                scale = np.hypot(difference[:, 0], difference[:, 1])
                norm_valid = observed[:, 5] & observed[:, 6] & (scale > epsilon)
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise SequenceError("normalization reference overflow", video_id)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            work = (work - center[:, None]) / np.where(norm_valid, scale, 1)[:, None, None]
        available = available & norm_valid[:, None]
        work[~available] = 0
        if not np.isfinite(work).all() or (np.abs(work) > np.finfo(np.float32).max).any():
            raise SequenceError("normalization overflow", video_id)
        out = work.astype(np.float32)
    out[~available] = 0
    return PreparedSequence(out, observed, interpolated, available, norm_valid,
                            report.missing_rate, report.all_missing)
