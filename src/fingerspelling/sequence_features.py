"""Float32 frame features with explicit timing and provenance channels."""
from dataclasses import dataclass

import numpy as np

from .sequence_preprocessing import prepare_sequence
from .sequence_validation import SequenceError, sequence_times


@dataclass(frozen=True)
class FeatureSequence:
    values: np.ndarray
    channels: tuple
    prepared: object


def build_sequence_features(xy, mask, *, fps=None, timestamps=None,
                            motion="velocity", **preprocessing):
    """Per joint: x,y,observed,interpolated,available[,dx,dy,motion_valid].

    Flatten joint-major to (T,265) or (T,424). Velocity is backward difference
    per second; delta is per frame. Motion requires both endpoints available,
    so it never bridges an unresolved gap. The first motion frame is zero.
    """
    if motion not in ("none", "delta", "velocity"):
        raise ValueError("motion must be none, delta or velocity")
    prepared = prepare_sequence(xy, mask, fps=fps, timestamps=timestamps, **preprocessing)
    times = None
    if motion == "velocity" or fps is not None or timestamps is not None:
        times = sequence_times(len(prepared.xy), fps=fps, timestamps=timestamps)
    channels = ("x", "y", "observed", "interpolated", "available")
    if motion != "none":
        channels += ("vx" if motion == "velocity" else "dx",
                     "vy" if motion == "velocity" else "dy", "motion_valid")
    features = np.zeros((*prepared.observed.shape, len(channels)), dtype=np.float32)
    features[..., :2] = prepared.xy
    features[..., 2] = prepared.observed
    features[..., 3] = prepared.interpolated
    features[..., 4] = prepared.available
    if motion != "none" and len(features) > 1:
        valid = prepared.available[1:] & prepared.available[:-1]
        change = prepared.xy[1:].astype(np.float64) - prepared.xy[:-1]
        if motion == "velocity":
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                change /= np.diff(times)[:, None, None]
        change[~valid] = 0
        if not np.isfinite(change).all() or (np.abs(change) > np.finfo(np.float32).max).any():
            raise SequenceError("motion overflow", preprocessing.get("video_id"))
        features[1:, :, 5:7] = change
        features[1:, :, 7] = valid
    return FeatureSequence(features.reshape(len(features), -1), channels, prepared)
