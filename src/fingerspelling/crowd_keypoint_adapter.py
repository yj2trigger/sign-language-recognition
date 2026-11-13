"""Explicit geometry/timing adapters for the observed CROWD schema.

The common layout has 51 anatomical points. Existing pose indices 2/5 mean
LEFT_EYE/RIGHT_EYE. Their export-specific correspondence remains unverified.
The common51 policy omits both eye slots.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .crowd_keypoints import KeypointError, VIDEO, KeypointFrame

# Common order: nose, left/right ear, shoulder, elbow, wrist, left/right hand.
BODY25_COMMON = (0, 18, 17, 5, 2, 6, 3, 7, 4)
MEDIAPIPE53_COMMON = (0, 3, 4, 5, 6, 7, 8, 9, 10, *range(11, 53))
COMMON_NAMES = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                *(f"left_hand_{i}" for i in range(21)),
                *(f"right_hand_{i}" for i in range(21)))


@dataclass(frozen=True)
class SourceSpec:
    """Caller-supplied verified source metadata. No values are inferred.

    evidence is a private reference to the verification source. BODY_25/Hand21
    semantics and anatomical handedness must be verified for this export.
    coordinates supports pixels_top_left or normalized_top_left, x right/y down.
    image_size is the coordinate canvas width/height before fit_16x9 padding.
    """
    layout: str
    coordinates: str
    handedness: str
    image_size: tuple[int, int]
    evidence: str = field(repr=False)


def conversion_blockers(source):
    """Return fixed codes suitable for aggregate reporting."""
    if source is None:
        return ("source_spec_missing",)
    if not isinstance(source, SourceSpec):
        return ("source_spec_type",)
    reasons = []
    if not isinstance(source.layout, str) or source.layout != "openpose_body25_hand21":
        reasons.append("layout_unverified")
    if not isinstance(source.coordinates, str) or source.coordinates not in ("pixels_top_left", "normalized_top_left"):
        reasons.append("coordinate_system")
    if not isinstance(source.handedness, str) or source.handedness != "anatomical_unmirrored":
        reasons.append("handedness_unverified")
    if (not isinstance(source.image_size, tuple) or len(source.image_size) != 2
            or any(type(value) is not int or not 0 < value <= 1_000_000
                   for value in source.image_size)):
        reasons.append("image_size")
    if not isinstance(source.evidence, str) or not source.evidence.strip():
        reasons.append("metadata_evidence")
    return tuple(reasons)


def _positive(value, code):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise KeypointError(code)
    return value


def _indices_and_times(frame_indices, length, fps, timestamps):
    if isinstance(frame_indices, (list, tuple)) and any(
            isinstance(value, (bool, np.bool_)) for value in frame_indices):
        raise KeypointError("frame_indices")
    try:
        indices = np.asarray(frame_indices)
    except (TypeError, ValueError, OverflowError):
        raise KeypointError("frame_indices") from None
    if (indices.shape != (length,) or indices.dtype.kind not in "iu"
            or (indices < 0).any() or (indices > 2**53).any()
            or (indices[1:] <= indices[:-1]).any()):
        raise KeypointError("frame_indices")
    if fps is not None:
        _positive(fps, "fps")
    if timestamps is None:
        if fps is None:
            raise KeypointError("timing_missing")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            times = indices.astype(np.float64) / fps
    else:
        if isinstance(timestamps, (list, tuple)) and any(
                isinstance(value, (bool, np.bool_)) for value in timestamps):
            raise KeypointError("timestamps")
        try:
            raw = np.asarray(timestamps)
        except (TypeError, ValueError, OverflowError):
            raise KeypointError("timestamps") from None
        if raw.shape != (length,) or raw.dtype.kind not in "fiu":
            raise KeypointError("timestamps")
        times = raw.astype(np.float64, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = np.diff(times)
        span = times[-1] - times[0]
    if (not np.isfinite(times).all() or (times < 0).any()
            or not np.isfinite(deltas).all() or (deltas <= 0).any()
            or not np.isfinite(span)):
        raise KeypointError("timestamps")
    # Explicit timestamps are authoritative for irregular frame sampling.
    return indices.astype(np.int64, copy=True), times


@dataclass(frozen=True, repr=False)
class CommonLandmarks:
    xy: np.ndarray
    mask: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    video_id: str


def _check_common(common):
    if not isinstance(common, CommonLandmarks):
        raise KeypointError("common_type")
    xy, mask = common.xy, common.mask
    if (not isinstance(xy, np.ndarray) or xy.dtype != np.float32 or xy.ndim != 3
            or xy.shape[1:] != (51, 2) or not len(xy)
            or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
            or mask.shape != xy.shape[:2]):
        raise KeypointError("common_shape")
    if not np.isfinite(xy).all():
        raise KeypointError("nonfinite")
    _indices_and_times(common.frame_indices, len(xy), None, common.timestamps)


def adapt_crowd_sequence(frames, frame_indices, *, video_id, source=None, fps=None,
                         timestamps=None, confidence_threshold=0.5,
                         expected_frame_count=None, max_frames=100_000):
    """Create common51 normalized to the project's fit_16x9 canvas.

    Frame-index gaps preserve elapsed time through index/fps. No interpolation,
    resampling, automatic hand swapping or FPS default is applied. Positive
    confidence at or above the explicit threshold creates an observed slot.
    """
    reasons = conversion_blockers(source)
    if reasons:
        raise KeypointError(reasons[0])
    if not isinstance(video_id, str) or VIDEO.fullmatch(video_id) is None:
        raise KeypointError("video_id")
    if type(max_frames) is not int or max_frames < 1:
        raise KeypointError("invalid_limit")
    if (type(confidence_threshold) not in (float, int)
            or not 0 <= confidence_threshold <= 1):
        raise KeypointError("confidence_threshold")
    if expected_frame_count is not None and (
            type(expected_frame_count) is not int or expected_frame_count < 1):
        raise KeypointError("frame_count")
    width, height = source.image_size
    target_width, target_height = width, height
    pad_x = pad_y = 0
    if abs(width / height - 16 / 9) >= 1e-3:
        if width / height < 16 / 9:
            target_width = int(round(height * 16 / 9))
            pad_x = (target_width - width) // 2
        else:
            target_height = int(round(width / (16 / 9)))
            pad_y = (target_height - height) // 2
    xys, masks = [], []
    for frame in frames:
        if len(xys) >= max_frames:
            raise KeypointError("frame_limit")
        if not isinstance(frame, KeypointFrame):
            raise KeypointError("frame_type")
        xy = np.zeros((51, 2), np.float64)
        mask = np.zeros(51, bool)
        for points, count, source_indices, output in (
                (frame.pose, 25, BODY25_COMMON, slice(0, 9)),
                (frame.left_hand, 21, tuple(range(21)), slice(9, 30)),
                (frame.right_hand, 21, tuple(range(21)), slice(30, 51))):
            if (not isinstance(points, np.ndarray) or points.dtype.kind not in "fiu"
                    or points.shape not in ((0, 3), (count, 3))):
                raise KeypointError("joint_count")
            if not np.isfinite(points).all():
                raise KeypointError("nonfinite")
            if not len(points):
                continue
            if ((points[:, 2] < 0) | (points[:, 2] > 1)).any():
                raise KeypointError("confidence")
            selected = points[list(source_indices)]
            observed = (selected[:, 2] > 0) & (selected[:, 2] >= confidence_threshold)
            coords = selected[:, :2].astype(np.float64, copy=True)
            coords[~observed] = 0
            if source.coordinates == "normalized_top_left":
                coords *= (width, height)
            if ((coords[observed] < 0).any()
                    or (coords[observed] > (width, height)).any()):
                raise KeypointError("coordinate_bounds")
            coords = (coords + (pad_x, pad_y)) / (target_width, target_height)
            xy[output] = np.where(observed[:, None], coords, 0)
            mask[output] = observed
        xys.append(xy.astype(np.float32))
        masks.append(mask)
    if not xys:
        raise KeypointError("empty_sequence")
    if expected_frame_count is not None and len(xys) != expected_frame_count:
        raise KeypointError("frame_count")
    indices, times = _indices_and_times(frame_indices, len(xys), fps, timestamps)
    return CommonLandmarks(np.stack(xys), np.stack(masks), times, indices, video_id)


def adapt_mediapipe53(xy, mask, *, timestamps, video_id="", frame_indices=None):
    """Select common51 from existing fit_16x9 normalized MediaPipe arrays."""
    try:
        xy, mask = np.asarray(xy), np.asarray(mask)
    except (TypeError, ValueError, OverflowError):
        raise KeypointError("mediapipe_shape") from None
    if xy.ndim != 3 or xy.shape[1:] != (53, 2) or not len(xy) or xy.dtype.kind not in "fiu":
        raise KeypointError("mediapipe_shape")
    if mask.shape != xy.shape[:2] or mask.dtype != np.bool_:
        raise KeypointError("mask_shape")
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = xy.astype(np.float32)
    if not np.isfinite(coordinates).all():
        raise KeypointError("nonfinite")
    if frame_indices is None:
        frame_indices = np.arange(len(xy))
    indices, times = _indices_and_times(frame_indices, len(xy), None, timestamps)
    observed = mask[:, MEDIAPIPE53_COMMON].copy()
    coordinates = coordinates[:, MEDIAPIPE53_COMMON].copy()
    coordinates[~observed] = 0
    return CommonLandmarks(coordinates, observed, times, indices, video_id)


def to_ctc53(common, *, allow_missing_inner_eyes=False):
    """Return xy/mask/timestamps for PR #16 under an explicit missing-eye policy.

    Slots 1/2 are always unobserved zero sentinels. They correspond to MediaPipe
    LEFT_EYE/RIGHT_EYE (pose 2/5). The parameter/error names mentioning inner
    eyes are retained for PR #18 API compatibility. No eye mapping is applied.
    Training and inference must opt into this same missing-eye policy.
    """
    _check_common(common)
    if allow_missing_inner_eyes is not True:
        raise KeypointError("inner_eyes_unavailable")
    xy = np.zeros((len(common.xy), 53, 2), np.float32)
    mask = np.zeros((len(common.xy), 53), bool)
    xy[:, MEDIAPIPE53_COMMON] = common.xy
    mask[:, MEDIAPIPE53_COMMON] = common.mask
    xy[~mask] = 0
    return {"xy": xy, "mask": mask, "timestamps": common.timestamps.copy()}


def adapt_labeled_sequence(archive, sequence, *, label_lookup, source=None,
                           fps=None, timestamps=None, **options):
    """Use an exact-video label lookup supplied by PR #12's validated mapping.

    Returns private (CommonLandmarks, text). Duplicate/conflicting label records
    must be resolved by the label provider. IDs are never joined by FS alone.
    """
    text = label_lookup(sequence.video_id)
    if not isinstance(text, str) or not text:
        raise KeypointError("label_missing")
    common = adapt_crowd_sequence(archive.read_frames(sequence), sequence.frame_indices,
                                  video_id=sequence.video_id, source=source,
                                  fps=fps, timestamps=timestamps, **options)
    return common, text
