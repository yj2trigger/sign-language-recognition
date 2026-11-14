"""Segment adapter for the timestamped buffer used by src/realtime.py."""
import math

import numpy as np

from fingerspelling_runtime import read_fingerspelling
from fingerspelling.sequence_validation import validate_sequence


def read_segment(buffer, start, end, *, margin=0.0, reader=read_fingerspelling):
    """Read an observed 15fps segment without synthesizing missing frames.

    The caller retains segmentation and buffer ownership. Irregular timing needs
    an explicit resampling policy before this fixed-rate model can be used.
    """
    bounds = (start, end, margin)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in bounds):
        raise ValueError('invalid segment bounds')
    if start < 0 or end < start or margin < 0:
        raise ValueError('invalid segment bounds')
    selected = []
    previous = None
    for row in buffer:
        timestamp = row[0]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float, np.floating)):
            raise ValueError('invalid timestamp')
        if not math.isfinite(timestamp) or (previous is not None and timestamp <= previous):
            raise ValueError('timestamps must increase')
        previous = timestamp
        if start - margin <= timestamp <= end + margin:
            selected.append(row)
    if not selected:
        return ''
    times = np.asarray([row[0] for row in selected])
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / 15, atol=0.002, rtol=0):
        raise ValueError('segment requires explicit 15fps resampling')
    xy = np.stack([row[1] for row in selected])
    mask = np.stack([row[2] for row in selected])
    validate_sequence(xy, mask, allow_all_missing=True)
    result = reader(xy, mask)
    if not isinstance(result, str):
        raise TypeError('reader must return str')
    return result
