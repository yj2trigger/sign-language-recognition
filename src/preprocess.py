"""좌표 정규화, 결측 처리, 고정 길이 리샘플, 특징 벡터 조립.

입력은 landmarks.py 가 만든 xy (T, 53, 2)와 mask (T, 53).
점 순서: 상체 11점(0~10) + 왼손 21점(11~31) + 오른손 21점(32~52).
"""
import numpy as np

L_SH, R_SH = 5, 6            # 상체 11점 안에서 왼쪽·오른쪽 어깨 위치
HAND_STARTS = (11, 32)        # 왼손·오른손 21점이 시작하는 위치
MIDDLE_MCP = 9                # 손 21점 안에서 가운뎃손가락 뿌리(손 크기 기준)


def _hold_nearest(x, valid):
    """valid 가 False 인 프레임을 가장 가까운 앞쪽 유효 값으로 채운다(앞쪽에 없으면 뒤쪽). x: (T, ...)"""
    idx = np.where(valid, np.arange(len(valid)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(valid)
    idx[idx < 0] = first
    return x[idx]


def normalize(xy, mask):
    """어깨 중점을 원점, 어깨너비를 1로. 어깨가 안 보인 프레임은 가까운 유효 프레임 값을 쓴다.
    어깨가 한 번도 유효하지 않으면 None."""
    width = np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=-1)
    valid = mask[:, L_SH] & mask[:, R_SH] & (width > 1e-6)
    if not valid.any():
        return None
    center = _hold_nearest((xy[:, L_SH] + xy[:, R_SH]) / 2, valid)
    scale = _hold_nearest(width, valid)
    return (xy - center[:, None, :]) / scale[:, None, None]


def fill_missing(xy, mask, max_gap=5):
    """점별로 짧은 결측(max_gap 프레임 이하, 양쪽에 유효 값이 있을 때)은 선형 보간,
    그 밖의 결측은 가까운 유효 값을 유지한다. 한 번도 안 보인 점은 0."""
    out = np.zeros_like(xy)
    t = np.arange(len(xy))
    for p in range(xy.shape[1]):
        valid = mask[:, p]
        if valid.all():
            out[:, p] = xy[:, p]                 # 결측이 없는 점은 그대로(대부분의 점이 여기에 해당)
            continue
        if not valid.any():
            continue
        held = _hold_nearest(xy[:, p], valid)
        interp = np.stack([np.interp(t, t[valid], xy[valid, p, c]) for c in range(2)], axis=-1)
        # 연속 결측 구간의 길이를 구해 짧은 구간(양쪽이 유효한 경우)만 보간 값을 쓴다
        out[:, p] = held
        i = 0
        while i < len(valid):
            if valid[i]:
                i += 1
                continue
            j = i
            while j < len(valid) and not valid[j]:
                j += 1
            if i > 0 and j < len(valid) and (j - i) <= max_gap:
                out[i:j, p] = interp[i:j]
            i = j
    return out


def resample(x, length):
    """시간축을 선형 보간해 length 프레임으로 맞춘다. x: (T, ...)"""
    n = len(x)
    src = np.linspace(0, n - 1, length)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def hand_relative(xy, mask):
    """손목 기준 상대 좌표를 손 크기(손목~가운뎃손가락 뿌리 거리)로 나눈 값. 안 보인 손은 0. 반환 (T, 42, 2)"""
    parts = []
    for s in HAND_STARTS:
        h = xy[:, s:s + 21]
        size = np.linalg.norm(h[:, MIDDLE_MCP] - h[:, 0], axis=-1)
        ok = mask[:, s] & (size > 1e-6)
        rel = (h - h[:, :1]) / np.where(ok, size, 1.0)[:, None, None]
        parts.append(np.where(ok[:, None, None], rel, 0.0))
    return np.concatenate(parts, axis=1)


def build_features(xy, mask, length, max_gap=5, with_hand_rel=True):
    """(T, 53, 2)·(T, 53) -> (length, F). 순서: 몸 기준 좌표 106 + [손 상대 좌표 84] + 마스크 53.
    어깨가 한 번도 안 보인 클립은 None."""
    norm = normalize(xy, mask)
    if norm is None:
        return None
    filled = fill_missing(norm, mask, max_gap)
    parts = [filled.reshape(len(filled), -1)]
    if with_hand_rel:
        parts.append(hand_relative(filled, mask).reshape(len(filled), -1))
    parts.append(mask.astype(np.float32))
    feat = np.concatenate(parts, axis=1)
    return resample(feat, length).astype(np.float32)    # 보간 가중치가 float64 라 리샘플 뒤에 float32 로 맞춘다
