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


NOSE = 0


def normalize(xy, mask, origin="shoulder"):
    """원점(어깨 중점 또는 코)을 0, 어깨너비를 1로. 기준점이 안 보인 프레임은 가까운 유효 프레임 값을 쓴다.
    origin="nose" 는 얼굴 기준이다. 수행자마다 어깨와 얼굴 사이 간격(자세·체형)이 달라 어깨 기준이면 손 높이가 달라 보이는
    차이를 줄이려는 것(T-025 후속). 코가 한 번도 안 보이면 어깨 중점으로 대신한다. 어깨가 한 번도 유효하지 않으면 None."""
    width = np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=-1)
    valid = mask[:, L_SH] & mask[:, R_SH] & (width > 1e-6)
    if not valid.any():
        return None
    if origin == "nose" and mask[:, NOSE].any():
        center = _hold_nearest(xy[:, NOSE], mask[:, NOSE])
    else:
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


# 손가락 관절 각도를 잴 세 점(가운데 점이 꼭짓점): 손가락마다 3개 x 5 = 15
FINGER_TRIPLES = [(a, b, c) for f in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
                  for a, b, c in zip(f, f[1:], f[2:])]


def finger_angles(xy, mask):
    """손가락 관절 15개의 굽은 정도(각도/pi, 곧게 펴면 1). 회전·크기에 영향받지 않는 손 모양 특징. 안 보인 손은 0. 반환 (T, 30)"""
    parts = []
    for s in HAND_STARTS:
        h = xy[:, s:s + 21]
        a = np.stack([h[:, i] - h[:, j] for i, j, _ in FINGER_TRIPLES], 1)
        b = np.stack([h[:, k] - h[:, j] for _, j, k in FINGER_TRIPLES], 1)
        cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
        ang = np.arccos(np.clip(cos, -1, 1)) / np.pi
        parts.append(np.where(mask[:, s:s + 1], ang, 0.0))
    return np.concatenate(parts, axis=1)


def wrist_trajectory(xy, mask):
    """두 손목의 궤적에서 클립 평균 위치를 뺀 값(위치와 무관한 움직임 모양). 안 보인 프레임·손은 0. 반환 (T, 4)"""
    parts = []
    for s in HAND_STARTS:
        ok = mask[:, s]
        w = xy[:, s]
        center = w[ok].mean(0) if ok.any() else np.zeros(2)
        parts.append(np.where(ok[:, None], w - center, 0.0))
    return np.concatenate(parts, axis=1)


HAND_BONES = [(a, b) for f in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
              for a, b in zip(f, f[1:])]   # 손 뼈대 20개


def bone_ratios(xy, mask):
    """손 뼈대 20개 x 두 손의 '현재 화면 길이 / 클립 안 최대 화면 길이'. 손가락이 카메라 쪽으로 기울면 화면에서 짧아지므로
    2D 좌표만으로 깊이 방향 기울기를 근사한다(앞뒤 부호는 모름). 손 크기·위치에 영향받지 않는다. 안 보인 손은 0. 반환 (T, 40)"""
    parts = []
    for s in HAND_STARTS:
        h = xy[:, s:s + 21]
        ln = np.stack([np.linalg.norm(h[:, b] - h[:, a], axis=-1) for a, b in HAND_BONES], 1)
        ok = mask[:, s]
        mx = ln[ok].max(0) if ok.any() else np.ones(len(HAND_BONES))
        parts.append(np.where(ok[:, None], ln / np.maximum(mx, 1e-9), 0.0))
    return np.concatenate(parts, axis=1)


def build_features(xy, mask, length, max_gap=5, with_hand_rel=True, with_angles=False, with_traj=False,
                   with_bones=False, origin="shoulder"):
    """(T, 53, 2)·(T, 53) -> (length, F). 순서: 몸 기준 좌표 106 + [손 상대 좌표 84] + [손가락 각도 30] + [손목 궤적 4]
    + [손 뼈대 길이 비율 40] + 마스크 53. 어깨가 한 번도 안 보인 클립은 None."""
    norm = normalize(xy, mask, origin)
    if norm is None:
        return None
    filled = fill_missing(norm, mask, max_gap)
    parts = [filled.reshape(len(filled), -1)]
    if with_hand_rel:
        parts.append(hand_relative(filled, mask).reshape(len(filled), -1))
    if with_angles:
        parts.append(finger_angles(filled, mask))
    if with_traj:
        parts.append(wrist_trajectory(filled, mask))
    if with_bones:
        parts.append(bone_ratios(filled, mask))
    parts.append(mask.astype(np.float32))
    feat = np.concatenate(parts, axis=1)
    return resample(feat, length).astype(np.float32)    # 보간 가중치가 float64 라 리샘플 뒤에 float32 로 맞춘다
