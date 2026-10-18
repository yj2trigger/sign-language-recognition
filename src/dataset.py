"""추출한 클립(npz)을 학습용 배열로 바꾸고, 수행자 단위 분할을 만든다."""
import glob
import hashlib
import json
import os

import numpy as np

import preprocess as pp


def load_clips(feature_dirs, word_ids):
    """npz 클립 중 선정한 단어 번호의 것만 읽는다. 각 클립은 {xy, mask, word, signer} 딕셔너리."""
    keep = set(word_ids)
    clips = []
    for d in feature_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            if int(z["word_id"]) in keep:
                clips.append({"xy": z["xy"], "mask": z["mask"], "word": int(z["word_id"]), "signer": int(z["signer"])})
    return clips


def subsample(xy, mask, ratio):
    """프레임 간격을 ratio 배로 키운다(1.5 = 30fps -> 20fps). 낮은 fps 카메라 입력을 흉내 내는 데 쓴다."""
    idx = np.unique(np.clip(np.round(np.arange(0, len(xy), ratio)).astype(int), 0, len(xy) - 1))
    return xy[idx], mask[idx]


def augment(xy, mask, rng, fps_aug=False):
    """시간 속도 변화(구간 일부만 쓰기), 좌표 잡음, 작은 회전, 프레임 결측. 입력은 건드리지 않고 새 배열을 돌려준다.
    fps_aug 이면 절반쯤은 먼저 20fps 로 낮춘다(30fps 로 학습한 모델이 낮은 fps 입력에 견디게)."""
    if fps_aug and rng.random() < 0.5:
        xy, mask = subsample(xy, mask, 1.5)
    n = len(xy)
    keep = max(8, int(round(n * rng.uniform(0.8, 1.0))))      # 구간을 줄여 쓰면 리샘플 뒤 속도가 빨라진다
    s = rng.integers(0, n - keep + 1)
    xy, mask = xy[s:s + keep].copy(), mask[s:s + keep].copy()
    xy += rng.normal(0, 0.005, xy.shape).astype(np.float32)   # 이미지 폭의 0.5% 수준
    a = np.deg2rad(rng.uniform(-8, 8))
    c = xy.reshape(-1, 2).mean(0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    xy = (xy - c) @ rot.T + c
    drop = rng.random(len(xy)) < 0.08                         # 일부 프레임을 통째로 결측 처리
    mask[drop] = False
    return xy, mask


def to_arrays(clips, classes, length, with_hand_rel=True, rng=None, fps_ratio=1.0, fps_aug=False):
    """클립 -> (X (N, length, F), y (N,), signer (N,)). rng 를 주면 증강을 적용한다. 어깨가 한 번도 안 보인 클립은 뺀다.
    fps_ratio != 1 이면(증강을 안 쓸 때) 모든 클립의 프레임을 그 비율로 건너뛰어 낮은 fps 카메라를 흉내 낸다(평가용)."""
    index = {w: i for i, w in enumerate(classes)}
    X, y, sg = [], [], []
    for c in clips:
        if rng is not None:
            xy, mask = augment(c["xy"], c["mask"], rng, fps_aug)
        elif fps_ratio != 1.0:
            xy, mask = subsample(c["xy"], c["mask"], fps_ratio)
        else:
            xy, mask = c["xy"], c["mask"]
        f = pp.build_features(xy, mask, length, with_hand_rel=with_hand_rel)
        if f is None:
            continue
        X.append(f)
        y.append(index[c["word"]])
        sg.append(c["signer"])
    return np.stack(X), np.array(y, dtype=np.int64), np.array(sg)     # Windows 의 numpy 기본 정수는 int32 라 torch 손실 함수가 거부한다


def augmented_bank(clips, classes, length, with_hand_rel, k, seed, fps_aug=False):
    """증강본을 k 벌 미리 만든다. 반환: (Xb (k, N, length, F), y (N,)). 학습 때 에폭마다 표본별로 한 벌을 고른다.
    매 에폭 새로 증강하면 특징 계산이 병목이라서 한 번만 만들고, 모든 모델·시드가 같은 증강본을 쓰게 한다."""
    rng = np.random.default_rng(seed)
    banks = []
    for _ in range(k):
        X, y, _ = to_arrays(clips, classes, length, with_hand_rel, rng=rng, fps_aug=fps_aug)
        banks.append(X)
    return np.stack(banks), y


def make_folds(signers, n_folds, seed=0):
    """수행자를 섞어 n_folds 개 묶음으로 나누고, 묶음 i 를 시험, i+1 을 검증, 나머지를 학습으로 쓴다."""
    s = sorted(signers)
    np.random.default_rng(seed).shuffle(s)
    chunks = [sorted(int(x) for x in c) for c in np.array_split(s, n_folds)]
    return [{"test": chunks[i], "val": chunks[(i + 1) % n_folds],
             "train": sorted(x for j, c in enumerate(chunks) if j not in (i, (i + 1) % n_folds) for x in c)}
            for i in range(n_folds)]


def sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]
