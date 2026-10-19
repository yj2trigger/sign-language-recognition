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
                wh = z["wh"] if "wh" in z.files else (1920, 1080)       # 예전 추출(z 없음)은 모두 1920x1080
                clips.append({"xy": z["xy"], "mask": z["mask"], "word": int(z["word_id"]), "signer": int(z["signer"]),
                              "z": z["z"] if "z" in z.files else None, "aspect": float(wh[0]) / float(wh[1])})
    return clips


def subsample(xy, mask, ratio, z=None):
    """프레임 간격을 ratio 배로 키운다(1.5 = 30fps -> 20fps). 낮은 fps 카메라 입력을 흉내 내는 데 쓴다. z 를 주면 같이 건너뛴다."""
    idx = np.unique(np.clip(np.round(np.arange(0, len(xy), ratio)).astype(int), 0, len(xy) - 1))
    return (xy[idx], mask[idx]) if z is None else (xy[idx], mask[idx], z[idx])


HAND_PTS = [9, 10] + list(range(11, 53))   # 몸 쪽 손목 2점 + 양손 42점(손 위치 이동 증강 대상)


def shift_hands(xy, mask, rng, frac=0.3):
    """두 손(몸 쪽 손목 포함)을 몸에 대해 같은 양만큼 옮긴다. 이동량은 어깨너비의 ±frac 안에서 상하·좌우 무작위. 새 배열을 돌려준다."""
    out = xy.copy()
    ok = mask[:, pp.L_SH] & mask[:, pp.R_SH]
    if ok.any():
        width = np.median(np.linalg.norm(xy[ok, pp.L_SH] - xy[ok, pp.R_SH], axis=-1))
        out[:, HAND_PTS] += (rng.uniform(-frac, frac, 2) * width).astype(np.float32)
    return out


def augment(xy, mask, rng, fps_aug=False, pos_aug=False, z=None):
    """시간 속도 변화(구간 일부만 쓰기), 좌표 잡음, 작은 회전, 프레임 결측. 입력은 건드리지 않고 새 배열을 돌려준다.
    fps_aug 이면 절반쯤은 먼저 20fps 로 낮춘다(30fps 로 학습한 모델이 낮은 fps 입력에 견디게).
    pos_aug 이면 두 손 전체를 몸에 대해 같은 양만큼 상하·좌우로 옮긴다(어깨너비의 ±30%, 개인별 손 높이 차이 흉내).
    기본값(False)일 때는 난수를 더 쓰지 않으므로 기존 실험의 증강본이 바뀌지 않는다.
    z(깊이)를 주면 시간 구간 자르기·fps 건너뛰기를 같은 인덱스로 적용해 (xy, mask, z)를 돌려준다(난수 추가 사용 없음).
    화면 안 회전은 깊이를 바꾸지 않으므로 z 에는 회전·잡음을 적용하지 않는다."""
    if fps_aug and rng.random() < 0.5:
        if z is None:
            xy, mask = subsample(xy, mask, 1.5)
        else:
            xy, mask, z = subsample(xy, mask, 1.5, z)
    n = len(xy)
    keep = max(8, int(round(n * rng.uniform(0.8, 1.0))))      # 구간을 줄여 쓰면 리샘플 뒤 속도가 빨라진다
    s = rng.integers(0, n - keep + 1)
    xy, mask = xy[s:s + keep].copy(), mask[s:s + keep].copy()
    if z is not None:
        z = z[s:s + keep].copy()
    if pos_aug:
        xy = shift_hands(xy, mask, rng)
    xy += rng.normal(0, 0.005, xy.shape).astype(np.float32)   # 이미지 폭의 0.5% 수준
    a = np.deg2rad(rng.uniform(-8, 8))
    c = xy.reshape(-1, 2).mean(0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    xy = (xy - c) @ rot.T + c
    drop = rng.random(len(xy)) < 0.08                         # 일부 프레임을 통째로 결측 처리
    mask[drop] = False
    return (xy, mask) if z is None else (xy, mask, z)


def to_arrays(clips, classes, length, with_hand_rel=True, rng=None, fps_ratio=1.0, fps_aug=False,
              angles=False, traj=False, pos_aug=False, bones=False, origin="shoulder", hand_z=False, folds=False):
    """클립 -> (X (N, length, F), y (N,), signer (N,)). rng 를 주면 증강을 적용한다. 어깨가 한 번도 안 보인 클립은 뺀다.
    fps_ratio != 1 이면(증강을 안 쓸 때) 모든 클립의 프레임을 그 비율로 건너뛰어 낮은 fps 카메라를 흉내 낸다(평가용)."""
    index = {w: i for i, w in enumerate(classes)}
    X, y, sg = [], [], []
    for c in clips:
        z = c.get("z") if hand_z else None
        if hand_z and z is None:
            raise ValueError("hand_z 를 쓰려면 z 가 저장된 추출 결과(features-z)가 필요하다")
        if rng is not None:
            out = augment(c["xy"], c["mask"], rng, fps_aug, pos_aug, z=z)
        elif fps_ratio != 1.0:
            out = subsample(c["xy"], c["mask"], fps_ratio, z=z)
        else:
            out = (c["xy"], c["mask"]) if z is None else (c["xy"], c["mask"], z)
        xy, mask = out[0], out[1]
        f = pp.build_features(xy, mask, length, with_hand_rel=with_hand_rel, with_angles=angles, with_traj=traj,
                              with_bones=bones, origin=origin, z=out[2] if hand_z else None,
                              aspect=c.get("aspect", 16 / 9), with_hand_z=hand_z,
                              with_folds=folds)
        if f is None:
            continue
        X.append(f)
        y.append(index[c["word"]])
        sg.append(c["signer"])
    return np.stack(X), np.array(y, dtype=np.int64), np.array(sg)     # Windows 의 numpy 기본 정수는 int32 라 torch 손실 함수가 거부한다


def augmented_bank(clips, classes, length, with_hand_rel, k, seed, fps_aug=False, angles=False, traj=False, pos_aug=False,
                   bones=False, origin="shoulder", hand_z=False, folds=False):
    """증강본을 k 벌 미리 만든다. 반환: (Xb (k, N, length, F), y (N,)). 학습 때 에폭마다 표본별로 한 벌을 고른다.
    매 에폭 새로 증강하면 특징 계산이 병목이라서 한 번만 만들고, 모든 모델·시드가 같은 증강본을 쓰게 한다."""
    rng = np.random.default_rng(seed)
    banks = []
    for _ in range(k):
        X, y, _ = to_arrays(clips, classes, length, with_hand_rel, rng=rng, fps_aug=fps_aug,
                            angles=angles, traj=traj, pos_aug=pos_aug, bones=bones, origin=origin, hand_z=hand_z,
                            folds=folds)
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
