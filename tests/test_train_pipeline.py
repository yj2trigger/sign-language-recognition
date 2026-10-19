"""분할·모델·학습 루프 확인용 테스트(합성 데이터). 실행: .venv\\Scripts\\python tests\\test_train_pipeline.py
'코드가 끝까지 돌고 학습이 되는지'만 확인한다. 실제 수어 데이터의 성능이나 결론의 근거가 아니다."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset as ds
import models
import train
import preprocess as pp


def synthetic_clips(n_signers=10, n_class=5, T=60, reps=8, seed=0):
    """클래스마다 오른손 궤적의 주파수가 다르고, 수행자마다 위치·크기·속도가 조금 다른 합성 클립.
    수행자·클래스당 reps 개씩 만든다(위상이 조금씩 다름). 클립이 너무 적으면 최적화 단계 수가 모자라 학습이 안 된다."""
    rng = np.random.default_rng(seed)
    clips = []
    for s in range(1, n_signers + 1):
        off, sc, sp = rng.normal(0, 0.03, 2), rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
        for k in range(n_class):
            for _ in range(reps):
                t = np.linspace(0, 1, T) * sp + rng.uniform(0, 0.3)
                xy = np.zeros((T, 53, 2), np.float32)
                xy[:, pp.L_SH] = (0.6, 0.5)
                xy[:, pp.R_SH] = (0.4, 0.5)
                cx = 0.35 + 0.12 * sc * np.sin(2 * np.pi * (k + 1) * t)
                cy = 0.5 + 0.12 * sc * np.cos(2 * np.pi * (k % 3 + 1) * t)
                hand = rng.normal(0, 0.004, (21, 2)) + np.linspace(0, 0.06, 21)[:, None]
                xy[:, 32:53] = np.stack([cx, cy], 1)[:, None, :] + hand[None] + off
                xy[:, 11:32] = (0.62, 0.8)
                clips.append({"xy": xy, "mask": np.ones((T, 53), bool), "word": 100 + k, "signer": s})
    return clips


def test_folds_have_no_signer_leak():
    folds = ds.make_folds(range(1, 19), 6, seed=0)
    assert ds.sha(folds) == ds.sha(ds.make_folds(range(1, 19), 6, seed=0))          # 재현 가능
    tests = [s for f in folds for s in f["test"]]
    assert sorted(tests) == list(range(1, 19))                                       # 모든 수행자가 시험에 정확히 한 번
    for f in folds:
        assert not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"]) or set(f["val"]) & set(f["test"]))
        assert len(f["train"]) + len(f["val"]) + len(f["test"]) == 18


def test_models_shape_and_comparable_size():
    x = torch.zeros(4, 64, 243)
    sizes = {}
    for name in ("gru", "tcn"):
        m = models.build(name, 243, 100)
        assert m(x).shape == (4, 100)
        sizes[name] = models.n_params(m)
    assert abs(sizes["gru"] - sizes["tcn"]) / max(sizes.values()) < 0.2, sizes          # 파라미터 수 20% 이내
    assert 100_000 < min(sizes.values()) and max(sizes.values()) < 400_000, sizes
    base = models.build("mlp", 243, 100)                                                 # 기준선은 시간 순서를 안 쓰므로 순서를 바꿔도 출력이 같다
    base.eval()
    assert base(x).shape == (4, 100)
    r = torch.randn(2, 64, 243)
    assert torch.allclose(base(r), base(r[:, torch.randperm(64)]), atol=1e-5)


def test_augment_does_not_mutate_and_keeps_shapes():
    c = synthetic_clips(1, 1)[0]
    before = c["xy"].copy()
    xy, mask = ds.augment(c["xy"], c["mask"], np.random.default_rng(0))
    assert np.array_equal(before, c["xy"]) and xy.shape[1:] == (53, 2) and 8 <= len(xy) <= len(before)


def test_pos_aug_moves_hands_only_and_default_unchanged():
    c = synthetic_clips(1, 1)[0]
    a1, m1 = ds.augment(c["xy"], c["mask"], np.random.default_rng(5))
    a2, m2 = ds.augment(c["xy"], c["mask"], np.random.default_rng(5), pos_aug=False)
    assert np.array_equal(a1, a2) and np.array_equal(m1, m2)       # 기본값이면 기존 증강과 완전히 같다(기존 실험 재현성)
    xy = c["xy"]
    for seed in range(20):
        out = ds.shift_hands(xy, c["mask"], np.random.default_rng(seed))
        d = out - xy
        assert np.allclose(d[:, :9], 0)                              # 머리·어깨·팔꿈치는 그대로
        assert np.allclose(d[:, ds.HAND_PTS], d[0, 9], atol=1e-6)    # 손 쪽 점은 모두 같은 양만큼 이동(float32 반올림 허용)
        assert (np.abs(d[0, 9]) <= 0.3 * 0.2 + 1e-6).all()           # 이동량은 어깨너비(0.2)의 ±30% 안
    assert np.array_equal(xy, c["xy"])                                # 입력은 바뀌지 않음


def test_augment_crops_z_with_same_frames():
    c = synthetic_clips(1, 1)[0]
    z = np.repeat(np.arange(60, dtype=np.float32)[:, None], 53, 1)    # z 에 프레임 번호를 넣어 둠
    a = ds.augment(c["xy"], c["mask"], np.random.default_rng(3))
    b = ds.augment(c["xy"], c["mask"], np.random.default_rng(3), z=z)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])  # z 를 줘도 x·y 증강 결과는 같다(난수 추가 사용 없음)
    zz = b[2][:, 0]
    assert len(zz) == len(b[0]) and np.array_equal(zz, np.arange(zz[0], zz[0] + len(zz)))   # 같은 연속 구간으로 잘림
    _, _, zs = ds.subsample(c["xy"], c["mask"], 1.5, z=z)
    assert np.array_equal(zs[:3, 0], [0, 2, 3])                     # fps 건너뛰기도 같은 인덱스(0, 1.5→2, 3)
    cz = dict(c, z=z, aspect=16 / 9)
    X, _, _ = ds.to_arrays([cz], [c["word"]], 64, hand_z=True)
    assert X.shape == (1, 64, 243 + 42)


def test_subsample_emulates_lower_fps():
    c = synthetic_clips(1, 1, T=90)[0]
    xy, mask = ds.subsample(c["xy"], c["mask"], 1.5)
    assert len(xy) == 60 and len(mask) == 60                      # 30fps 90프레임 -> 20fps 60프레임(같은 길이의 시간)
    assert np.array_equal(xy[1], c["xy"][2])                       # 두 번째 프레임은 원본 2번(1.5 를 반올림)
    X1, _, _ = ds.to_arrays([c], [c["word"]], 64)
    X2, _, _ = ds.to_arrays([c], [c["word"]], 64, fps_ratio=1.5)
    assert X1.shape == X2.shape == (1, 64, 243)                    # 길이를 64 로 맞추므로 fps 가 달라도 모양은 같다
    X3, _, _ = ds.to_arrays([c], [c["word"]], 64, rng=np.random.default_rng(0), fps_aug=True)
    assert X3.shape == (1, 64, 243)


def test_train_one_learns_synthetic_task():
    clips = synthetic_clips()
    classes = [100 + k for k in range(5)]
    sel = lambda ids: [c for c in clips if c["signer"] in ids]
    va = ds.to_arrays(sel({7, 8}), classes, 64)
    te = ds.to_arrays(sel({9, 10}), classes, 64)
    cfg = {"lr": 3e-3, "wd": 1e-2, "batch": 16, "epochs": 40, "patience": 12, "length": 64, "hand_rel": True}
    bank = ds.augmented_bank(sel({1, 2, 3, 4, 5, 6}), classes, 64, True, k=4, seed=0)
    for name in ("gru", "tcn"):
        model, best, hist = train.train_one(name, bank, va[:2], classes, cfg, seed=0)
        res = train.metrics(train.predict(model, te[0]), te[1])
        assert best > 0.8 and res["macro_f1"] > 0.7, (name, best, res, hist)             # 우연 수준(0.2)보다 훨씬 위
        print("  %s: 검증 F1 %.2f 시험 F1 %.2f (%d에폭)" % (name, best, res["macro_f1"], len(hist)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
