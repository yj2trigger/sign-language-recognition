"""영상 전체·목표 fps 추출 확인용 테스트(합성 데이터). 실행: .venv\\Scripts\\python tests\\test_extract_full.py
프레임 선택과 라벨 구간 자르기가 계산대로 되는지만 확인한다. MediaPipe 결과의 품질은 확인하지 않는다."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset as ds
import landmarks


def test_select_frames():
    assert landmarks.select_frames(10, 30, 15, 0) == [0, 2, 4, 6, 8]
    assert landmarks.select_frames(10, 30, 15, 1) == [1, 3, 5, 7, 9]
    assert landmarks.select_frames(13, 60, 15, 3) == [3, 7, 11]
    idx = landmarks.select_frames(300, 29.97, 15, 0)                # 정수배가 아니어도 간격이 2 근처로 고르다
    assert len(idx) == 150 and set(np.diff(idx)) <= {1, 2, 3}
    assert landmarks.select_frames(5, 15, 15, 0) == [0, 1, 2, 3, 4]


def _save(path, t, full):
    T = len(t)
    extra = dict(full=True, t=np.asarray(t, np.float32), seg_start=1.0, seg_end=2.0,
                 face=np.zeros((T, 33, 2), np.float32), face_mask=np.ones((T, 33), bool)) if full else {}
    np.savez(path, xy=np.arange(T * 53 * 2, dtype=np.float32).reshape(T, 53, 2), mask=np.ones((T, 53), bool),
             z=np.zeros((T, 53), np.float32), word_id=1501, signer=1, wh=np.array([1920, 1080]), **extra)


def test_segment_and_load():
    with tempfile.TemporaryDirectory() as d:
        t = np.arange(0, 4.0, 1 / 15)                                 # 영상 4초, 15fps
        _save(os.path.join(d, "NIA_SL_WORD1501_REAL01_F.npz"), t, full=True)
        clip = ds.load_clips([d], [1501])[0]
        keep = (t >= 0.5 - 1e-6) & (t <= 2.5 + 1e-6)                  # 라벨 1~2초 ±0.5초
        assert len(clip["xy"]) == keep.sum() and clip["face"].shape == (keep.sum(), 33, 2)
        assert np.array_equal(clip["xy"][0], np.load(os.path.join(d, "NIA_SL_WORD1501_REAL01_F.npz"))["xy"][keep][0])
    with tempfile.TemporaryDirectory() as d:                          # 예전 npz 는 그대로, face 는 None
        _save(os.path.join(d, "NIA_SL_WORD1501_REAL01_F.npz"), np.arange(20), full=False)
        clip = ds.load_clips([d], [1501])[0]
        assert len(clip["xy"]) == 20 and clip["face"] is None


def _body(T, rng):
    xy = rng.uniform(0.3, 0.7, (T, 53, 2)).astype(np.float32)
    xy[:, 5], xy[:, 6] = (0.6, 0.4), (0.4, 0.4)                       # 어깨(왼·오른)를 고정해 정규화가 되게
    return xy, np.ones((T, 53), bool)


def test_face_features():
    import preprocess as pp
    rng = np.random.default_rng(0)
    xy, mask = _body(20, rng)
    face = rng.uniform(0.45, 0.55, (20, 33, 2)).astype(np.float32)
    fmask = np.ones((20, 33), bool)
    base = pp.build_features(xy, mask, 64)
    with_face = pp.build_features(np.concatenate([xy, face], 1), np.concatenate([mask, fmask], 1), 64, with_face=True)
    assert base.shape == (64, 243) and with_face.shape == (64, 243 + 73)
    assert np.allclose(with_face[:, :190], base[:, :190]) and np.allclose(with_face[:, -53:], base[:, -53:])   # 몸·손·마스크는 그대로
    expr = pp.face_expression(face, fmask)
    moved = pp.face_expression(face * 1.7 + 0.2, fmask)               # 얼굴 위치·크기를 바꿔도 표정 모양은 같다
    assert np.allclose(expr, moved, atol=1e-5)
    fmask[3] = False                                                  # 얼굴이 안 잡힌 프레임은 0
    assert not pp.face_expression(face, fmask)[3].any()


def test_to_arrays_with_face_augments_together():
    rng = np.random.default_rng(1)
    xy, mask = _body(30, rng)
    clip = {"xy": xy, "mask": mask, "word": 1501, "signer": 1, "z": None, "aspect": 16 / 9,
            "face": rng.uniform(0.45, 0.55, (30, 33, 2)).astype(np.float32), "face_mask": np.ones((30, 33), bool)}
    X, y, _ = ds.to_arrays([clip], [1501], 64, rng=np.random.default_rng(2), face=True)
    X0, _, _ = ds.to_arrays([clip], [1501], 64)
    assert X.shape == (1, 64, 316) and X0.shape == (1, 64, 243) and y.tolist() == [0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
