"""augmented_bank 를 사전 할당으로 바꾼 뒤에도 값이 예전 방식(벌마다 to_arrays → astype → np.stack)과 비트 단위로 같은지."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds
from test_train_pipeline import synthetic_clips


def old_bank(clips, classes, k, seed, dtype):
    rng = np.random.default_rng(seed)
    banks = []
    for _ in range(k):
        X, y, _ = ds.to_arrays(clips, classes, 64, True, rng=rng)
        banks.append(X.astype(dtype, copy=False))
    return np.stack(banks), y


def test_prealloc_bank_matches_old():
    clips = synthetic_clips(n_signers=3, n_class=4, reps=2)
    clips[1] = dict(clips[1], mask=np.zeros_like(clips[1]["mask"]))   # 건너뛰는 클립도 하나 넣어 자르기까지 확인
    classes = sorted({c["word"] for c in clips})
    for dtype in (np.float32, np.float16):
        Xo, yo = old_bank(clips, classes, 3, 7, dtype)
        Xn, yn = ds.augmented_bank(clips, classes, 64, True, 3, 7, dtype=dtype)
        assert Xn.dtype == dtype and Xn.shape == Xo.shape and len(yn) == len(clips) - 1
        assert np.array_equal(Xn, Xo) and np.array_equal(yn, yo)
