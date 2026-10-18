"""전처리 확인용 테스트. 실행: .venv\\Scripts\\python tests\\test_preprocess.py
합성 데이터로 '코드가 의도대로 동작하는지'만 확인한다. 성능·결론의 근거가 아니다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as pp


def make_clip(T=20):
    xy = np.zeros((T, 53, 2), np.float32)
    xy[:, pp.L_SH] = (0.6, 0.5)   # 어깨너비 0.2, 중점 (0.5, 0.5)
    xy[:, pp.R_SH] = (0.4, 0.5)
    xy[:, 11:32] = np.linspace(0, 1, 21)[None, :, None] * 0.1 + 0.55   # 왼손
    return xy, np.ones((T, 53), bool)


def test_normalize():
    xy, mask = make_clip()
    n = pp.normalize(xy, mask)
    assert np.allclose((n[:, pp.L_SH] + n[:, pp.R_SH]) / 2, 0, atol=1e-6)      # 중점이 원점
    assert np.allclose(np.linalg.norm(n[:, pp.L_SH] - n[:, pp.R_SH], axis=-1), 1, atol=1e-5)  # 어깨너비 1


def test_normalize_no_shoulder():
    xy, mask = make_clip()
    mask[:, pp.L_SH] = False
    assert pp.normalize(xy, mask) is None


def test_fill_short_gap_interpolates_long_gap_holds():
    T = 30
    xy = np.zeros((T, 53, 2), np.float32)
    xy[:, 0, 0] = np.arange(T, dtype=np.float32)       # 0번 점의 x가 프레임 번호와 같다
    mask = np.ones((T, 53), bool)
    mask[5:8, 0] = False        # 짧은 결측(3프레임)
    mask[15:25, 0] = False      # 긴 결측(10프레임)
    out = pp.fill_missing(xy, mask, max_gap=5)
    assert np.allclose(out[5:8, 0, 0], [5, 6, 7])       # 선형 보간이 원래 값과 일치
    assert np.allclose(out[15:25, 0, 0], 14)            # 긴 결측은 직전 유효 값 유지


def test_features_shape_and_finite():
    xy, mask = make_clip()
    mask[:, 32:53] = False      # 오른손 안 보임
    f = pp.build_features(xy, mask, length=32)
    assert f.shape == (32, 106 + 84 + 53)
    assert f.dtype == np.float32                        # torch 가 float32 만 받는다(보간 때문에 float64 가 되기 쉬움)
    assert np.isfinite(f).all()
    assert (f[:, 106 + 42:106 + 84] == 0).all()         # 안 보인 오른손의 상대 좌표는 0
    assert pp.build_features(xy, mask, 32, with_hand_rel=False).shape == (32, 106 + 53)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
