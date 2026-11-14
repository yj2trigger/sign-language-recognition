"""구간 탐지기(boundary.py) 확인(합성 데이터, MediaPipe 없음).
실행: python -m pytest tests/test_boundary.py  (torch 가 없으면 모델 테스트만 건너뛴다)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import boundary as bd
import dataset_bio as db

FPS = 15.0


# ---- 프레임 특징 ----

def test_sanitize_masks_bad_points():
    xy, mask = np.full((53, 2), 0.5, np.float32), np.ones(53, bool)
    xy[3], xy[4] = np.nan, (2.0, 0.5)                     # NaN, 화면 밖으로 크게 벗어남
    out, m = bd.sanitize(xy, mask)
    assert m[0] and not m[3] and not m[4] and (out[3] == 0).all()
    with pytest.raises(ValueError):
        bd.sanitize(np.zeros((52, 2)), np.zeros(52, bool))


def test_batch_features_shape_and_hand_dropout():
    s = db.synthetic_sentence(np.random.default_rng(11), n_words=2)
    f = bd.batch_features(s["t"], s["xy"], s["mask"])
    assert f.shape == (len(s["t"]), bd.N_FEAT) and np.isfinite(f).all() and np.abs(f).max() <= 4.0
    col = {n: i for i, n in enumerate(bd.FEAT_NAMES)}
    r = s["mask"][:, 32]
    ok = np.flatnonzero(r[:-2] & r[1:-1] & r[2:]) + 1      # 앞·뒤 프레임에도 오른손이 보이는 프레임
    i = int(ok[len(ok) // 2])
    mask = s["mask"].copy()
    mask[i, 11:] = False                                   # 손 두 개가 한 프레임 안 잡힌다
    g = bd.batch_features(s["t"], s["xy"], mask)
    assert g[i, col["L_present"]] == 0 and g[i, col["R_present"]] == 0 and g[i, col["R_on_change"]] == -1
    assert g[i + 1, col["R_on_change"]] == 1 and g[i + 1, col["R_speed"]] == 0   # 다시 보인 첫 프레임은 속도 0(튀는 값 방지)
    assert bd.batch_features(np.zeros(0), np.zeros((0, 53, 2)), np.zeros((0, 53), bool)).shape == (0, bd.N_FEAT)


def test_face_signals_reads_brow_and_mouth():
    face, fm = np.zeros((33, 2), np.float32), np.ones(33, bool)
    face[:, 0] = 0.5
    face[32, 1], face[31, 1] = 0.1, 0.5                   # 이마, 턱 -> 얼굴 높이 0.4
    face[0:10, 1], face[10:18, 1] = 0.20, 0.24             # 눈썹, 눈
    face[26, 1], face[27, 1] = 0.40, 0.44                  # 입술 안쪽 위·아래
    assert bd.face_signals(face, fm) == pytest.approx((0.1, 0.1))
    fm[5] = False
    assert bd.face_signals(face, fm) is None              # 점 하나라도 안 보이면 계산하지 않는다
    assert bd.face_signals(np.zeros((33, 2)), np.ones(33, bool)) is None   # 얼굴 높이 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
