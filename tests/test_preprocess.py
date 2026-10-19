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


def test_finger_angles_straight_and_invariant():
    xy, mask = make_clip()
    xy[:, 11:32] = np.stack([np.linspace(0.5, 0.6, 21), np.full(21, 0.7)], 1)   # 왼손 21점이 일직선 = 모든 손가락이 곧게 펴짐
    a = pp.finger_angles(xy, mask)
    assert a.shape == (20, 30)
    assert np.allclose(a[:, :15], 1.0, atol=1e-5)                    # 곧게 펴면 각도/pi = 1
    big = xy.copy()
    big[:, 11:32] = (xy[:, 11:32] - 0.5) * 2 + 0.3                   # 손을 2배로 키우고 옮겨도
    assert np.allclose(pp.finger_angles(big, mask)[:, :15], a[:, :15], atol=1e-5)   # 각도는 같다
    mask[:, 32:53] = False
    assert (pp.finger_angles(xy, mask)[:, 15:] == 0).all()           # 안 보인 손은 0


def test_trajectory_removes_position():
    xy, mask = make_clip()
    xy[:, 11, 0] = np.linspace(0.4, 0.6, 20)                          # 왼손목이 오른쪽으로 이동
    moved = xy.copy()
    moved[:, 11] += 0.2                                               # 같은 움직임을 다른 위치에서
    assert np.allclose(pp.wrist_trajectory(xy, mask), pp.wrist_trajectory(moved, mask), atol=1e-6)
    f = pp.build_features(xy, mask, 32, with_angles=True, with_traj=True)
    assert f.shape == (32, 106 + 84 + 30 + 4 + 53) and f.dtype == np.float32


def test_bone_ratios_detect_foreshortening():
    xy, mask = make_clip()
    xy[:, 11:32] = np.stack([np.linspace(0.5, 0.6, 21), np.full(21, 0.7)], 1)
    xy[10:, 15] = xy[10:, 14] + (xy[10:, 15] - xy[10:, 14]) * 0.5   # 뒤쪽 프레임에서 엄지 끝 마디(3-4)가 화면에서 절반으로 짧아짐
    r = pp.bone_ratios(xy, mask)
    assert r.shape == (20, 40)
    assert np.allclose(r[:10, 3], 1.0) and np.allclose(r[10:, 3], 0.5)    # 뼈대 3번 = (3, 4)
    assert np.allclose(r[:, 0], 1.0)                                      # 다른 뼈대는 그대로
    big = xy.copy(); big[:, 11:32] = (xy[:, 11:32] - 0.5) * 2 + 0.3
    assert np.allclose(pp.bone_ratios(big, mask)[:, :20], r[:, :20])      # 손 크기·위치에 영향받지 않음


def test_nose_origin():
    xy, mask = make_clip()
    xy[:, 0] = (0.5, 0.3)                                                 # 코
    n = pp.normalize(xy, mask, origin="nose")
    assert np.allclose(n[:, 0], 0, atol=1e-6)                             # 코가 원점
    assert np.allclose(np.linalg.norm(n[:, pp.L_SH] - n[:, pp.R_SH], axis=-1), 1, atol=1e-5)   # 크기 기준은 어깨너비 그대로
    mask[:, 0] = False
    assert np.allclose(pp.normalize(xy, mask, origin="nose"), pp.normalize(xy, mask))   # 코가 안 보이면 어깨 기준으로 대신
    f = pp.build_features(xy, mask, 32, with_bones=True, origin="nose")
    assert f.shape == (32, 106 + 84 + 40 + 53) and f.dtype == np.float32


def test_hand_depth_scaled_by_hand_size_with_aspect():
    xy, mask = make_clip()
    z = np.zeros((20, 53), np.float32)
    xy[:, 32] = (0.5, 0.5)                                  # 오른손목
    xy[:, 32 + 9] = (0.5, 0.5 + 0.03 * 16 / 9)            # 가운뎃손가락 뿌리: 세로로 화면 너비 단위 0.03 떨어짐(16:9)
    z[:, 32 + 8] = -0.03                                    # 검지 끝이 손목보다 손 크기만큼 카메라 쪽
    d = pp.hand_depth(xy, z, mask, aspect=16 / 9)
    assert d.shape == (20, 42)
    assert np.allclose(d[:, 21 + 8], -1.0, atol=1e-5)       # 손 크기 단위로 -1
    assert np.allclose(d[:, 21 + 0], 0.0)                   # 손목 자신은 0
    mask[:, 32:53] = False
    assert (pp.hand_depth(xy, z, mask, 16 / 9)[:, 21:] == 0).all()   # 안 보인 손은 0
    f = pp.build_features(xy, mask, 32, z=z, with_hand_z=True)
    assert f.shape == (32, 106 + 84 + 42 + 53) and f.dtype == np.float32


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
