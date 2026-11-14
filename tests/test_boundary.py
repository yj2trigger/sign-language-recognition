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


# ---- 모델 단독(torch 필요) ----

def test_models_ignore_future_frames():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    x = torch.randn(1, 80, bd.N_FEAT)
    x2 = x.clone()
    x2[:, 50:] = torch.randn(1, 30, bd.N_FEAT)
    for arch in ("tcn", "gru"):
        m = bd.build_model(arch).eval()
        with torch.no_grad():
            y, y2 = m(x), m(x2)
        assert y.shape == (1, 80, 3)
        assert torch.allclose(y[:, :50], y2[:, :50], atol=1e-5) and not torch.allclose(y[:, 50:], y2[:, 50:])


def test_receptive_field_and_window():
    torch = pytest.importorskip("torch")
    tcn, gru = bd.build_model("tcn").eval(), bd.build_model("gru")
    rf = bd.receptive_field(tcn)
    assert rf == 61 and bd.model_window(tcn, 96) == 61 and bd.receptive_field(gru) is None and bd.model_window(gru, 96) == 96
    with pytest.raises(ValueError):
        bd.model_window(tcn, 40)
    x = torch.randn(1, 100, bd.N_FEAT)
    far, near = x.clone(), x.clone()
    far[:, 100 - rf - 1] += 5.0                            # 범위 밖 프레임은 마지막 출력에 영향이 없다
    near[:, 100 - rf] += 5.0                               # 범위 맨 앞 프레임은 영향이 있다
    with torch.no_grad():
        last = tcn(x)[0, -1]
        assert torch.allclose(tcn(far)[0, -1], last, atol=1e-5) and not torch.allclose(tcn(near)[0, -1], last, atol=1e-5)


def test_load_model_roundtrip_and_feature_check(tmp_path):
    torch = pytest.importorskip("torch")
    m = bd.build_model("tcn")
    ck = {"state": m.state_dict(), "arch": "tcn", "n_feat": bd.N_FEAT, "feat_names": bd.FEAT_NAMES}
    torch.save(ck, tmp_path / "m.pt")
    m2, saved = bd.load_model(str(tmp_path / "m.pt"))
    x = torch.randn(1, 20, bd.N_FEAT)
    with torch.no_grad():
        assert not m2.training and torch.allclose(m.eval()(x), m2(x)) and saved["arch"] == "tcn"
    torch.save(dict(ck, feat_names=bd.FEAT_NAMES[:-1]), tmp_path / "old.pt")
    with pytest.raises(ValueError):
        bd.load_model(str(tmp_path / "old.pt"))


# ---- 지표·기록 ----


def test_metrics():
    assert bd.iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    sc = bd.segment_scores([(0, 1, "word"), (5, 6, "word")], [(0, 1.1), (2, 3)])
    assert sc["precision"] == 0.5 and sc["recall"] == 0.5 and sc["mean_iou"] == pytest.approx(1 / 1.1)
    t = np.arange(10) / 10
    assert bd.frame_f1(t, [(0.2, 0.5)], [(0.2, 0.5)])["f1"] == 1.0
    assert bd.frame_f1(t, [], [(0.2, 0.5)])["f1"] == 0.0
    r = bd.evaluate_stream([(1.3, 0.0, 1.0, "word"), (2.5, 0.0, 1.0, "sentence_end")], t, [(0.0, 1.0, "word")])
    assert r["latency_s"] == [pytest.approx(0.3)] and r["sentence_end_latency_s"] == pytest.approx(1.5)


def test_append_trial_numbers_entries(tmp_path):
    p = tmp_path / "logs" / "sign-language" / "trials.md"
    assert bd.append_trial(str(p), "A", "h", "a", "r", "d", date="2026-11-11") == "T-001"
    assert bd.append_trial(str(p), "B", "h", "a", "r", "d") == "T-002"
    text = p.read_text(encoding="utf-8")
    assert text.startswith("## T-001 A (2026-11-11)\n") and "\n## T-002 B" in text and "- 가설: h" in text
    assert bd.next_trial_id("## T-041 x\n") == "T-042"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
