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


def test_concat_default_gaps_never_reach_sentence_pause():
    def clip(n, s, e):
        return {"t": np.arange(n) / FPS, "xy": np.zeros((n, 53, 2), np.float32), "mask": np.zeros((n, 53), bool),
                "face": np.zeros((n, 33, 2), np.float32), "face_mask": np.zeros((n, 33), bool),
                "seg_start": s, "seg_end": e, "signer": 1}
    rng = np.random.default_rng(1)
    for _ in range(50):
        segs = db.concat_sentence([clip(60, 1.5, 2.8) for _ in range(4)], rng)["segments"]
        gaps = [b[0] - a[1] for a, b in zip(segs, segs[1:])]
        assert max(gaps) < bd.DetectorConfig().pause_s     # 문장 안 간격이 문장 끝 멈춤(0.8초)에 닿지 않는다


@pytest.mark.skipif(sys.platform != "win32", reason="드라이브 문자는 Windows 에만 있다")


# ---- update() 입력 처리 ----


def _blank():
    return np.zeros((53, 2), np.float32), np.zeros(53, bool)


def test_update_empty_frames_returns_none():
    det = bd.BoundaryDetector()
    assert all(det.update(i / FPS, *_blank()) is None for i in range(30))
    assert det.flush() == []


def test_update_rejects_wrong_shapes_and_bad_time():
    det = bd.BoundaryDetector()
    with pytest.raises(ValueError):
        det.update(0.0, np.zeros((42, 2)), np.zeros(42, bool))
    with pytest.raises(ValueError):
        det.update(float("nan"), *_blank())


def test_update_ignores_repeated_time_and_nan_points():
    det = bd.BoundaryDetector()
    xy, mask = _blank()
    xy[11:32] = np.nan
    mask[11:32] = True                   # NaN 좌표에 마스크가 켜져 와도 꺼진 것으로 본다
    assert det.update(1.0, xy, mask) is None
    assert det.fx.last["present"] is False
    assert det.update(1.0, xy, mask) is None and det.update(0.5, xy, mask) is None   # 같은·이전 시각은 무시


# ---- 규칙 모드 탐지 ----


def _run(s, det=None):
    return bd.run_stream(det or bd.BoundaryDetector(), s["t"], s["xy"], s["mask"], s["face"], s["face_mask"])


def test_rule_detector_finds_each_word_and_one_sentence_end():
    rng = np.random.default_rng(1)
    for _ in range(10):
        s = db.synthetic_sentence(rng, n_words=3)
        ev = _run(s)
        r = bd.evaluate_stream(ev, s["t"], s["segments"])
        assert r["segment"]["f1"] == 1.0 and r["segment"]["mean_iou"] > 0.9, (ev, s["segments"])
        assert r["n_sentence_end"] == 1
        sent = [e for e in ev if e[3] == "sentence_end"][0]
        assert abs(sent[1] - s["segments"][0][0]) < 0.15 and abs(sent[2] - s["segments"][-1][1]) < 0.15
        assert max(r["latency_s"]) <= 0.34 + 1e-6                    # 끝 확정은 내림 4프레임 또는 멈춤 0.25초 + 1프레임 안


def test_hand_dropout_shorter_than_off_frames_does_not_split():
    rng = np.random.default_rng(2)
    s = db.synthetic_sentence(rng, n_words=1)
    a, b, _ = s["segments"][0]
    mid = np.where((s["t"] > (a + b) / 2))[0][:2]                  # 단어 가운데 2프레임 손이 화면 밖(마스크 0)
    s["mask"][mid, 9:53] = False
    words = [e for e in _run(s) if e[3] != "sentence_end"]
    assert len(words) == 1


def test_hand_leaving_frame_ends_segment():
    rng = np.random.default_rng(3)
    s = db.synthetic_sentence(rng, n_words=1)
    a, b, _ = s["segments"][0]
    cut = np.where(s["t"] > a + 0.4)[0][0]
    s["mask"][cut:, 9:53] = False                                  # 손이 나가고 다시 안 들어온다
    words = [e for e in _run(s) if e[3] != "sentence_end"]
    assert len(words) == 1 and abs(words[0][2] - s["t"][cut - 1]) < 1e-6


def test_fingerspelling_is_labelled():
    rng = np.random.default_rng(4)
    s = db.synthetic_sentence(rng, n_words=3, fingerspell_prob=0.5)
    ev = [e for e in _run(s) if e[3] != "sentence_end"]
    m = bd.match_segments([e[1:] for e in ev], s["segments"])
    assert len(m) == 3 and all(ev[i][3] == s["segments"][j][2] for i, j, _ in m)
    assert "fingerspelling" in [k for *_, k in s["segments"]]


def test_sentence_end_waits_for_pause():
    rng = np.random.default_rng(5)
    s = db.synthetic_sentence(rng, n_words=2)
    lat = bd.evaluate_stream(_run(s), s["t"], s["segments"])["sentence_end_latency_s"]
    cfg = bd.DetectorConfig()
    assert cfg.pause_s - bd.EPS <= lat <= cfg.max_pause_s + 1 / FPS   # 12프레임 멈춤은 0.7999.. 로 계산된다


def test_face_cue_reflects_brow_raise():
    for end_face, lo, hi in ((True, 0.99, 1.0), (False, 0.0, 0.01)):
        s = db.synthetic_sentence(np.random.default_rng(6), n_words=2, end_face=end_face)
        det = bd.BoundaryDetector(None, bd.DetectorConfig(face_cue=True))
        cues = []
        det._check_sentence_end = lambda t, d=det: cues.append(d.sentence_cues(t)) or None   # 판정 직전 단서만 본다
        _run(s, det)
        assert cues and lo <= cues[-1][1] <= hi, cues[-1]


def test_face_missing_is_tolerated():
    assert bd.face_signals(None, None) is None
    assert bd.face_signals(np.zeros((33, 2)), np.zeros(33, bool)) is None
    s = db.synthetic_sentence(np.random.default_rng(7), n_words=2)
    ev = bd.run_stream(bd.BoundaryDetector(), s["t"], s["xy"], s["mask"])   # 얼굴 없이
    assert [e[3] for e in ev].count("sentence_end") == 1


def test_min_length_is_same_at_every_offset():
    det = bd.BoundaryDetector()
    kept = []
    for k in range(30):                    # 3프레임(0.2초) 구간: 시작 위치에 따라 0.2 / 0.19999.. 로 계산된다
        det.start = k / FPS
        kept.append(det._close((k + 3) / FPS) is not None)
    assert all(kept)


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


# ---- 모델(torch 필요) ----


def test_model_is_causal_and_runs_in_detector(tmp_path):
    torch = pytest.importorskip("torch")
    for arch in ("tcn", "gru"):
        torch.manual_seed(0)
        m = bd.build_model(arch).eval()
        x = torch.randn(1, 40, bd.N_FEAT)
        y = x.clone()
        y[:, 25:] = torch.randn(1, 15, bd.N_FEAT)                  # 미래 프레임만 바꾼다
        with torch.no_grad():
            assert torch.allclose(m(x)[:, :25], m(y)[:, :25], atol=1e-5)
    p = tmp_path / "b.pt"
    torch.save({"state": m.state_dict(), "arch": "gru", "n_feat": bd.N_FEAT, "feat_names": bd.FEAT_NAMES,
                "detector": {"act_thr": 0.6}}, p)
    det = bd.BoundaryDetector(str(p))
    assert det.cfg.act_thr == 0.6
    s = db.synthetic_sentence(np.random.default_rng(8), n_words=2)
    bd.run_stream(det, s["t"], s["xy"], s["mask"])                 # 학습 안 한 모델이라 결과는 보지 않고 돌기만 확인
    assert det.last_probs is not None and abs(det.last_probs.sum() - 1) < 1e-5


def _scripted(probs):
    """프레임마다 정해 둔 (O, B, I) 확률을 내는 가짜 모델."""
    torch = pytest.importorskip("torch")

    class Scripted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.i = 0

        def forward(self, x):
            out = torch.zeros(1, x.shape[1], 3)
            out[0, -1] = torch.log(torch.tensor(probs[self.i]) + 1e-6)
            self.i += 1
            return out
    return Scripted()


def _hands_up():
    xy, mask = _blank()
    xy[5], xy[6] = (0.6, 0.4), (0.4, 0.4)
    xy[9] = xy[10] = (0.5, 0.45)
    mask[[5, 6, 9, 10]] = True                   # 손 21점은 없어도 포즈 손목이 보이면 '보임'
    return xy, mask


def _run_scripted(probs, frame=_hands_up):
    det = bd.BoundaryDetector(_scripted(probs))
    ev = [r for k in range(len(probs)) if (r := det.update(k / FPS, *frame()))]
    return [e for e in ev + det.flush() if e[2] != "sentence_end"]


def test_model_ignored_when_no_hands_visible():
    I = [0.05, 0.05, 0.9]
    assert _run_scripted([I] * 30, frame=_blank) == []       # 사람이 화면에 없으면 모델이 활동이라 해도 구간이 없다
    seen = iter([True] * 10 + [False] * 20)
    words = _run_scripted([I] * 30, frame=lambda: _hands_up() if next(seen) else _blank())
    assert len(words) == 1 and round(words[0][1] * FPS) == 9, words   # 사라지면 마지막으로 보인 프레임에서 닫힌다


def test_lingering_begin_does_not_move_start():
    O, B, I = [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]
    words = _run_scripted([O] * 5 + [B] * 6 + [I] * 19 + [O] * 10)    # 단어 첫 6프레임 내내 B 가 높다
    assert len(words) == 1 and abs(words[0][0] - 5 / FPS) < 1e-6, words  # 예전에는 시작이 3프레임 밀렸다


def test_begin_after_inside_splits_words():
    O, B, I = [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]
    words = _run_scripted([O] * 5 + [B] + [I] * 9 + [B] * 2 + [I] * 9 + [O] * 10)   # 쉬지 않고 이어진 두 단어
    assert [round(w[0] * FPS) for w in words] == [5, 15], words


def test_detector_puts_model_in_eval_mode():
    pytest.importorskip("torch")
    m = bd.build_model("tcn").train()
    assert bd.BoundaryDetector(m).model.training is False


def test_window_matches_training():
    torch = pytest.importorskip("torch")
    tcn, gru = bd.build_model("tcn").eval(), bd.build_model("gru").eval()
    rf = bd.receptive_field(tcn)
    x = torch.randn(1, 100, bd.N_FEAT)
    with torch.no_grad():
        base = tcn(x)[0, -1]
        for back, changes in ((rf, True), (rf + 1, False)):     # 마지막에서 rf 번째 프레임까지는 출력을 바꾸고, 그 앞은 못 바꾼다
            y = x.clone()
            y[0, -back] += 5.0
            assert (not torch.allclose(tcn(y)[0, -1], base, atol=1e-6)) == changes, back
    assert rf == 61 and bd.model_window(tcn, 96) == 61
    assert bd.receptive_field(gru) is None and bd.model_window(gru, 96) == 96
    with pytest.raises(ValueError):
        bd.model_window(tcn, 40)                                # 학습 조각이 범위보다 짧다
    with pytest.raises(ValueError):
        bd.BoundaryDetector(tcn, bd.DetectorConfig(window=30))  # 실시간 창이 범위보다 짧다


def test_saved_gru_window_must_equal_train_crop(tmp_path):
    torch = pytest.importorskip("torch")
    ck = {"state": bd.build_model("gru").state_dict(), "arch": "gru", "n_feat": bd.N_FEAT,
          "feat_names": bd.FEAT_NAMES, "train_crop": 96}
    p = tmp_path / "g.pt"
    torch.save(ck | {"detector": {"window": 96}}, p)
    assert bd.BoundaryDetector(str(p)).feats.maxlen == 96
    torch.save(ck | {"detector": {"window": 61}}, p)
    with pytest.raises(ValueError):
        bd.BoundaryDetector(str(p))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
