"""재현용 도구 확인: 문장 끝 지표(boundary.sentence_end_stats), 문장 영상 추출(extract_sentences.py), 평가(eval_boundary.py).
실행: python -m pytest tests/test_eval_tools.py  (MediaPipe·실제 데이터 없음)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import boundary as bd
import dataset_bio as db
import eval_boundary as eb
import extract_sentences as es


def test_sentence_end_stats_counts_only_alerts_inside_video():
    true = [(0.5, 1.0, "word"), (1.2, 2.0, "word")]
    words = [(1.1, 0.5, 1.0, "word"), (2.3, 1.2, 2.0, "word")]
    assert bd.sentence_end_stats(words, true) == {"in_video": False, "mid_false": 0, "latency_s": None}
    ev = words[:1] + [(1.9, 0.5, 1.0, "sentence_end")] + words[1:] + [(2.9, 1.2, 2.0, "sentence_end")]
    got = bd.sentence_end_stats(ev, true)
    assert got["in_video"] and got["mid_false"] == 1 and got["latency_s"] == pytest.approx(-0.1)   # 첫 알림이 문장 중간
    with pytest.raises(ValueError):
        bd.sentence_end_stats([], [])


def test_flush_alert_is_not_counted_as_in_video():
    s = db.synthetic_sentence(np.random.default_rng(3), n_words=2)
    keep = s["t"] <= s["segments"][-1][1] + 0.3            # 마지막 단어 직후에 영상이 끝난다(문장 끝 멈춤 전)
    args = (s["t"][keep], s["xy"][keep], s["mask"][keep])
    det = bd.BoundaryDetector()
    assert any(e[3] == "sentence_end" for e in bd.run_stream(det, *args))                  # flush 는 문장 끝을 내 준다
    assert not bd.sentence_end_stats(bd.run_stream(det, *args, flush=False), s["segments"])["in_video"]


def test_label_name_and_saved_arrays_have_no_gloss():
    base = "NIA_SL_SEN0001_REAL03_F"
    assert es.label_name(base) == "morpheme/03/NIA_SL_SEN0001_REAL03_F_morpheme.json"
    n = 5
    res = {"t": np.arange(n) / 15.0, "xy": np.zeros((n, 53, 2), np.float32), "mask": np.zeros((n, 53), bool),
           "face": np.zeros((n, 33, 2), np.float32), "face_mask": np.zeros((n, 33), bool)}
    lab = {"metaData": {"duration": 4.2},
           "data": [{"start": 0.5, "end": 1.0, "attributes": [{"name": "단어가"}]},
                    {"start": 1.2, "end": 1.9, "attributes": [{"name": "단어나"}]}]}
    out = es.sentence_npz(res, lab)
    assert sorted(out) == ["duration", "face", "face_mask", "mask", "seg", "t", "xy"]
    assert np.allclose(out["seg"], [[0.5, 1.0], [1.2, 1.9]]) and out["duration"] == 4.2
    assert not any(isinstance(v, str) or (isinstance(v, np.ndarray) and v.dtype.kind in "US") for v in out.values())


def test_val_split_matches_train_boundary():
    real = list(range(196 * 10 + 7))
    got = eb.val_split(real, 0)
    idx = np.random.default_rng(0).permutation(len(real))              # train_boundary.py 의 실제 문장 분할
    assert got == [real[i] for i in idx[:max(1, int(len(real) * 0.1))]]


def test_evaluate_reports_segment_and_sentence_end_metrics():
    rng = np.random.default_rng(5)
    data = [db.synthetic_sentence(rng, n_words=3) for _ in range(3)]
    r = eb.evaluate(bd.BoundaryDetector(), data)
    assert r["n_videos"] == 3 and r["segment_f1"] > 0.9
    assert r["sentence_end_in_video_rate"] == 1.0 and r["sentence_end_mid_false"] == 0 and r["sentence_end_latency_median_s"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
