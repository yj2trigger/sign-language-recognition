"""관절 녹화 저장(realtime.save_recording)과 막대 인형 그리기(render_skeleton) 확인(합성 데이터, MediaPipe 불필요)."""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import realtime as rt
import render_skeleton as rs


def _pose(n=15):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.2, 0.8, (n, 53, 2)).astype(np.float32)
    mask = np.ones((n, 53), bool)
    mask[:, 32:] = False                                  # 오른손은 안 보임
    face = rng.uniform(0.3, 0.6, (n, 33, 2)).astype(np.float32)
    return xy, mask, face, np.ones((n, 33), bool)


def test_save_recording_roundtrip():
    xy, mask, face, fm = _pose(5)
    rec = {"t": [0.0, 1 / 15, 3 / 15, 4 / 15, 5 / 15], "xy": list(xy), "mask": list(mask), "face": list(face),
           "face_mask": list(fm), "wh": (1920, 1080),
           "ev": [("word", 0.1, 0.3, 0.4, 1345, 0.9, ""), ("fingerspelling", 0.1, 0.3, 0.4, -1, 0.0, "가나"),
                  ("sentence_end", 0.0, 0.3, 0.4, -1, 0.0, "")]}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "rec.npz")                 # 없는 폴더도 만든다
        rt.save_recording(rec, p, 15.0)
        with np.load(p) as z:
            assert z["xy"].shape == (5, 53, 2) and z["face"].shape == (5, 33, 2) and z["mask"].dtype == bool
            assert z["t"].tolist() == rec["t"] and z["wh"].tolist() == [1920, 1080]      # 버린 프레임(2/15)도 시각으로 남는다
            assert z["ev_kind"].tolist() == ["word", "fingerspelling", "sentence_end"]
            assert z["ev_word"].tolist() == [1345, -1, -1] and z["ev_text"].tolist() == ["", "가나", ""]
            assert not any(z[k].ndim == 4 for k in z.files)                              # 영상 화면(이미지 배열)은 없다
        t, xy2, m2, f2, fm2, aspect, ev = rs.load_recording(p)
        assert np.array_equal(xy2, xy) and abs(aspect - 16 / 9) < 1e-6 and len(ev) == 3


def test_render_frames_and_size():
    xy, mask, face, fm = _pose(15)
    t = np.arange(15) / 15.0
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.mp4")
        n = rs.render(t, xy, mask, face, fm, p, scale=180, speed=1.0, loops=1)
        assert n == 15
        cap = cv2.VideoCapture(p)
        ok, f = cap.read()
        assert ok and f.shape == (180, 320, 3) and f.any()                            # 16:9, 검지 않은 화면
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
        cap.release()
        p2 = os.path.join(d, "b.mp4")                                                 # 0.5배속 × 2번 + 멈춤
        assert rs.render(t, xy, mask, None, None, p2, scale=180, speed=0.5, loops=2, hold_s=0.6) == 2 * (int(round(14 / 15 / 0.5 * 15)) + 1 + 9)


def test_irregular_times_play_in_real_time():
    xy, mask, face, fm = _pose(4)
    t = np.array([0.0, 1 / 15, 4 / 15, 5 / 15])           # 2~3번째 사이 2프레임을 버렸다
    with tempfile.TemporaryDirectory() as d:
        assert rs.render(t, xy, mask, face, fm, os.path.join(d, "c.mp4"), scale=90) == 6   # 시각대로 6프레임


if __name__ == "__main__":
    test_save_recording_roundtrip()
    test_render_frames_and_size()
    test_irregular_times_play_in_real_time()
    print("ok")
