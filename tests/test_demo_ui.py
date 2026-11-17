"""시연 화면 그리기 확인(창 없이). MediaPipe·모델 불필요."""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import realtime as rt
from demo_ui import DemoUI


def _pose():
    xy = np.random.default_rng(0).uniform(0.3, 0.7, (53, 2)).astype(np.float32)
    return xy, np.ones(53, bool)


def test_compose_size_and_panel_updates():
    sc = rt.LiveScorer("A1", [5, 6], sentence=False, names={5: "가", 6: "나"})
    ui = DemoUI({5: "가", 6: "나"}, sc, window=False)
    xy, mask = _pose()
    img = ui.compose(None, xy, mask, None, None, active=False)
    assert img.shape == (540, 960 + 440, 3) and img[:, :960].any()          # 왼쪽 관절이 그려짐
    before = ui.panel.copy()
    sc.on_token(5)
    ui.word(5, 0.9, 300)
    assert ui.last[1] is True                                               # 대본과 맞음
    ui.compose(None, xy, mask, None, None, active=True)
    assert not np.array_equal(before, ui.panel)                             # 패널이 다시 그려짐


def test_sentence_panel_and_camera_overlay():
    sc = rt.LiveScorer("S1", [5, 6], sentence=True, names={5: "가", 6: "나"})
    ui = DemoUI({5: "가", 6: "나"}, sc, show_camera=True, window=False)
    xy, mask = _pose()
    cam = np.full((480, 640, 3), 120, np.uint8)                             # 4:3 카메라 → 16:9 틀에 맞춰 깔림
    sc.on_token(5), sc.on_token(6), sc.on_sentence_end()
    ui.sentence_end()
    img = ui.compose(cam, xy, mask, None, None, active=False)
    assert img.shape[0] == 540 and img[270, 480].sum() > 0


def test_demo_video_keeps_real_time():
    xy, mask = _pose()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "demo.mp4")
        ui = DemoUI({}, None, video_path=p, fps=15.0, height=180, panel_w=200, window=False)
        for tick in (0.0, 1 / 15, 4 / 15):                                  # 2박자 버림 → 같은 화면을 더 넣는다
            assert ui.frame(None, xy, mask, None, None, False, tick)
        ui.close()
        assert ui.n_written == 1 + 1 + 3
        cap = cv2.VideoCapture(p)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320 + 200
        cap.release()
