"""실시간 루프 부품 확인(합성 데이터). 실행: .venv\\Scripts\\python tests\\test_realtime.py
MediaPipe·모델은 쓰지 않는다. 구간 탐지 규칙과 영상 시계가 계산대로 되는지만 본다."""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import realtime as rt


def _pose(up):
    xy, mask = np.zeros((53, 2), np.float32), np.zeros(53, bool)
    xy[5], xy[6] = (0.6, 0.4), (0.4, 0.4)                    # 어깨, 너비 0.2
    xy[9] = xy[10] = (0.5, 0.45 if up else 0.9)              # 손목: 어깨 아래 0.05(올림) / 0.5(내림, 기준 0.24 넘음)
    mask[[5, 6, 9, 10]] = True
    return xy, mask


def test_segmenter():
    seg = rt.HandRaiseSegmenter(on=2, off=4)
    pattern = [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   # 사이 1프레임 내림은 끊지 않는다, 1프레임 올림은 시작 아님
    hits = [(i, s) for i, up in enumerate(pattern) if (s := seg.update(i, *_pose(up)))]
    assert hits == [(12, (3, 8))], hits                      # 시작 = 올림 2번째(3), 끝 = 마지막 올림(8), 내림 4번째(12)에 확정


def test_video_clock():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for v in range(2):                                   # 30fps 10장짜리 영상 2개, 프레임 밝기 = 영상*100 + 번호
            p = os.path.join(d, "%d.avi" % v)
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
            for i in range(10):
                w.write(np.full((32, 32, 3), v * 100 + i * 5, np.uint8))
            w.release()
            paths.append(p)
        clk = rt.VideoClock(paths)
        val = lambda t: int(round(clk.frame_at(t).mean()))
        assert abs(val(0.0) - 0) <= 2 and abs(val(0.2) - 30) <= 2      # 0.2초 = 6번째 프레임(중간은 건너뜀)
        assert abs(val(1 / 3 + 0.11) - 115) <= 2                         # 첫 영상 끝(1/3초) 뒤 0.11초 = 둘째 영상 3번째
        assert clk.frame_at(1.0) is None                                # 두 영상 다 끝


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
