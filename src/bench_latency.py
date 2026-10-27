"""지연 분리 측정: MediaPipe 프레임당 추출, 특징 계산, 모델 1윈도우 추론을 따로 재고 합친다.
사용: python src/bench_latency.py <정면 영상 mp4> [프레임 수=160]
같은 기기·전원 조건에서 예열 후 반복 측정하고 중앙값과 95번째 백분위를 낸다. 다른 프로그램이 CPU 를 쓰는 중이면 값이 나빠지므로 측정 조건을 함께 기록할 것.
"""
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

import models
import preprocess as pp
from winpower import keep_awake, no_power_throttling


def stats(ms):
    a = np.asarray(ms)
    return "중앙값 %.1f ms, p95 %.1f ms" % (np.percentile(a, 50), np.percentile(a, 95))


def mediapipe_ms(frames, complexity):
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity) as h:
        for f in frames[:10]:
            h.process(f)                                  # 예열
        out = []
        for f in frames[10:]:
            t = time.perf_counter()
            h.process(f)
            out.append((time.perf_counter() - t) * 1000)
    return out


def model_ms(name, threads, reps=300):
    torch.set_num_threads(threads)
    m = models.build(name, 243, 100).eval()
    x = torch.randn(1, 64, 243)
    out = []
    with torch.no_grad():
        for _ in range(30):
            m(x)
        for _ in range(reps):
            t = time.perf_counter()
            m(x)
            out.append((time.perf_counter() - t) * 1000)
    return out


def features_ms(reps=200):
    xy = np.random.rand(64, 53, 2).astype(np.float32) * 0.2 + 0.4
    xy[:, pp.L_SH], xy[:, pp.R_SH] = (0.6, 0.5), (0.4, 0.5)
    mask = np.ones((64, 53), bool)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        pp.build_features(xy, mask, 64)
        out.append((time.perf_counter() - t) * 1000)
    return out


def main():
    # 창 없는 백그라운드 실행은 Windows 효율 모드로 느려져 지연이 1.5배까지 부풀었다(T-039, C-15). 측정 전에 빼 달라고 요청한다.
    print("효율 모드 제외 %s, 절전 방지 %s" % (no_power_throttling(), keep_awake()))
    video = sys.argv[1].replace("\\", "/")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print("영상 %dx%d, 측정 프레임 %d, 스레드 기본값 %d" % (frames[0].shape[1], frames[0].shape[0], len(frames) - 10, torch.get_num_threads()))
    mp_res = {}
    for cx in (0, 1):
        mp_res[cx] = mediapipe_ms(frames, cx)
        print("MediaPipe complexity %d (프레임당): %s" % (cx, stats(mp_res[cx])))
    f = features_ms()
    print("특징 계산 (64프레임 창 1개): %s" % stats(f))
    for name in ("mlp", "gru", "tcn"):
        for th in (1, 4):
            print("모델 %s, 스레드 %d (창 1개 추론): %s" % (name, th, stats(model_ms(name, th))))
    # 한 프레임이 도착해서 결과가 나오기까지(최악: 그 프레임에서 창 추론이 함께 일어남)
    for cx in (0, 1):
        for name in ("gru", "tcn"):
            tot = np.array(mp_res[cx]) + np.median(f) + np.median(model_ms(name, 4, 100))
            print("합계 추정 complexity %d + 특징 + %s: 중앙값 %.1f ms, p95 %.1f ms → 처리율 %.1f fps" % (
                cx, name, np.percentile(tot, 50), np.percentile(tot, 95), 1000 / np.percentile(tot, 50)))


if __name__ == "__main__":
    main()
