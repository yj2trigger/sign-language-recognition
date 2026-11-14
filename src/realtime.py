"""실시간 루프: 카메라(또는 실제 시계로 재생하는 영상 파일) -> MediaPipe(목표 fps) -> 버퍼 -> 구간 탐지 -> 단어 분류 -> 출력.

사용: python src/realtime.py <모델.pt> (--camera 0 | 영상.mp4 [영상2.mp4 ...]) [--fps 15] [--names 단어.csv]
영상을 여러 개 주면 이어서 재생한다(단어를 연달아 하는 사람 흉내).
영상 파일도 실제 시계에 맞춰 흘러간다: 처리가 늦으면 그동안 지나간 프레임은 버린다(실제 카메라와 같다).
구간 탐지기는 HandRaiseSegmenter(손 높이 규칙)가 기본이며 update(t, xy, mask) -> (시작, 끝[, 종류]) 또는 None 인 것이면 갈아 끼울 수 있다
(boundary.BoundaryDetector 는 종류 'word'·'fingerspelling'·'sentence_end' 를 붙인다. 문장 끝은 분류하지 않고 표시만 한다).
--boundary 로 BoundaryDetector 를 고른다. 탐지기의 uses_face 가 참이면(표정 단서를 켰을 때) 얼굴 33점도 뽑아 update(..., face=, face_mask=) 로 넘긴다.
손 높이 규칙은 루프·지연 점검용이다. 구간 탐지 성능 비교에는 쓰지 않는다(BIO 구간 탐지기로 바꿀 자리).
지연 = 단어 결과가 나온 시각 - 탐지기가 정한 단어 끝 시각(영상 시계 기준). 추출 등 다른 작업과 CPU 를 나눠 쓰면 값이 부푼다.
"""
import argparse
import csv
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

import landmarks
import models
import preprocess as pp

L_WRIST, R_WRIST = 9, 10            # 53점 중 몸 쪽 손목(landmarks.POSE_IDX 순서)


class VideoClock:
    """영상 파일들을 이어서, 실제 시계로 재생하는 카메라처럼 다룬다. frame_at(t): 영상 시각 t 에 화면에 있을 프레임(끝나면 None)."""

    def __init__(self, paths):
        self.paths, self.cap, self.base, self.n, self.fps, self.frame = list(paths), None, 0.0, 0, 30.0, None
        self._open()

    def _open(self):
        if self.cap is not None:
            self.base += self.n / self.fps
            self.cap.release()
        if not self.paths:
            self.cap = None
            return
        self.cap = cv2.VideoCapture(self.paths.pop(0))
        self.fps, self.n, self.read = self.cap.get(cv2.CAP_PROP_FPS) or 30.0, 0, 0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame_at(self, t):
        while self.cap is not None:
            want = int((t - self.base) * self.fps)       # 이 영상에서 t 에 보일 프레임 번호
            if want >= self.n:
                self._open()
                continue
            while self.read <= want:                     # 지나간 프레임은 디코딩만 하고 버린다
                ok, self.frame = self.cap.read()
                self.read += 1
                if not ok:
                    self.n = self.read - 1
                    break
            else:
                return self.frame
        return None


class CameraClock:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)

    def frame_at(self, t):
        ok, f = self.cap.read()
        return f if ok else None


class HandRaiseSegmenter:
    """손목이 어깨선에서 어깨너비 rise 배 아래보다 높으면 '손 올림'. 올림이 on 프레임 이어지면 시작,
    내림이 off 프레임 이어지면 끝(마지막 올림 프레임)으로 본다. off=4 는 15fps 에서 약 0.27초 대기."""

    def __init__(self, rise=1.2, on=2, off=4):
        self.rise, self.on, self.off = rise, on, off
        self.run_up = self.run_down = 0
        self.start = self.last_up = None

    def raised(self, xy, mask):
        if not (mask[pp.L_SH] and mask[pp.R_SH]):
            return False
        sh_y = (xy[pp.L_SH, 1] + xy[pp.R_SH, 1]) / 2
        width = np.linalg.norm(xy[pp.L_SH] - xy[pp.R_SH])
        return any(mask[w] and xy[w, 1] < sh_y + self.rise * width for w in (L_WRIST, R_WRIST))

    def update(self, t, xy, mask):
        if self.raised(xy, mask):
            self.run_up, self.run_down, self.last_up = self.run_up + 1, 0, t
            if self.start is None and self.run_up == self.on:
                self.start = t
            return None
        self.run_up, self.run_down = 0, self.run_down + 1
        if self.start is not None and self.run_down == self.off:
            seg, self.start = (self.start, self.last_up), None
            return seg
        return None


def load_model(path):
    ck = torch.load(path, weights_only=False)
    model = models.build(ck["model"], ck["n_feat"], len(ck["classes"]))
    model.load_state_dict(ck["state"])
    return model.eval(), ck["classes"], ck["config"]


def classify(model, classes, cfg, buf, start, end, margin=0.5):
    """버퍼에서 [start - margin, end + margin] 프레임(학습의 라벨 ±0.5초와 같게, 끝쪽은 지금까지 온 만큼)을 분류."""
    sel = [b for b in buf if start - margin <= b[0] <= end + margin]
    face = cfg.get("face", False)
    xy = np.stack([np.concatenate([b[1], b[3]]) if face else b[1] for b in sel])
    mask = np.stack([np.concatenate([b[2], b[4]]) if face else b[2] for b in sel])
    f = pp.build_features(xy, mask, cfg["length"], with_hand_rel=cfg["hand_rel"], with_angles=cfg["angles"],
                          with_traj=cfg["traj"], with_bones=cfg["bones"], origin=cfg["origin"], with_face=face)
    if f is None:
        return None, 0.0
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(f[None])), 1)[0]
    k = int(p.argmax())
    return classes[k], float(p[k])


def feed_segmenter(segmenter, t, r, with_face):
    """landmarks.process_frame 결과 r 을 구간 탐지기에 넣는다. with_face 이면 r 의 얼굴(r[4], r[5])도 넘긴다."""
    if with_face:
        return segmenter.update(t, r[0], r[1], face=r[4], face_mask=r[5])
    return segmenter.update(t, r[0], r[1])


def make_segmenter(spec):
    """'' -> HandRaiseSegmenter, 'rule' -> BoundaryDetector 규칙 모드, 그 밖 -> BoundaryDetector(그 .pt)."""
    if not spec:
        return HandRaiseSegmenter()
    import boundary
    return boundary.BoundaryDetector(None if spec == "rule" else spec)


def run(source, model, classes, cfg, fps=15.0, segmenter=None, names=None, keep_s=10.0):
    """루프 본체. 반환: (결과 [(영상 시각 끝, 단어, 확률, 지연 ms)], 프레임 처리 ms 목록, 버린 프레임 수)."""
    segmenter = segmenter or HandRaiseSegmenter()
    seg_face = getattr(segmenter, "uses_face", False)   # BoundaryDetector 는 얼굴을 문장 끝 보조 단서로 쓴다
    face = cfg.get("face", False) or seg_face            # 분류기·탐지기 중 하나라도 쓰면 얼굴을 뽑는다(버퍼 b[3], b[4])
    buf, out, proc_ms, dropped = [], [], [], 0
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as h:
        t0, k = time.perf_counter(), 0
        while True:
            tick = k / fps
            now = time.perf_counter() - t0
            if now < tick:
                time.sleep(tick - now)
            elif now - tick >= 1 / fps:                  # 처리가 늦어 지나간 박자는 건너뛴다(그 프레임은 버림)
                skip = int((now - tick) * fps)
                dropped, k, tick = dropped + skip, k + skip, (k + skip) / fps
            frame = source.frame_at(tick)
            if frame is None:
                break
            c = time.perf_counter()
            r = landmarks.process_frame(h, frame, with_face=face)
            proc_ms.append((time.perf_counter() - c) * 1000)
            buf.append((tick, r[0], r[1]) + ((r[4], r[5]) if face else ()))
            while buf and buf[0][0] < tick - keep_s:
                buf.pop(0)
            seg = feed_segmenter(segmenter, tick, r, seg_face)
            if seg and len(seg) == 3 and seg[2] == "sentence_end":     # boundary.BoundaryDetector 의 문장 끝 알림
                print("%7.2f초  [문장 끝] %.2f~%.2f초" % (tick, seg[0], seg[1]), flush=True)
            elif seg:
                word, prob = classify(model, classes, cfg, buf, seg[0], seg[1])
                lat = (time.perf_counter() - t0 - seg[1]) * 1000
                out.append((seg[1], word, prob, lat))
                label = names.get(word, word) if names else word
                print("%7.2f초  %s (%.2f)  구간 %.2f~%.2f초  지연 %.0f ms" % (tick, label, prob, seg[0], seg[1], lat), flush=True)
            k += 1
    return out, proc_ms, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--names", default="", help="단어 번호,이름 CSV(없으면 번호로 출력)")
    ap.add_argument("--boundary", default="", help="구간 탐지기: 비우면 손 높이 규칙, rule 이면 BoundaryDetector 규칙 모드, "
                                                   "그 밖은 train_boundary.py 가 저장한 .pt")
    a = ap.parse_args()
    model, classes, cfg = load_model(a.model)
    segmenter = make_segmenter(a.boundary)
    if a.boundary:
        print("구간 탐지기: %s, 입력 창 %d프레임" % ("규칙" if segmenter.model is None else a.boundary, segmenter.cfg.window), flush=True)
    names = {}
    if a.names:
        with open(a.names, encoding="utf-8-sig") as f:
            names = {int(r[0]): r[1] for r in csv.reader(f) if r and r[0].isdigit()}
    source = CameraClock(a.camera) if a.camera is not None else VideoClock(a.videos)
    out, proc, dropped = run(source, model, classes, cfg, a.fps, segmenter=segmenter, names=names)
    lat = [o[3] for o in out]
    print("단어 %d개, 프레임 %d장 처리(버림 %d), MediaPipe 중앙값 %.1f ms·p95 %.1f ms" % (
        len(out), len(proc), dropped, np.median(proc), np.percentile(proc, 95)))
    if lat:
        print("지연(단어 끝 -> 결과) 중앙값 %.0f ms, p95 %.0f ms" % (np.median(lat), np.percentile(lat, 95)))


if __name__ == "__main__":
    main()
