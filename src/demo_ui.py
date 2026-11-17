"""시연용 화면(realtime.py --ui): 왼쪽은 관절(선택: 카메라 위에 겹침), 오른쪽은 대본·인식 결과·맞음/틀림·기록·점수.
창에서 q 또는 Esc 를 누르면 끝낸다. --demo-video 를 주면 이 화면을 그대로 mp4 로 저장한다(시연 영상).
오른쪽 패널은 내용이 바뀔 때만 다시 그린다(한글은 PIL 로 그려서 매 프레임 그리면 느리다).
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import landmarks
from render_skeleton import draw_frame

FONT, BOLD = "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"
BG, CARD, DIM, TEXT = (24, 24, 28), (40, 40, 48), (150, 150, 160), (240, 240, 245)          # RGB(패널은 PIL 로 그림)
GOOD, BAD, ACCENT, OFF = (80, 200, 120), (235, 90, 90), (90, 170, 255), (80, 80, 90)
WIN = "Sign Language Demo"


def _font(path, size):
    return ImageFont.truetype(path if os.path.exists(path) else FONT, size) if os.path.exists(FONT) else ImageFont.load_default()


class DemoUI:
    def __init__(self, names, scorer=None, show_camera=False, video_path="", fps=15.0, height=540, panel_w=440, window=True):
        self.names, self.scorer, self.show_camera, self.fps, self.window = names or {}, scorer, show_camera, fps, window
        self.h, self.left_w, self.panel_w = height, int(round(height * 16 / 9)), panel_w
        self.f = {k: _font(BOLD if b else FONT, s) for k, (s, b) in
                  {"title": (22, True), "label": (15, False), "big": (40, True), "mid": (24, True), "body": (17, False)}.items()}
        self.last = None                                 # (이름, 맞음 여부 또는 None, 확률, 지연 ms)
        self.recent = []                                 # 대본 없이 쓸 때의 기록
        self.state, self.panel, self.dirty = (False, False), None, True
        self.writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.left_w + panel_w, height)) if video_path else None
        self.prev_tick, self.n_written = None, 0
        if window:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def name(self, w):
        return "지문자" if w == "FS" else ("(인식 실패)" if w is None else str(self.names.get(w, "번호 %s" % w)))

    # ---- 인식 결과(realtime.handle 이 부른다) ----
    def word(self, w, prob, lat_ms):
        sc = self.scorer
        ok = None
        if sc is not None and not sc.sentence and sc.hist and sc.hist[-1][0] == w:
            ok = sc.hist[-1][2]
        self.last = (self.name(w), ok, prob, lat_ms)
        self.recent = (self.recent + [self.name(w)])[-6:]
        self.dirty = True

    def fs(self, text, lat_ms):
        self.last = ("지문자: %s" % (text or "(인식 실패)"), None, None, lat_ms)
        self.recent = (self.recent + ["지문자"])[-6:]
        self.dirty = True

    def sentence_end(self):
        self.dirty = True

    # ---- 그리기 ----
    def _panel(self):
        img = Image.new("RGB", (self.panel_w, self.h), BG)
        d, f, x, W = ImageDraw.Draw(img), self.f, 20, self.panel_w - 40
        sc = self.scorer
        d.text((x, 14), "수어 인식 데모", font=f["title"], fill=TEXT)
        if sc is not None:
            d.text((self.panel_w - 20, 18), "세션 %s" % sc.session, font=f["label"], fill=DIM, anchor="ra")
        hands, active = self.state                       # 상태 표시등
        for i, (on, label, col) in enumerate(((hands, "손 인식", GOOD), (active, "구간 진행 중", ACCENT))):
            cx = x + 8 + i * 150
            d.ellipse((cx - 7, 56, cx + 7, 70), fill=col if on else OFF)
            d.text((cx + 14, 54), label, font=f["label"], fill=TEXT if on else DIM)
        y = 92
        if sc is not None and not sc.sentence:          # 단어 세션: 지금 할 단어
            d.text((x, y), "지금 할 단어  (%d/%d)" % (min(sc.i + 1, len(sc.exp)), len(sc.exp)), font=f["label"], fill=DIM)
            d.rounded_rectangle((x, y + 24, x + W, y + 92), 10, fill=CARD)
            d.text((x + W // 2, y + 58), self.name(sc.exp[sc.i]) if sc.i < len(sc.exp) else "끝!", font=f["big"], fill=TEXT, anchor="mm")
            y += 108
        elif sc is not None:                             # 문장 세션: 단어 순서와 지금까지 인식
            d.text((x, y), "이번 문장 (반복 %d)" % (len(sc.reps) + 1), font=f["label"], fill=DIM)
            d.rounded_rectangle((x, y + 24, x + W, y + 92), 10, fill=CARD)
            d.text((x + W // 2, y + 58), " → ".join(self.name(w) for w in sc.exp), font=f["mid"], fill=TEXT, anchor="mm")
            y += 104
            d.text((x, y), "지금까지: " + (" ".join(self.name(w) for w in sc.cur) or "-"), font=f["body"], fill=TEXT)
            y += 30
        d.text((x, y), "인식 결과", font=f["label"], fill=DIM)
        y += 24
        if self.last:
            nm, ok, prob, lat = self.last
            d.text((x, y), nm, font=f["mid"], fill=TEXT)
            if ok is not None:
                d.text((x + W, y + 4), "○ 맞음" if ok else "× 틀림", font=f["mid"], fill=GOOD if ok else BAD, anchor="ra")
            y += 36
            d.text((x, y), ("확신도 %.2f · " % prob if prob is not None else "") + "지연 %.1f초" % (lat / 1000), font=f["body"], fill=DIM)
        else:
            d.text((x, y), "-", font=f["mid"], fill=DIM)
            y += 36
        y += 34
        d.line((x, y, x + W, y), fill=CARD, width=2)
        y += 10
        d.text((x, y), "기록", font=f["label"], fill=DIM)
        y += 24
        rows = max(1, (self.h - 104 - y) // 26)         # 점수 상자와 겹치지 않는 만큼만
        if sc is not None and not sc.sentence:
            for k, (w, want, ok) in list(enumerate(sc.hist, 1))[-rows:]:
                d.text((x, y), "%d  %s" % (k, self.name(w)), font=f["body"], fill=TEXT)
                d.text((x + W, y), "○" if ok else "× → %s" % self.name(want), font=f["body"], fill=GOOD if ok else BAD, anchor="ra")
                y += 26
        elif sc is not None:
            for k, r in list(enumerate(sc.reps, 1))[-rows:]:
                d.text((x, y), "반복 %d  %s" % (k, " ".join(self.name(w) for w in r["pred"])), font=f["body"], fill=TEXT)
                d.text((x + W, y), "%d/%d" % (r["hit"], len(sc.exp)), font=f["body"], fill=GOOD if r["hit"] == len(sc.exp) else BAD, anchor="ra")
                y += 26
        else:
            for nm in self.recent[-rows:]:
                d.text((x, y), nm, font=f["body"], fill=TEXT)
                y += 26
        if sc is not None:                               # 점수(아래 고정)
            if sc.sentence:
                H, N = sum(r["hit"] for r in sc.reps), len(sc.reps) * len(sc.exp)
            else:
                H, N = sc.hit, sc.i
            d.rounded_rectangle((x, self.h - 92, x + W, self.h - 44), 10, fill=CARD)
            d.text((x + 16, self.h - 68), "점수", font=f["mid"], fill=DIM, anchor="lm")
            d.text((x + W - 16, self.h - 68), "%d / %d%s" % (H, N, " (%d%%)" % round(100 * H / N) if N else ""), font=f["mid"], fill=TEXT, anchor="rm")
        d.text((x, self.h - 30), "[q] 끝내기", font=f["label"], fill=DIM)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def compose(self, bgr, xy, mask, face, face_mask, active):
        """한 화면(왼쪽 관절/카메라 + 오른쪽 패널)을 만든다."""
        state = (bool(np.asarray(mask)[11:].any()), bool(active))
        if self.dirty or state != self.state or self.panel is None:
            self.state, self.dirty = state, False
            self.panel = self._panel()
        base = cv2.resize(landmarks.fit_16x9(bgr), (self.left_w, self.h)) if (self.show_camera and bgr is not None) else None
        left = draw_frame(xy, mask, face, face_mask, (self.left_w, self.h), base=base)
        return np.hstack([left, self.panel])

    def frame(self, bgr, xy, mask, face, face_mask, active, tick):
        """매 처리 프레임마다 부른다. False 를 돌려주면 끝내기(q·Esc·창 닫기)."""
        img = self.compose(bgr, xy, mask, face, face_mask, active)
        if self.writer is not None:                      # 버린 박자만큼 같은 화면을 넣어 실제 시간대로 재생되게
            n = 1 if self.prev_tick is None else max(1, int(round((tick - self.prev_tick) * self.fps)))
            for _ in range(n):
                self.writer.write(img)
            self.n_written += n
        self.prev_tick = tick
        if not self.window:
            return True
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27) and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.window:
            cv2.destroyAllWindows()
