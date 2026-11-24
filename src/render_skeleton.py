"""관절 좌표(몸 11·손 42·얼굴 33)를 검은 배경의 막대 인형 mp4 로 그린다. 실제 영상 화면은 쓰지 않는다.

사용:
  녹화 다시 보기: python src/render_skeleton.py <녹화.npz> --out <mp4> [--names 번호,이름 CSV]
      realtime.py --record 로 저장한 npz 를 그린다. 그때 낸 인식 결과(단어·지문자·문장 끝)를 화면 아래에 띄운다.
  연습 자료: python src/render_skeleton.py --words 1345,2318 --features <특징 폴더>... --out-dir <폴더> [--names CSV]
      학습 데이터에서 그 단어의 클립을 골라(수행자 5·7 은 라벨 오염이 있어 뺀다, 나머지는 길이가 가운데인 클립)
      word-<번호>.mp4 로 그린다. 기본은 0.5배속으로 3번 반복(사이에 잠깐 멈춤).
"""
import argparse
import csv
import os

import cv2
import numpy as np

BODY = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 0), (0, 2), (1, 3), (2, 4)]   # 몸 11점 (landmarks.POSE_IDX 순서)
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
        (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
WHITE, LEFT, RIGHT, GRAY = (235, 235, 235), (255, 160, 60), (60, 160, 255), (120, 120, 120)   # BGR
FONT = "C:/Windows/Fonts/malgun.ttf"


def _edges(frame_xy, frame_mask, pairs, offset, color, img, size):
    for a, b in pairs:
        if frame_mask[offset + a] and frame_mask[offset + b]:
            p = (int(frame_xy[offset + a, 0] * size[0]), int(frame_xy[offset + a, 1] * size[1]))
            q = (int(frame_xy[offset + b, 0] * size[0]), int(frame_xy[offset + b, 1] * size[1]))
            cv2.line(img, p, q, color, 3, cv2.LINE_AA)


def draw_frame(xy, mask, face, face_mask, size, base=None):
    """한 프레임을 검은 화면(또는 base: 같은 크기의 BGR 이미지, 예: 카메라 화면)에 그린다. size = (너비, 높이) 픽셀."""
    img = np.zeros((size[1], size[0], 3), np.uint8) if base is None else base
    _edges(xy, mask, BODY, 0, WHITE, img, size)
    _edges(xy, mask, HAND, 11, LEFT, img, size)
    _edges(xy, mask, HAND, 32, RIGHT, img, size)
    for k in range(53):
        if mask[k]:
            cv2.circle(img, (int(xy[k, 0] * size[0]), int(xy[k, 1] * size[1])), 4,
                       WHITE if k < 11 else (LEFT if k < 32 else RIGHT), -1, cv2.LINE_AA)
    if face is not None:
        for p, ok in zip(face, face_mask):
            if ok:
                cv2.circle(img, (int(p[0] * size[0]), int(p[1] * size[1])), 2, GRAY, -1, cv2.LINE_AA)
    return img


def _text(img, top, bottom):
    """한글이 되는 글꼴로 글자를 얹는다(cv2.putText 는 한글을 못 쓴다). top 은 위에서부터, bottom 은 아래에서부터 쌓는다."""
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    size = max(18, img.shape[0] // 22)
    font = ImageFont.truetype(FONT, size) if os.path.exists(FONT) else ImageFont.load_default()
    d, step = ImageDraw.Draw(pil), int(size * 1.35)
    for i, s in enumerate(top):
        d.text((12, 8 + i * step), s, font=font, fill=(255, 255, 255))
    for i, s in enumerate(bottom):
        d.text((12, img.shape[0] - 8 - (len(bottom) - i) * step), s, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def event_lines(ev, at, names):
    """at(데이터 시각) 직전 3초 안에 낸 인식 결과를 글자로. 단어는 번호(names 가 있으면 이름), 지문자는 글자."""
    out = []
    for kind, tick, word, prob, text in ev:
        if 0 <= at - tick < 3.0:
            if kind == "sentence_end":
                out.append("[문장 끝]")
            elif kind == "fingerspelling" and text:
                out.append("[지문자] %s" % text)
            else:
                out.append("%s (%.2f)" % (names.get(word, "번호 %d" % word), prob))
    return out[-3:]


def render(t, xy, mask, face, face_mask, out_path, aspect=16 / 9, scale=720, fps=15.0, speed=1.0, loops=1, hold_s=0.0,
           title="", ev=(), names=None):
    """t 가 불규칙(버린 프레임)이어도 시각대로 재생한다: 출력 프레임마다 그 시각 직전의 가장 가까운 프레임을 쓴다."""
    names, size = names or {}, (int(round(scale * aspect)), scale)
    codec = "VP80" if out_path.endswith(".webm") else "mp4v"     # .webm(VP8)은 브라우저에서 바로 재생된다
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, size)
    n_out, n_hold, n = int(round((t[-1] - t[0]) / speed * fps)) + 1, int(round(hold_s * fps)), 0
    for _ in range(loops):
        for j in range(n_out + n_hold):
            at = t[0] + min(j, n_out - 1) / fps * speed
            i = int(np.clip(np.searchsorted(t, at + 1e-9, "right") - 1, 0, len(t) - 1))
            img = draw_frame(xy[i], mask[i], None if face is None else face[i], None if face_mask is None else face_mask[i], size)
            top, bottom = ([title] if title else []), event_lines(ev, at, names)
            if top or bottom:
                img = _text(img, top, bottom)
            writer.write(img)
            n += 1
    writer.release()
    return n


def load_recording(path):
    with np.load(path) as z:                              # with: Windows 에서 파일을 쥐고 있지 않게
        face = z["face"] if "face" in z.files else None
        fm = z["face_mask"] if "face_mask" in z.files else None
        aspect = float(z["wh"][0]) / float(z["wh"][1]) if "wh" in z.files else 16 / 9
        ev = list(zip(z["ev_kind"].tolist(), z["ev_tick"], z["ev_word"], z["ev_prob"], z["ev_text"].tolist())) if "ev_kind" in z.files else []
        return z["t"], z["xy"], z["mask"], face, fm, aspect, ev


def pick_clips(dirs, word_ids, skip_signers=(5, 7)):
    """단어마다 클립 하나: 수행자 5·7 을 빼고 프레임 수가 가운데인 것."""
    import dataset as ds
    by = {}
    for c in ds.load_clips(dirs, word_ids):
        if c["signer"] not in skip_signers:
            by.setdefault(c["word"], []).append(c)
    return {w: sorted(cs, key=lambda c: len(c["xy"]))[len(cs) // 2] for w, cs in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--out", default="")
    ap.add_argument("--words", default="")
    ap.add_argument("--features", nargs="*", default=[])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--names", default="", help="번호,이름 CSV. 주면 화면에 이름이 나온다(이 mp4 는 레포에 올리지 말 것)")
    ap.add_argument("--scale", type=int, default=720)
    ap.add_argument("--speed", type=float, default=None, help="재생 배속(기본: 녹화 1.0, 연습 0.5)")
    ap.add_argument("--loops", type=int, default=None, help="반복 횟수(기본: 녹화 1, 연습 3)")
    a = ap.parse_args()
    names = {}
    if a.names:
        with open(a.names, encoding="utf-8-sig") as f:
            names = {int(r[0]): r[1] for r in csv.reader(f) if r and r[0].strip().isdigit()}
    if a.words:
        ids = [int(w) for w in a.words.split(",")]
        os.makedirs(a.out_dir, exist_ok=True)
        clips = pick_clips(a.features, ids)
        for w in ids:
            c = clips.get(w)
            if c is None:
                print("클립 없음: 번호 %d" % w)
                continue
            t = np.arange(len(c["xy"])) / 15.0
            title = "번호 %d%s" % (w, " " + names[w] if w in names else "")
            n = render(t, c["xy"], c["mask"], c["face"], c["face_mask"], os.path.join(a.out_dir, "word-%d.mp4" % w),
                       aspect=c["aspect"], scale=a.scale, speed=a.speed or 0.5, loops=a.loops or 3, hold_s=0.6, title=title)
            print("번호 %d: %d프레임 (수행자 %d)" % (w, n, c["signer"]))
        return
    t, xy, mask, face, fm, aspect, ev = load_recording(a.npz)
    n = render(t, xy, mask, face, fm, a.out, aspect=aspect, scale=a.scale, speed=a.speed or 1.0, loops=a.loops or 1, ev=ev, names=names)
    print("저장: %s (%d프레임, 인식 결과 %d건)" % (a.out, n, len(ev)))


if __name__ == "__main__":
    main()
