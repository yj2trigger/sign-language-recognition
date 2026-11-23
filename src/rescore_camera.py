"""카메라 실험 녹화(realtime.py --record 의 npz)를 따라 할 영상의 시간표로 채점한다.

① live  : 녹화 때 실시간으로 나온 인식 결과(npz 의 ev_*)를 그대로 채점.
② replay: npz 의 관절 좌표를 찍힌 시각 순서대로 realtime 의 실시간 루프(_frames)에 한 프레임씩 넣어 다시 인식한다.
          가짜 시계로 녹화 때 버린 박자를 그대로 건너뛰고, 미래 프레임은 보지 않는다. 모델·탐지기·margin·후처리를 바꿔 비교한다.
          조건을 녹화 때와 같게 주면 ①과 같은 결과가 나와야 한다(--check).

시간표: make_follow.py 와 같은 방식으로 계산하고 follow-<세션>.mp4 의 프레임 수와 정확히 맞는지 확인한다.
  단어 세션: 단어마다 [안내 2.5초 | 단어 영상 | 쉼 2초] → 그 구간에 정답이 한 번이라도 나오면 맞음.
  문장 세션: [안내 2.5초] + ([문장 영상 | 쉼 3초] × 반복) → 반복마다 LCS·편집 거리.
인식 결과는 구간 가운데 시각으로 어느 칸에 속하는지 정한다. 출력에는 단어 번호만 쓴다(이름 없음).

예: python rescore_camera.py D:/.../cam-A1-1.npz                 (① 만)
    python rescore_camera.py D:/.../cam-A1-1.npz --model ftr-f5-s2.pt --boundary te1-s0.pt --margin 0.5 --post "..." --check
"""
import argparse
import contextlib
import json
import os
import sys
import types

import cv2
import numpy as np

import realtime as rt
import segment_post as sp
from eval_video_words import edit_distance, lcs

P = "D:/sign-language-data/external/camera/practice"
FOLLOW = "D:/sign-language-data/external/camera/follow"
FPS = 30                                                  # make_follow.py 의 FPS


def _clip_frames(path):
    cap = cv2.VideoCapture(path)
    src, n = cap.get(cv2.CAP_PROP_FPS) or 30, 0
    while cap.read()[0]:
        n += 1
    return int(n / src * FPS)


def timeline(ses, sc):
    """칸 목록 [(시작초, 끝초, 정답)] — 정답은 단어 번호(단어 세션) 또는 단어 열(문장 세션)."""
    f, cells = 0, []
    card = lambda s: int(s * FPS)
    if ses in ("A1", "A2"):
        for w in sc[ses]:
            real = "%s/word-%d-real.mp4" % (P, w)
            a = f
            f += card(2.5) + _clip_frames(real if os.path.exists(real) else "%s/word-%d-skel.webm" % (P, w)) + card(2)
            cells.append((a / FPS, f / FPS, w))
    else:
        f += card(2.5)
        clip = _clip_frames("%s/%s-skel.webm" % (P, ses))
        for _ in range(sc["reps"]):
            a = f
            f += clip + card(3)
            cells.append((a / FPS, f / FPS, list(sc["sentences"][ses])))
    cap = cv2.VideoCapture("%s/follow-%s.mp4" % (FOLLOW, ses))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n != f:
        sys.exit("시간표 %d프레임 ≠ follow-%s.mp4 %d프레임 — make_follow.py 와 계산이 다름" % (f, ses, n))
    return cells


def tokens_from_npz(z):
    """[(구간 가운데 초, 토큰)] 토큰: 단어 번호 또는 'FS'. 문장 끝 알림은 뺀다."""
    out = []
    for k, s, e, w in zip(z["ev_kind"], z["ev_start"], z["ev_end"], z["ev_word"]):
        if k != "sentence_end":
            out.append(((s + e) / 2, "FS" if k == "fingerspelling" else int(w)))
    return out


def score(cells, toks, sentence):
    rows, outside = [], [w for t, w in toks if not any(a <= t < b for a, b, _ in cells)]
    for a, b, exp in cells:
        got = [w for t, w in toks if a <= t < b]
        if sentence:
            rows.append({"got": got, "lcs": lcs(exp, got), "ed": edit_distance(exp, got), "n": len(exp)})
        else:
            rows.append({"exp": exp, "got": got, "hit": exp in got, "wrong": sum(w != exp for w in got)})
    if sentence:
        n = sum(r["n"] for r in rows)
        tot = {"lcs": sum(r["lcs"] for r in rows), "n": n, "wer": sum(r["ed"] for r in rows) / n}
    else:
        tot = {"hit": sum(r["hit"] for r in rows), "n": len(rows), "wrong": sum(r["wrong"] for r in rows),
               "empty": sum(not r["got"] for r in rows)}
    tot["outside"] = len(outside)
    return {"rows": rows, "total": tot}


def show(label, res, sentence):
    t = res["total"]
    if sentence:
        print("%-8s 순서 맞춘 단어 %d/%d, WER %.2f, 칸 밖 %d" % (label, t["lcs"], t["n"], t["wer"], t["outside"]))
        for i, r in enumerate(res["rows"]):
            print("   반복 %d: LCS %d/%d, 편집 거리 %d, 인식 %s" % (i + 1, r["lcs"], r["n"], r["ed"], r["got"]))
    else:
        print("%-8s 맞음 %d/%d, 오답 %d, 빈 칸 %d, 칸 밖 %d" % (label, t["hit"], t["n"], t["wrong"], t["empty"], t["outside"]))
        for i, r in enumerate(res["rows"]):
            print("   %2d. 정답 %-5s %s  인식 %s" % (i + 1, r["exp"], "O" if r["hit"] else "X", r["got"]))


class ReplaySource:
    """npz 의 프레임 번호를 '영상 프레임' 대신 돌려준다. 실제 좌표는 가짜 process_frame 이 꺼낸다."""

    def __init__(self, t):
        self.idx = {int(round(x * 15)): i for i, x in enumerate(t)}
        self.last = int(round(t[-1] * 15))

    def frame_at(self, tick):
        k = int(round(tick * 15))
        return None if k > self.last else self.idx.get(k)


def replay(z, model_path, boundary, margin, post):
    """녹화 좌표를 실시간 루프에 그대로 흘려 다시 인식한다. 반환: npz 와 같은 모양의 ev 사전."""
    t, fps = z["t"], float(z["src_fps"])
    assert fps == 15.0, "ReplaySource 는 15fps 기준"
    clock = {"now": 0.0}

    def process_frame(h, i, with_face=False):            # 녹화된 좌표를 꺼내고 시계를 다음 녹화 프레임 시각으로 옮긴다
        clock["now"] = (t[i + 1] if i + 1 < len(t) else t[i] + 1 / fps) + 1e-6
        r = (z["xy"][i], z["mask"][i], None, tuple(z["wh"]))
        return r + ((z["face"][i], z["face_mask"][i]) if with_face else ())

    fake_time = types.SimpleNamespace(perf_counter=lambda: clock["now"], sleep=lambda s: None)
    fake_mp = types.SimpleNamespace(solutions=types.SimpleNamespace(holistic=types.SimpleNamespace(
        Holistic=lambda **k: contextlib.nullcontext())))
    real = rt.time, rt.mp, rt.landmarks.process_frame
    rt.time, rt.mp, rt.landmarks.process_frame = fake_time, fake_mp, process_frame
    try:
        fs_reader = None
        if os.environ.get("FINGERSPELLING_CHECKPOINT"):
            from fingerspelling_runtime import read_fingerspelling as fs_reader
        model, classes, cfg = rt.load_model(model_path)
        rec = {"on": True}
        with contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8")):
            rt.run(ReplaySource(t), model, classes, cfg, fps, segmenter=rt.make_segmenter(boundary), margin=margin,
                   fs_reader=fs_reader, rec=rec, post=sp.PostConfig.parse(post) if post else None)
    finally:
        rt.time, rt.mp, rt.landmarks.process_frame = real
    ev = rec["ev"]
    return {"ev_kind": np.array([e[0] for e in ev], dtype=str), "ev_start": np.array([e[1] for e in ev]),
            "ev_end": np.array([e[2] for e in ev]), "ev_word": np.array([e[4] for e in ev], dtype=np.int64),
            "replay_frames": len(rec["t"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--model", default="", help="주면 ② replay 도 한다")
    ap.add_argument("--boundary", default="")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--post", default="")
    ap.add_argument("--check", action="store_true", help="replay 결과가 live 와 같은지 비교(녹화 때와 같은 조건을 줄 때)")
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    meta = json.loads(str(z["score_json"]))
    ses = meta["session"]
    sc = json.load(open(P + "/script.json", encoding="utf-8"))
    sentence = ses not in ("A1", "A2")
    cells = timeline(ses, sc)
    print("세션 %s, 녹화 %.1f초 %d프레임, 시간표 %.1f초 %d칸" % (ses, z["t"][-1], len(z["t"]), cells[-1][1], len(cells)))
    out = {"npz": os.path.basename(a.npz), "session": ses, "live": score(cells, tokens_from_npz(z), sentence)}
    show("live", out["live"], sentence)
    if a.model:
        ev = replay(z, a.model, a.boundary, a.margin, a.post)
        out["replay"] = score(cells, tokens_from_npz(ev), sentence)
        out["replay"]["cond"] = {"model": a.model, "boundary": a.boundary, "margin": a.margin, "post": a.post}
        show("replay", out["replay"], sentence)
        print("replay 프레임 %d (녹화 %d)" % (ev["replay_frames"], len(z["t"])))
        if a.check:
            L = [(k, round(s, 3), round(e, 3), w) for k, s, e, w in zip(z["ev_kind"], z["ev_start"], z["ev_end"], z["ev_word"])]
            R = [(k, round(s, 3), round(e, 3), w) for k, s, e, w in zip(ev["ev_kind"], ev["ev_start"], ev["ev_end"], ev["ev_word"])]
            same = L == R
            print("check: live %d건, replay %d건 → %s" % (len(L), len(R), "같음" if same else "다름"))
            if not same:
                for i, (x, y) in enumerate(zip(L, R)):
                    if x != y:
                        print("   첫 차이 #%d live %s / replay %s" % (i, x, y))
                        break
            out["check_same"] = same
    dst = os.path.join(os.path.dirname(a.npz), "score-%s.json" % os.path.splitext(os.path.basename(a.npz))[0].replace("cam-", ""))
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=int)
    print("저장:", dst)


if __name__ == "__main__":
    main()
