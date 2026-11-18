"""지문자/단어 판별기(T-073): 구간 탐지기 조각을 간격 gap 이내로 묶은 후보마다 요약 특징 → 로지스틱 회귀.
지금 규칙(손가락/손목 움직임 비율 하나, segment_post ②)이 지문자 영상의 약 59% 를 놓쳐서(T-068 잡음 0.41) 바꿔 보는 실험.

데이터: 지문자 = 지문자 영상 npz(영상 전체가 지문자), 단어 = AI-Hub 문장 영상(수행자 1·2·3, 지문자 표시 없음 → 대부분 단어).
평가: 지문자 잡음 = 지문자 영상을 하나씩 빼고 학습해 그 영상에서 지문자 판정 구간이 영상 길이 절반 이상을 덮는 비율,
      오판 = 문장 수행자 한 명을 빼고 학습해 그 사람 문장 100개당 지문자 판정 수. 기준선 = 같은 묶음에 지금 규칙.
사용: python src/fs_judge.py --fs-npz <지문자 npz 폴더> --n-sent 400 --out <결과.json> --model <판별기.json>
     수행자별 판별기: ... --out x --target-fp 0.005 --save-heldout <폴더>/fs_judge-s%d.json
판별기 json 은 표준화 평균·표준편차·계수·절편·문턱값만 담는다(피클 없음).
"""
import argparse
import glob
import json
import os

import numpy as np

import boundary as bd
import segment_post as sp
from eval_video_words import load_npz

M = "D:/sign-language-data/models/boundary-rot/te%d-s0.pt"
SEN = {1: "D:/sign-language-data/sentences/real_sen_01/features", 2: "D:/sign-language-data/sentences/real_sen_03/features",
       3: "D:/sign-language-data/sentences/real_sen_05/features"}
NAMES = ["dur", "n_pieces", "finger_sum", "wrist_sum", "ratio", "finger_mean", "wrist_mean", "wrist_range", "hand_height",
         "one_hand", "two_hand", "finger_peak", "wrist_peak", "piece_mean"]


def groups(found, gap):
    out, cur = [], []
    for s, e, k in found:
        if cur and s - cur[-1][1] > gap:
            out.append(cur)
            cur = []
        cur.append((s, e))
    if cur:
        out.append(cur)
    return out


def features(t, xy, mask, mot, g):
    s, e = g[0][0], g[-1][1]
    sel = (t >= s) & (t <= e)
    fm, wm = mot[sel, 0], mot[sel, 1]
    x, m = xy[sel], mask[sel]
    sh = (m[:, 5] & m[:, 6])
    width = np.median(np.abs(x[sh, 5, 0] - x[sh, 6, 0])) if sh.any() else 0.2
    mid_y = np.median((x[sh, 5, 1] + x[sh, 6, 1]) / 2) if sh.any() else 0.5
    lh, rh = m[:, 11:32].any(1), m[:, 32:53].any(1)
    wr = []
    for hand, idx in ((lh, 11), (rh, 32)):
        if hand.sum() >= 2:
            wr.append(np.ptp(x[hand, idx], axis=0).max() / max(width, 1e-3))
    heights = [(mid_y - x[hand, idx, 1]).mean() / max(width, 1e-3) for hand, idx in ((lh, 11), (rh, 32)) if hand.any()]
    n = max(1, sel.sum())
    return [e - s, len(g), fm.sum(), wm.sum(), fm.sum() / max(wm.sum(), 1e-6), fm.sum() / n, wm.sum() / n, max(wr) if wr else 0.0,
            max(heights) if heights else -2.0, float((lh ^ rh).mean()), float((lh & rh).mean()),
            float(fm.max()) if len(fm) else 0.0, float(wm.max()) if len(wm) else 0.0, (e - s) / len(g)]


DETS = {}


def detector(k):
    if k not in DETS:
        DETS[k] = bd.BoundaryDetector(M % k)
    return DETS[k]


def candidates(det_ids, t, xy, mask, face=None, fmask=None, gap=0.5):
    """탐지기(te 번호 목록)마다 조각 → 묶음 → (특징, 시작, 끝)."""
    mot = sp.motion(t, xy, mask)
    out = []
    for k in det_ids:
        ev = bd.run_stream(detector(k), t, xy, mask, face, fmask)
        found = [(s, e, kk[0] if kk else "word") for _, s, e, *kk in ev if (kk[0] if kk else "word") != "sentence_end"]
        for g in groups(found, gap):
            out.append((k, features(t, xy, mask, mot, g), g[0][0], g[-1][1]))
    return out


def rule(f):
    """지금 규칙(T-068 ② 값): 길이 1.0초 이상 그리고 손가락/손목 ≥ 1.5."""
    return f[0] >= 1.0 and f[4] >= 1.5


class Judge:
    def __init__(self, mean, std, coef, intercept, thr):
        self.mean, self.std, self.coef, self.b, self.thr = map(np.asarray, (mean, std, coef, intercept, thr))

    def prob(self, F):
        z = (np.log1p(np.abs(F)) * np.sign(F) - self.mean) / self.std
        return 1 / (1 + np.exp(-(z @ self.coef + self.b)))

    def __call__(self, f):
        return bool(self.prob(np.asarray([f]))[0] >= self.thr)

    def to_json(self, path):
        json.dump({"names": NAMES, "mean": self.mean.tolist(), "std": self.std.tolist(), "coef": self.coef.tolist(),
                   "intercept": float(self.b), "thr": float(self.thr)}, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["mean"], d["std"], d["coef"], d["intercept"], d["thr"])


def fit(Fp, Fn, target_fp=0.03):
    """로지스틱 회귀(class_weight 균형). 문턱값은 학습용 단어 묶음의 오판율이 target_fp 이하가 되는 가장 낮은 값."""
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([Fp, Fn])
    y = np.r_[np.ones(len(Fp)), np.zeros(len(Fn))]
    Z = np.log1p(np.abs(X)) * np.sign(X)
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((Z - mean) / std, y)
    j = Judge(mean, std, lr.coef_[0], lr.intercept_[0], 0.5)
    pn = j.prob(np.asarray(Fn))
    j.thr = np.array(max(0.5, np.quantile(pn, 1 - target_fp)))
    return j


def clip_hit(cands, t, judge):
    cover = sum(e - s for _, f, s, e in cands if judge(f))
    return cover >= 0.5 * (t[-1] - t[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-npz", required=True)
    ap.add_argument("--n-sent", type=int, default=400, help="문장 수행자마다 쓸 문장 수(앞에서부터)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--target-fp", type=float, default=0.03)
    ap.add_argument("--save-heldout", default="", help="평가 없이 문장 수행자 s 를 뺀 판별기를 이 경로 형식(%%d 에 s)으로 저장")
    a = ap.parse_args()
    fs = []
    for p in sorted(glob.glob(os.path.join(a.fs_npz, "*.npz"))):
        with np.load(p) as z:
            t, xy, mask = z["t"], z["xy"], z["mask"]
        fs.append({"clip": os.path.basename(p)[:-4], "t": t, "c": candidates((1, 2, 3), t, xy, mask)})
    print("지문자 영상 %d개, 후보 묶음 %d개" % (len(fs), sum(len(c["c"]) for c in fs)), flush=True)
    sent = {}
    for s, d in SEN.items():
        rows = []
        for p in sorted(glob.glob(os.path.join(d, "*.npz")))[:a.n_sent]:
            dd, _ = load_npz(p)
            rows.append(candidates((s,), dd["t"], dd["xy"], dd["mask"], dd["face"], dd["face_mask"]))
        sent[s] = rows
        print("문장 수행자 %d: 문장 %d개, 후보 묶음 %d개" % (s, len(rows), sum(map(len, rows))), flush=True)
    pos = lambda clips: [f for c in clips for _, f, _, _ in c["c"] if f[0] >= 0.6]
    neg = lambda ss: [f for s in ss for row in sent[s] for _, f, _, _ in row]
    if a.save_heldout:
        for s in SEN:
            fit(pos(fs), neg([x for x in SEN if x != s]), a.target_fp).to_json(a.save_heldout % s)
            print("저장:", a.save_heldout % s)
        return
    # 지문자 잡음: 영상 하나씩 빼고(같은 영상의 탐지기 3개 결과 모두 함께 뺌)
    hit_j = hit_r = 0
    for i, c in enumerate(fs):
        j = fit(pos(fs[:i] + fs[i + 1:]), neg((1, 2, 3)))
        hit_j += clip_hit(c["c"], c["t"], j)
        hit_r += clip_hit(c["c"], c["t"], rule)
    # 오판: 문장 수행자 하나씩 빼고
    fp = {}
    for s in SEN:
        j = fit(pos(fs), neg([x for x in SEN if x != s]))
        fp[s] = {"judge": 100 * sum(j(f) for row in sent[s] for _, f, _, _ in row) / len(sent[s]),
                 "rule": 100 * sum(rule(f) for row in sent[s] for _, f, _, _ in row) / len(sent[s])}
    res = {"n_fs_clips": len(fs), "n_sent_per_signer": a.n_sent,
           "fs_recall": {"judge": hit_j / len(fs), "rule": hit_r / len(fs)}, "fp_per100": fp}
    print("지문자 잡음: 판별기 %.2f / 규칙 %.2f" % (res["fs_recall"]["judge"], res["fs_recall"]["rule"]))
    for s in SEN:
        print("문장 수행자 %d 100문장당 지문자 판정: 판별기 %.1f / 규칙 %.1f" % (s, fp[s]["judge"], fp[s]["rule"]))
    # 문턱값별 맞교환: 오판 목표(학습용 단어 묶음 비율)를 바꿔 같은 방식으로 다시 잼
    sweep = []
    for tfp in (0.003, 0.005, 0.01, 0.02, 0.03):
        h = sum(clip_hit(c["c"], c["t"], fit(pos(fs[:i] + fs[i + 1:]), neg((1, 2, 3)), tfp)) for i, c in enumerate(fs)) / len(fs)
        f = {s_: 100 * sum(j(x) for row in sent[s_] for _, x, _, _ in row) / len(sent[s_])
             for s_ in SEN for j in [fit(pos(fs), neg([y for y in SEN if y != s_]), tfp)]}
        sweep.append({"target_fp": tfp, "fs_recall": h, "fp_per100": f, "fp_mean": sum(f.values()) / 3})
        print("오판 목표 %.3f: 지문자 잡음 %.2f, 문장 100개당 오판 %s (평균 %.1f)" % (tfp, h, {k: round(v, 1) for k, v in f.items()}, sweep[-1]["fp_mean"]), flush=True)
    res["sweep"] = sweep
    final = fit(pos(fs), neg((1, 2, 3)))
    res["coef"] = dict(zip(NAMES, map(float, final.coef)))
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    if a.model:
        final.to_json(a.model)


if __name__ == "__main__":
    main()
