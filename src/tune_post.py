"""구간 후처리(segment_post) 값 고르기(T-068). 탐지기 모델은 그대로, 탐지기는 영상마다 한 번 돌리고 분류는 기억해 다시 쓴다.

1단계 지문자(②): 지문자 영상(--fs-npz, 영상 전체가 지문자)을 지문자로 잡는 비율과 문장 영상 100개당 지문자 판정 수(오탐 대용).
2단계 합치기(①③): 수행자 셋을 돌아가며 둘에서 문장 WER(어휘 안)이 가장 낮은 값을 골라 남은 하나에 적용한다.
사용: python src/tune_post.py --fs-npz <지문자 npz 폴더> --out <결과.json>
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

import boundary as bd
import segment_post as sp
from eval_video_words import classify, load_labels, load_npz, load_word, sentence_row

E, S, M = "D:/sign-language-data/external/aihub-sen", "D:/sign-language-data/sentences", "D:/sign-language-data/models"
SIGNERS = {1: ("real_sen_01", "ft/ftr-f1-s1.pt"), 2: ("real_sen_03", "ft/ftr-f5-s2.pt"), 3: ("real_sen_05", "ft/ftr-f0-s3.pt")}


def load_signer(s):
    sen_dir, word_pt = SIGNERS[s]
    model, classes, cfg, vocab = load_word("%s/%s" % (M, word_pt))
    det = bd.BoundaryDetector("%s/boundary-rot/te%d-s0.pt" % (M, s))
    vids = []
    for vid, segs in sorted(load_labels("%s/labels-s%d.csv" % (E, s), "").items()):
        p = "%s/%s/features/%s.npz" % (S, sen_dir, vid)
        if not os.path.exists(p):
            continue
        d, aspect = load_npz(p)
        ev = bd.run_stream(det, d["t"], d["xy"], d["mask"], d["face"], d["face_mask"])
        found = [(s_, e_, k[0] if k else "word") for _, s_, e_, *k in ev if (k[0] if k else "word") != "sentence_end"]
        cache = {}

        def top1(a, b, d=d, aspect=aspect, cache=cache):
            key = (round(a, 3), round(b, 3))
            if key not in cache:
                cache[key] = (classify(model, classes, cfg, d, a, b, aspect, 0.0) or [[None]])[0][0]
            return cache[key]
        vids.append({"vid": vid, "segs": segs, "found": found, "d": d, "top1": top1, "mot": sp.motion(d["t"], d["xy"], d["mask"])})
    print("수행자 %d: 영상 %d개" % (s, len(vids)), flush=True)
    return {"vids": vids, "vocab": vocab}


def score(data, post):
    rows = [sentence_row(v["vid"], v["segs"], v["found"], data["vocab"], v["top1"], post, v["d"]["t"], v["d"]["xy"],
                         v["d"]["mask"], v["mot"]) for v in data["vids"]]
    nv, ng = sum(r["n_vocab"] for r in rows), sum(r["n_gold"] for r in rows)
    return {"wer_vocab": sum(r["ed_vocab"] for r in rows) / nv, "wer_all": sum(r["ed_all"] for r in rows) / ng,
            "n_pred": sum(len(r["pred"]) for r in rows), "n_gold": ng, "fs_per100": 100 * sum(r["n_fs"] for r in rows) / len(rows)}


def fs_recall(fs_vids, post):
    """지문자 영상마다 지문자로 판정된 구간이 영상 길이의 절반 이상을 덮으면 잡은 것으로 센다."""
    hit = 0
    for v in fs_vids:
        t = v["t"]
        segs = sp.apply(v["found"], t, None, None, post, v["mot"]) if post.fs_gap_s else v["found"]
        cover = sum(e - s for s, e, k in segs if k == "fingerspelling")
        hit += cover >= 0.5 * (t[-1] - t[0])
    return hit / len(fs_vids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = {s: load_signer(s) for s in SIGNERS}
    fs_vids = []
    for p in sorted(glob.glob(os.path.join(a.fs_npz, "*.npz"))):
        with np.load(p) as z:
            t, xy, mask = z["t"], z["xy"], z["mask"]
        for s in SIGNERS:                                    # 탐지기 셋 모두로(지문자 영상 수행자는 문장 수행자와 다름)
            ev = bd.run_stream(bd.BoundaryDetector("%s/boundary-rot/te%d-s0.pt" % (M, s)), t, xy, mask)
            fs_vids.append({"t": t, "found": [(s_, e_, k[0] if k else "word") for _, s_, e_, *k in ev if (k[0] if k else "word") != "sentence_end"],
                            "mot": sp.motion(t, xy, mask)})
    res = {"fs": [], "merge": [], "rotation": []}
    base = sp.PostConfig()
    print("기준(후처리 없음): 지문자 잡음 %.2f, 문장 100개당 지문자 판정 %.1f" % (
        fs_recall(fs_vids, base), np.mean([score(data[s], base)["fs_per100"] for s in SIGNERS])), flush=True)
    # 1단계: 지문자 묶음 판정
    for g, mn, r in itertools.product((0.15, 0.3, 0.5), (0.6, 1.0, 1.5), (1.0, 1.5, 2.0, 3.0)):
        post = sp.PostConfig(fs_gap_s=g, fs_min_s=mn, fs_ratio=r)
        per100 = np.mean([score(data[s], post)["fs_per100"] for s in SIGNERS])
        res["fs"].append({"fs_gap_s": g, "fs_min_s": mn, "fs_ratio": r, "recall": fs_recall(fs_vids, post), "fs_per100": per100})
    for x in sorted(res["fs"], key=lambda x: (-x["recall"], x["fs_per100"]))[:10]:
        print("지문자 묶음 %(fs_gap_s).2f/%(fs_min_s).1f/%(fs_ratio).1f: 잡음 %(recall).2f, 문장 100개당 판정 %(fs_per100).1f" % x, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    # 2단계: 합치기(지문자 묶음은 끈 채로 — 문장 정답에 지문자 표시가 없어 WER 로는 지문자를 고를 수 없다)
    grid = list(itertools.product((0.0, 0.2, 0.3, 0.4, 0.5), (0.1, 0.2, 0.3), (0.0, 0.3, 0.6)))
    table = {}
    for fr, gp, dd in grid:
        post = sp.PostConfig(frag_s=fr, gap_s=gp, dedup_gap_s=dd)
        table[(fr, gp, dd)] = {s: score(data[s], post) for s in SIGNERS}
        res["merge"].append({"frag_s": fr, "gap_s": gp, "dedup_gap_s": dd, **{"s%d" % s: table[(fr, gp, dd)][s] for s in SIGNERS}})
    print("합치기 값 %d개 계산 끝" % len(grid), flush=True)
    for t_ in SIGNERS:
        others = [s for s in SIGNERS if s != t_]
        best = min(grid, key=lambda k: np.mean([table[k][s]["wer_vocab"] for s in others]))
        b0, b1 = table[(0.0, 0.1, 0.0)][t_], table[best][t_]
        res["rotation"].append({"test": t_, "best": best, "before": b0, "after": b1})
        print("시험 수행자 %d: 고른 값 frag_s=%.1f gap_s=%.1f dedup=%.1f | WER 어휘 안 %.3f → %.3f, 전체 %.3f → %.3f, 예측/정답 %.2f → %.2f" % (
            t_, *best, b0["wer_vocab"], b1["wer_vocab"], b0["wer_all"], b1["wer_all"], b0["n_pred"] / b0["n_gold"], b1["n_pred"] / b1["n_gold"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
