"""실제 영상에서 구간 탐지기로 단어를 잘라 단어 모델로 분류하고, 정답 구간과 비교한다(끝까지 평가).

사용:
  추출(영상마다 한 번): python src/eval_video_words.py extract <영상.mp4>... --out-dir <npz 폴더> [--fps 15]
  평가: python src/eval_video_words.py eval <npz 폴더> --labels <정답.csv> --word <단어.pt> --boundary <탐지기.pt|rule>
        [--names 번호,단어 CSV] [--iou 0.5] [--out 결과.json]

정답 CSV 열: video_id, start_sec, end_sec 와 word_id(또는 label + --names 로 번호 변환). video_id = npz 파일 이름(확장자 빼고).
word_id 가 비었거나 모델 어휘 밖이면 그 구간은 구간 탐지 채점에만 쓰고 단어 채점에서는 뺀다.
두 가지를 같이 잰다:
  정답 구간: 정답 구간 ±0.5초를 잘라 분류(경계 오류를 뺀 분류 성능)
  탐지 구간: 탐지기가 낸 구간을 정답 구간과 IoU 로 1:1 짝지은 뒤, 짝지어진 구간을 분류(실제로 쓸 때의 성능)
단어 모델 전처리는 .pt 안의 config 를 따르고, 화면비는 영상의 너비/높이를 쓴다(학습 영상은 16:9).
결과 json 에는 번호·수치만 넣는다(단어 이름 없음).
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import boundary as bd
import models
import preprocess as pp

LEFT, RIGHT = slice(11, 32), slice(32, 53)        # landmarks.py 점 순서: 상체 11 + 왼손 21 + 오른손 21


def extract(paths, out_dir, fps):
    import landmarks
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        r = landmarks.extract_video(p, fps)
        if r is None:
            print("실패:", p)
            continue
        dst = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".npz")
        np.savez(dst, t=r["t"], xy=r["xy"], mask=r["mask"], face=r["face"], face_mask=r["face_mask"],
                 wh=np.asarray(r["wh"], np.int32), src_fps=r["src_fps"])
        m = r["mask"]
        print("%s: 프레임 %d, %s, 왼손 %.3f 오른손 %.3f 어깨 %.3f" % (
            dst, len(r["t"]), tuple(r["wh"]), m[:, LEFT].any(1).mean(), m[:, RIGHT].any(1).mean(), m[:, 5:7].all(1).mean()))


def aihub_labels(npz_dirs, label_zip, names, out):
    """AI-Hub 문장 npz 마다 morpheme json 의 구간을 정답 CSV(video_id, start_sec, end_sec, word_id)로. 이름 대신 번호만 쓴다.
    단어 목록에 없는 이름(어휘 밖)은 word_id 를 비운다."""
    import zipfile
    from extract_sentences import label_name
    with open(names, encoding="utf-8-sig") as f:
        ids = {r[1].strip(): int(r[0]) for r in csv.reader(f) if r and r[0].strip().isdigit()}
    n = hit = 0
    with zipfile.ZipFile(label_zip) as z, open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "start_sec", "end_sec", "word_id"])
        for d in npz_dirs:
            for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
                base = os.path.basename(p)[:-4]
                for item in json.loads(z.read(label_name(base))).get("data") or []:
                    names_ = [a.get("name", "").strip() for a in item.get("attributes") or []]
                    wid = ids.get(names_[0], "") if len(names_) == 1 else ""
                    w.writerow([base, float(item["start"]), float(item["end"]), wid])
                    n, hit = n + 1, hit + (wid != "")
    print("구간 %d개, 어휘 안 %d개 -> %s" % (n, hit, out))


def load_labels(path, names):
    ids = {}
    if names:
        with open(names, encoding="utf-8-sig") as f:
            ids = {r[1].strip(): int(r[0]) for r in csv.reader(f) if r and r[0].strip().isdigit()}
    out = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            w = (r.get("word_id") or "").strip()
            wid = int(w) if w else ids.get((r.get("label") or "").strip(), -1)
            out[r["video_id"].strip()].append((float(r["start_sec"]), float(r["end_sec"]), wid))
    return out


def classify(model, classes, cfg, d, start, end, aspect, margin=0.5):
    """[start - margin, end + margin] 프레임을 분류. 반환: (상위 5개 번호, 확률) 또는 None."""
    sel = (d["t"] >= start - margin) & (d["t"] <= end + margin)
    if sel.sum() < 2:
        return None
    face = cfg.get("face", False)
    xy = np.concatenate([d["xy"][sel], d["face"][sel]], 1) if face else d["xy"][sel]
    mask = np.concatenate([d["mask"][sel], d["face_mask"][sel]], 1) if face else d["mask"][sel]
    f = pp.build_features(xy, mask, cfg["length"], with_hand_rel=cfg["hand_rel"], with_angles=cfg["angles"],
                          with_traj=cfg["traj"], with_bones=cfg["bones"], origin=cfg["origin"], aspect=aspect,
                          with_folds=cfg.get("folds", False), with_face=face)
    if f is None:
        return None
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(f[None])), 1)[0]
    top = torch.topk(p, 5)
    return [int(classes[i]) for i in top.indices], float(top.values[0])


def score(rows):
    n = len(rows)
    top1 = sum(r["pred"] is not None and r["pred"][0] == r["true"] for r in rows)
    top5 = sum(r["pred"] is not None and r["true"] in r["pred"] for r in rows)
    return {"n": n, "top1": top1, "top5": top5, "top1_rate": top1 / n if n else None, "top5_rate": top5 / n if n else None}


def evaluate(npz_dir, labels, word_pt, boundary_spec, iou_thr, margin=0.5):
    ck = torch.load(word_pt, weights_only=False)
    model = models.build(ck["model"], ck["n_feat"], len(ck["classes"]))
    model.load_state_dict(ck["state"])
    model.eval()
    classes, cfg, vocab = ck["classes"], ck["config"], set(int(c) for c in ck["classes"])
    det = bd.BoundaryDetector(None if boundary_spec == "rule" else boundary_spec)
    oracle, e2e, videos = [], [], []
    n_true = n_pred = n_match = 0
    for vid, segs in sorted(labels.items()):
        path = os.path.join(npz_dir, vid + ".npz")
        if not os.path.exists(path):
            print("npz 없음:", vid)
            continue
        z = np.load(path)
        d = {k: z[k] for k in ("t", "xy", "mask", "face", "face_mask")}
        aspect = float(z["wh"][0]) / float(z["wh"][1]) if "wh" in z.files else 16 / 9
        ev = bd.run_stream(det, d["t"], d["xy"], d["mask"], d["face"], d["face_mask"])
        pred = [(s, e) for _, s, e, *k in ev if (k[0] if k else "word") != "sentence_end"]
        true = [(s, e) for s, e, _ in segs]
        m = bd.match_segments(pred, true, iou_thr)
        n_true, n_pred, n_match = n_true + len(true), n_pred + len(pred), n_match + len(m)
        for s, e, w in segs:
            if w in vocab:
                oracle.append({"video": vid, "seg": [s, e], "true": w, "pred": (classify(model, classes, cfg, d, s, e, aspect, margin) or [None])[0]})
        for i, j, v in m:
            s, e, w = segs[j]
            if w in vocab:
                r = classify(model, classes, cfg, d, pred[i][0], pred[i][1], aspect, margin)
                e2e.append({"video": vid, "seg": [s, e], "pred_seg": list(pred[i]), "iou": v, "true": w, "pred": r[0] if r else None})
        # 정답과 짝지어지지 않은 정답 구간도 단어 채점 분모에 넣는다(놓친 단어 = 틀림)
        hit = {j for _, j, _ in m}
        e2e += [{"video": vid, "seg": [s, e], "true": w, "pred": None, "missed": True}
                for j, (s, e, w) in enumerate(segs) if j not in hit and w in vocab]
        mk = d["mask"]
        videos.append({"video": vid, "frames": int(len(d["t"])), "wh": [int(x) for x in z["wh"]] if "wh" in z.files else None,
                       "left_hand": float(mk[:, LEFT].any(1).mean()), "right_hand": float(mk[:, RIGHT].any(1).mean()),
                       "n_true": len(true), "n_pred": len(pred), "n_match": len(m),
                       "pred_segments": [[round(s, 2), round(e, 2)] for s, e in pred]})
    return {"word_model": word_pt, "word_classes": len(classes), "boundary": boundary_spec, "iou_thr": iou_thr, "margin": margin,
            "segments": {"n_true": n_true, "n_pred": n_pred, "n_match": n_match,
                         "recall": n_match / n_true if n_true else None, "precision": n_match / n_pred if n_pred else None},
            "oracle": score(oracle), "end_to_end": score(e2e), "videos": videos,
            "oracle_rows": oracle, "end_to_end_rows": e2e}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("videos", nargs="+")
    x.add_argument("--out-dir", required=True)
    x.add_argument("--fps", type=float, default=15.0)
    e = sub.add_parser("eval")
    e.add_argument("npz_dir")
    e.add_argument("--labels", required=True)
    e.add_argument("--word", required=True)
    e.add_argument("--boundary", default="rule")
    e.add_argument("--names", default="")
    e.add_argument("--iou", type=float, default=0.5)
    e.add_argument("--margin", type=float, default=0.5, help="분류할 때 구간 앞뒤로 더 넣는 초(학습은 라벨 ±0.5초)")
    e.add_argument("--out", default="")
    g = sub.add_parser("labels-aihub")
    g.add_argument("npz_dirs", nargs="+")
    g.add_argument("--label-zip", required=True)
    g.add_argument("--names", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "labels-aihub":
        return aihub_labels(a.npz_dirs, a.label_zip, a.names, a.out)
    if a.cmd == "extract":
        return extract(a.videos, a.out_dir, a.fps)
    res = evaluate(a.npz_dir, load_labels(a.labels, a.names), a.word, a.boundary, a.iou, a.margin)
    s = res["segments"]
    print("구간: 정답 %d, 탐지 %d, 짝 %d (재현율 %s, 정밀도 %s)" % (s["n_true"], s["n_pred"], s["n_match"], s["recall"], s["precision"]))
    for k in ("oracle", "end_to_end"):
        print("%s: %s" % (k, {kk: v for kk, v in res[k].items()}))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
