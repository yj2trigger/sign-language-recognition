"""실제 문장 영상 특징(extract_sentences.py 결과)으로 구간 탐지기를 평가한다. 문턱값 여러 개를 한 번에 비교할 수 있다.

구간 지표(프레임 F1, 구간 F1·IoU, 지연)는 영상 끝 flush 를 포함해 계산하고(train_boundary.py 와 같음),
문장 끝 지표는 flush 없이 계산한다(boundary.sentence_end_stats): 영상 안 문장 끝 비율, 문장 중간 오알림 수, 문장 끝 지연.

사용:
  시험 수행자 평가   python src/eval_boundary.py <모델.pt> <features 폴더>... [--begin-thr 0.3] [--max-pause 1.2] [--face-cue 0]
  검증 문장으로 고르기 python src/eval_boundary.py <모델.pt> <학습 features 폴더>... --val-seed 0 --begin-thr 0.2,0.3,0.4,0.5
  --val-seed 를 주면 train_boundary.py --real-train <같은 폴더> --seed <같은 값> 이 검증으로 뺀 문장만 쓴다.
  값을 쉼표로 여러 개 주면 모든 조합을 돈다. 주지 않은 값은 모델 파일에 저장된 탐지기 설정을 쓴다.
  --out 을 주면 결과를 json 으로 쓴다(수치와 설정만, 단어명 없음).
"""
import argparse
import dataclasses
import itertools
import json
import os

import numpy as np

import boundary as bd
import dataset_bio as db


def val_split(real, seed, frac=0.1):
    """train_boundary.py 의 실제 문장 검증 분할과 같은 문장을 고른다(같은 폴더·같은 seed 일 때)."""
    idx = np.random.default_rng(seed).permutation(len(real))
    nv = max(1, int(len(real) * frac))
    return [real[i] for i in idx[:nv]]


def evaluate(det, data):
    """문장 목록 하나를 평가한다. 반환: 구간 지표(boundary.summarize) + 문장 끝 지표."""
    res, inv, mid, lat = [], 0, 0, []
    for d in data:
        F, FM = d["face"], d["face_mask"]
        ev = bd.run_stream(det, d["t"], d["xy"], d["mask"], F, FM, flush=False)
        se = bd.sentence_end_stats(ev, d["segments"])
        inv += se["in_video"]
        mid += se["mid_false"]
        if se["latency_s"] is not None:
            lat.append(se["latency_s"])
        res.append(bd.evaluate_stream(bd.run_stream(det, d["t"], d["xy"], d["mask"], F, FM), d["t"], d["segments"]))
    s = bd.summarize(res)
    keep = ("n_videos", "frame_f1", "segment_f1", "mean_iou", "latency_median_s", "latency_p95_s")
    out = {k: s[k] for k in keep}
    out.update(sentence_end_in_video_rate=inv / len(data) if data else 0.0, sentence_end_mid_false=int(mid),
               sentence_end_latency_median_s=float(np.median(lat)) if lat else None)
    return out


def floats(text):
    return [float(x) for x in text.split(",")] if text else [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="train_boundary.py 가 저장한 .pt, 또는 rule")
    ap.add_argument("features", nargs="+", help="실제 문장 특징 npz 폴더(extract_sentences.py 의 <out>/features)")
    ap.add_argument("--val-seed", type=int, default=None, help="주면 train_boundary.py 의 검증 문장만 쓴다")
    ap.add_argument("--val-frac", type=float, default=0.1, help="train_boundary.py --real-val-frac 와 같은 값")
    ap.add_argument("--begin-thr", default="", help="새 단어 시작 문턱(쉼표로 여러 개)")
    ap.add_argument("--max-pause", default="", help="이만큼 멈추면 무조건 문장 끝(초, 쉼표로 여러 개)")
    ap.add_argument("--face-cue", default="", help="표정 단서 1/0(쉼표로 여러 개)")
    ap.add_argument("--out", default="", help="결과 json 경로")
    a = ap.parse_args()

    data = db.load_real_sentences(a.features)
    if not data:
        raise SystemExit("문장 npz 가 없다: %s" % a.features)
    if a.val_seed is not None:
        data = val_split(data, a.val_seed, a.val_frac)
    if a.model == "rule":
        model, saved = None, {}
    else:
        model, ck = bd.load_model(a.model)
        saved = ck.get("detector", {})
    base = bd.DetectorConfig(**saved)
    print("문장 %d개(%s), 수행자 %s" % (len(data), "검증" if a.val_seed is not None else "시험",
                                   sorted({d["signer"] for d in data})), flush=True)
    rows = []
    for thr, mp, fc in itertools.product(floats(a.begin_thr), floats(a.max_pause), floats(a.face_cue)):
        cfg = dataclasses.replace(base, **{k: v for k, v in (("begin_thr", thr), ("max_pause_s", mp)) if v is not None},
                                  **({} if fc is None else {"face_cue": bool(fc)}))
        r = evaluate(bd.BoundaryDetector(model, cfg), data)
        r["config"] = {"begin_thr": cfg.begin_thr, "max_pause_s": cfg.max_pause_s, "face_cue": cfg.face_cue}
        rows.append(r)
        lat = r["latency_median_s"]
        print("begin_thr %.2f max_pause %.1f face_cue %d | 구간F1 %.3f 프레임F1 %.3f IoU %.3f 지연 %s | "
              "영상 안 문장 끝 %.3f 문장 중간 오알림 %d" % (
                  cfg.begin_thr, cfg.max_pause_s, cfg.face_cue, r["segment_f1"], r["frame_f1"], r["mean_iou"],
                  "-" if lat is None else "%.0fms" % (1000 * lat), r["sentence_end_in_video_rate"], r["sentence_end_mid_false"]),
              flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"model": os.path.basename(a.model), "split": "val" if a.val_seed is not None else "test",
                       "val_seed": a.val_seed, "signers": sorted({d["signer"] for d in data}), "results": rows},
                      f, ensure_ascii=False, indent=1)
        print("결과 ->", a.out)


if __name__ == "__main__":
    main()
