"""구간 탐지기(인과 TCN/GRU, 프레임별 B/I/O)를 수행자 독립 분할로 학습하고, 실시간과 같은 방식(프레임 하나씩)으로 평가한다.

사용: python src/train_boundary.py --features <full-fps npz 폴더>... [--fold 0] [--arch tcn] [--seed 0]
      [--epochs 30] [--n-train 3000] [--save-model D:\\...\\boundary-tcn-f0.pt]
      [--trial "제목" --hypothesis "..." --attempt "..." --decision "..."]
      데이터 없이 확인: python src/train_boundary.py --synthetic --epochs 3
학습 문장은 같은 수행자의 단어 영상을 이어 붙인 의사 문장이다(dataset_bio.make_sentences). 분할은 configs/split-6fold.json.
실제 문장: --real-train <폴더>... --real-test <폴더>... (수행자가 겹치면 거부. --real-test 를 비우면 전부 학습하는 최종 모델용). --features 를 함께 주면 학습 수행자의 의사 문장을 섞는다.
평가: 시험 수행자 문장마다 BoundaryDetector 에 프레임을 하나씩 넣어 프레임 F1, 구간 F1(IoU≥0.5), 평균 IoU, 지연을 잰다.
같은 문장에 규칙 모드(모델 없음)도 돌려 기준선으로 함께 남긴다.
결과 json(지표·설정만)은 runs/boundary/, 모델(.pt)은 데이터 파생물이라 레포 밖에 저장한다.
"""
import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

import boundary as bd
import dataset_bio as db

REPO = db.REPO


def git_state():
    try:
        g = ["git", "-C", REPO, "-c", "safe.directory=" + REPO.replace("\\", "/")]
        h = subprocess.run(g + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(g + ["status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
        return h + ("+수정중" if dirty else "")
    except Exception:
        return "알 수 없음"


def prepare(sentences, fps=15.0):
    """문장마다 (특징 (T,F), B/I/O (T,)) 를 만든다."""
    return [(bd.batch_features(s["t"], s["xy"], s["mask"], fps), db.bio_labels(s["t"], s["segments"])[0].astype(np.int64))
            for s in sentences]


def crops(data, length, rng):
    """문장마다 무작위 위치에서 length 프레임을 잘라 배치로. 짧은 문장은 뒤를 0·무시(-100)로 채운다."""
    X = np.zeros((len(data), length, bd.N_FEAT), np.float32)
    Y = np.full((len(data), length), -100, np.int64)
    for i, (f, y) in enumerate(data):
        s = int(rng.integers(0, max(1, len(f) - length + 1)))
        n = min(length, len(f) - s)
        X[i, :n], Y[i, :n] = f[s:s + n], y[s:s + n]
    return X, Y


def frame_scores(model, data):
    """문장 전체를 한 번에 넣은 프레임별 3분류 결과로 (macro-F1, 활동 F1). 인과 모델이라 실시간 출력과 같다(TCN 은 받아들이는 범위 안에서)."""
    model.eval()
    cm = np.zeros((3, 3), int)
    with torch.no_grad():
        for f, y in data:
            p = model(torch.from_numpy(f[None]))[0].argmax(1).numpy()
            np.add.at(cm, (y, p), 1)
    f1 = []
    for k in range(3):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k].sum() - cm[k, k]
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    act_tp = cm[1:, 1:].sum()
    act_f1 = 2 * act_tp / (2 * act_tp + cm[0, 1:].sum() + cm[1:, 0].sum()) if act_tp else 0.0
    return float(np.mean(f1)), float(act_f1)


def train(arch, tr, va, epochs, patience, seed, crop=96, batch=64, steps=40, lr=2e-3, b_weight=5.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = bd.build_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, b_weight, 1.0]), ignore_index=-100)  # B 는 단어마다 1프레임뿐이라 가중
    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            idx = rng.choice(len(tr), size=min(batch, len(tr)), replace=False)
            X, Y = crops([tr[i] for i in idx], crop, rng)
            logits = model(torch.from_numpy(X))
            loss = loss_fn(logits.reshape(-1, 3), torch.from_numpy(Y).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        mf1, af1 = frame_scores(model, va)
        hist.append({"epoch": ep + 1, "loss": tot / steps, "val_macro_f1": mf1, "val_active_f1": af1})
        print("에폭 %d 손실 %.4f 검증 macro-F1 %.4f 활동 F1 %.4f" % (ep + 1, tot / steps, mf1, af1), flush=True)
        if mf1 > best:
            best, best_state, wait = mf1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best, hist


def stream_eval(sentences, model=None, cfg=None):
    det = bd.BoundaryDetector(model, cfg)
    res = [bd.evaluate_stream(bd.run_stream(det, s["t"], s["xy"], s["mask"], s.get("face"), s.get("face_mask")),
                              s["t"], s["segments"]) for s in sentences]
    return bd.summarize(res)


def _fmt(m):
    lat = m["latency_median_s"]
    return "프레임 F1 %.3f, 구간 F1 %.3f, IoU %.3f, 지연 중앙값 %s, 문장 끝 1회 비율 %.2f" % (
        m["frame_f1"], m["segment_f1"], m["mean_iou"], "-" if lat is None else "%.0f ms" % (lat * 1000), m["sentence_end_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="*", default=[], help="build_dataset.py --full-fps 15 결과 폴더")
    ap.add_argument("--real-train", nargs="*", default=[], help="실제 문장 npz 폴더(학습). --features 도 주면 의사 문장을 섞는다")
    ap.add_argument("--real-test", nargs="*", default=[], help="실제 문장 npz 폴더(시험). 학습 수행자와 겹치면 안 된다. 비우면 검증 문장으로 점수를 낸다(최종 모델용)")
    ap.add_argument("--real-val-frac", type=float, default=0.1, help="실제 학습 문장 중 검증으로 뺄 비율")
    ap.add_argument("--synthetic", action="store_true", help="실제 데이터 대신 합성 문장으로 학습·평가(동작 확인용)")
    ap.add_argument("--split", default=os.path.join(REPO, "configs", "split-6fold.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arch", default="tcn", choices=["tcn", "gru"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--crop", type=int, default=96, help="학습 조각 길이(프레임). GRU 는 실시간 입력 창도 이 길이")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--words", default="2,5", help="문장당 단어 수 범위")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "boundary"))
    ap.add_argument("--save-model", default="", help=".pt 저장 경로(레포 밖)")
    ap.add_argument("--trial", default="", help="주면 trials.md 에 이 제목으로 기록한다")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--attempt", default="")
    ap.add_argument("--decision", default="(작성자가 채울 것)")
    ap.add_argument("--trial-log", default=os.path.join(REPO, "logs", "sign-language", "trials.md"))
    a = ap.parse_args()
    if a.save_model and db._inside_repo(a.save_model) and not a.synthetic:
        raise SystemExit("모델은 데이터 파생물이라 레포 안에 저장하지 않는다")
    words = tuple(int(x) for x in a.words.split(","))

    if a.synthetic:
        def gen(n, seed):
            rng = np.random.default_rng(seed)
            return [db.synthetic_sentence(rng, int(rng.integers(words[0], words[1] + 1)), fingerspell_prob=0.2) for _ in range(n)]
        parts = {"train": gen(a.n_train, 100 + a.seed), "val": gen(a.n_val, 200), "test": gen(a.n_test, 300)}
        fold = {"train": [], "val": [], "test": []}
        tag = "synthetic-%s-s%d" % (a.arch, a.seed)
    elif a.real_train:
        real = db.load_real_sentences(a.real_train)
        test = db.load_real_sentences(a.real_test) if a.real_test else []
        tr_s, te_s = {s["signer"] for s in real}, {s["signer"] for s in test}
        if tr_s & te_s:
            raise SystemExit("학습·시험 수행자가 겹친다: %s" % sorted(tr_s & te_s))
        if not real or (a.real_test and not test):
            raise SystemExit("--real-train, --real-test 폴더에 문장 npz 가 없다")
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(real))
        nv = max(1, int(len(real) * a.real_val_frac))
        parts = {"train": [real[i] for i in idx[nv:]], "val": [real[i] for i in idx[:nv]], "test": test or [real[i] for i in idx[:nv]]}
        if not test:                                      # 최종 모델용: 시험 수행자 없이 전부 학습, 점수는 검증 문장(같은 수행자라 낙관적)
            print("시험 수행자 없음: 시험 점수는 검증 문장으로 낸다(낙관적)", flush=True)
        if a.features:                                    # 의사 문장을 섞는다(학습 수행자와 같은 사람만)
            clips = db.load_full_clips(a.features, signers=tr_s)
            parts["train"] += db.make_sentences(clips, a.n_train, words, a.seed) if clips else []
        fold = {"train": sorted(tr_s), "val": sorted(tr_s), "test": sorted(te_s)}
        tag = "real-%s-tr%s-te%s-s%d%s" % (a.arch, "".join("%02d" % x for x in sorted(tr_s)),
                                            "".join("%02d" % x for x in sorted(te_s)), a.seed, "-mix" if a.features else "")
    else:
        if not a.features:
            raise SystemExit("--features, --real-train 또는 --synthetic 이 필요하다")
        with open(a.split, encoding="utf-8") as f:
            split = json.load(f)
        fold = split["folds"][a.fold]
        clips = db.load_full_clips(a.features)
        parts = {}
        for k, n, seed in (("train", a.n_train, a.seed), ("val", a.n_val, 1000), ("test", a.n_test, 2000)):
            sel = [c for c in clips if c["signer"] in set(fold[k])]
            if not sel:
                raise SystemExit("%s 수행자 %s 의 영상 전체 npz 가 없다" % (k, fold[k]))
            parts[k] = db.make_sentences(sel, n, words, seed)
        tag = "%s-f%d-s%d" % (a.arch, a.fold, a.seed)
    print("문장 학습 %d, 검증 %d, 시험 %d" % tuple(len(parts[k]) for k in ("train", "val", "test")), flush=True)

    t0 = time.time()
    tr, va = prepare(parts["train"]), prepare(parts["val"])
    print("특징 준비 %.0f초" % (time.time() - t0), flush=True)
    t0 = time.time()
    model, best, hist = train(a.arch, tr, va, a.epochs, a.patience, a.seed, crop=a.crop)
    train_s = time.time() - t0
    cfg = bd.DetectorConfig(window=bd.model_window(model, a.crop))   # 시험·저장 모두 학습과 같은 입력 창
    print("실시간 입력 창 %d프레임(학습 조각 %d)" % (cfg.window, a.crop), flush=True)
    te_model = stream_eval(parts["test"], model, cfg)
    te_rule = stream_eval(parts["test"], None, cfg)
    print("시험(모델):", _fmt(te_model), flush=True)
    print("시험(규칙):", _fmt(te_rule), flush=True)

    res = {"arch": a.arch, "fold": a.fold, "seed": a.seed, "synthetic": a.synthetic, "git": git_state(),
           "split": {k: fold[k] for k in ("train", "val", "test")}, "words": list(words),
           "train_crop": a.crop, "n": {k: len(v) for k, v in parts.items()}, "feat_names": bd.FEAT_NAMES, "detector": asdict(cfg),
           "params": sum(p.numel() for p in model.parameters()), "train_s": train_s, "best_val_macro_f1": best,
           "history": hist, "test_model": te_model, "test_rule": te_rule}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, tag + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("결과 ->", path)
    if a.save_model:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_model)), exist_ok=True)
        torch.save({"state": model.state_dict(), "arch": a.arch, "n_feat": bd.N_FEAT, "feat_names": bd.FEAT_NAMES,
                    "detector": asdict(cfg), "train_crop": a.crop, "split": res["split"], "git": res["git"]}, a.save_model + ".tmp")
        os.replace(a.save_model + ".tmp", a.save_model)
        print("모델 ->", a.save_model)
    if a.trial:
        try:
            shown = os.path.relpath(path, REPO)
        except ValueError:                    # 결과 폴더가 다른 드라이브면 상대 경로를 만들 수 없다
            shown = path
        tid = bd.append_trial(a.trial_log, a.trial, a.hypothesis or "(작성자가 채울 것)",
                              a.attempt or "%s, 폴드 %d, 시드 %d, 문장 %d개(단어 %d~%d)%s" % (
                                  a.arch, a.fold, a.seed, len(parts["train"]), words[0], words[1], ", 합성" if a.synthetic else ""),
                              "모델: %s / 규칙: %s (%s)" % (_fmt(te_model), _fmt(te_rule), shown),
                              a.decision)
        print("실험 기록 %s -> %s" % (tid, a.trial_log))


if __name__ == "__main__":
    main()
