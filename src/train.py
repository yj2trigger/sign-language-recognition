"""수행자 독립 분할에서 GRU 와 TCN 을 같은 조건으로 학습·평가하고 결과를 runs/ 에 json 으로 남긴다.

같은 조건 = 같은 클래스 목록, 같은 분할(수행자 폴드), 같은 시드, 같은 증강·옵티마이저·조기 종료 기준.
사용: python src/train.py --features <폴더>... --classes configs/classes-100.json [--models gru,tcn] [--seeds 0,1,2]
      [--folds 6] [--max-folds 0] [--epochs 60] [--patience 12] [--length 64] [--out runs]
결과 json 에는 설정·시드·분할 해시·커밋 해시·지표가 들어가 어떤 코드와 설정에서 나온 값인지 추적할 수 있다.
"""
import argparse
import copy
import json
import os
import subprocess
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

import dataset as ds
import models
from winpower import keep_awake, no_power_throttling


def git_state():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
        return h + ("+수정중" if dirty else "")
    except Exception:
        return "알 수 없음"


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def metrics(logits, y):
    pred = logits.argmax(1)
    out = {"acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    if logits.shape[1] >= 5:
        out["top5"] = float((np.argsort(-logits, 1)[:, :5] == y[:, None]).any(1).mean())
    return out


def train_one(name, bank, va, classes, cfg, seed):
    """bank = (Xb (k, N, T, F), y (N,)) 미리 만든 증강본, va = (X, y). 검증 macro-F1 이 가장 좋았던 가중치를 돌려준다."""
    Xb, y = bank
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_feat = va[0].shape[2]
    model = models.build(name, n_feat, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, bad, hist = -1.0, None, 0, []
    for epoch in range(cfg["epochs"]):
        X = Xb[rng.integers(0, len(Xb), len(y)), np.arange(len(y))]          # 표본마다 증강본 한 벌을 무작위로 고른다
        model.train()
        order = rng.permutation(len(X))
        for i in range(0, len(X), cfg["batch"]):
            b = order[i:i + cfg["batch"]]
            if len(b) < 2:                                    # BatchNorm 은 표본 1개 배치를 못 받는다
                continue
            opt.zero_grad()
            loss_fn(model(torch.from_numpy(X[b])), torch.from_numpy(y[b])).backward()
            opt.step()
        sched.step()
        f1 = metrics(predict(model, va[0]), va[1])["macro_f1"]
        hist.append(round(f1, 4))
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--models", default="gru,tcn")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--bank", type=int, default=8, help="미리 만들 증강본 벌 수")
    ap.add_argument("--fps-aug", action="store_true", help="학습 증강에 20fps 흉내(프레임 건너뛰기)를 섞는다")
    ap.add_argument("--test-ratio", type=float, default=1.0, help="검증·시험 클립의 프레임 간격 배수(1.5 = 20fps 카메라 흉내)")
    ap.add_argument("--no-hand-rel", action="store_true")
    ap.add_argument("--angles", action="store_true", help="손가락 관절 각도 채널(30)을 더한다")
    ap.add_argument("--traj", action="store_true", help="평균 위치를 뺀 손목 궤적 채널(4)을 더한다")
    ap.add_argument("--pos-aug", action="store_true", help="학습 증강에 두 손 위치 이동(어깨너비 ±30%)을 섞는다")
    ap.add_argument("--bones", action="store_true", help="손 뼈대 길이 비율 채널(40, 2D 로 근사한 깊이 방향 기울기)을 더한다")
    ap.add_argument("--hand-z", action="store_true", help="손가락 깊이 채널(42, MediaPipe 손 z)을 더한다. features-z 추출 결과 필요")
    ap.add_argument("--origin", default="shoulder", choices=["shoulder", "nose"], help="좌표 원점: 어깨 중점 또는 코(얼굴 기준)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    print("효율 모드 제외 요청:", "성공" if no_power_throttling() else "적용 안 됨", flush=True)
    print("절전 방지 요청:", "성공" if keep_awake() else "적용 안 됨", flush=True)
    cfg = {"lr": 2e-3, "wd": 1e-2, "batch": 64, "epochs": a.epochs, "patience": a.patience,
           "length": a.length, "hand_rel": not a.no_hand_rel, "bank": a.bank,
           "fps_aug": a.fps_aug, "test_ratio": a.test_ratio,
           "angles": a.angles, "traj": a.traj, "pos_aug": a.pos_aug, "bones": a.bones, "origin": a.origin, "hand_z": a.hand_z}
    feat = {"angles": a.angles, "traj": a.traj, "bones": a.bones, "origin": a.origin, "hand_z": a.hand_z}

    with open(a.classes, encoding="utf-8") as f:
        classes = sorted(json.load(f)["word_ids"])
    clips = ds.load_clips(a.features, classes)
    # 선정한 단어를 거의 다 수행한 수행자만 쓴다(일부만 받은 수행자가 시험 집합을 흔들지 않게)
    per = {}
    for c in clips:
        per[c["signer"]] = per.get(c["signer"], 0) + 1
    signers = sorted(s for s, n in per.items() if n >= 0.9 * len(classes))
    print("클립 %d개, 사용 수행자 %d명 %s, 클래스 %d개" % (len(clips), len(signers), signers, len(classes)), flush=True)
    if len(signers) < 3 * 2:
        raise SystemExit("수행자 독립 분할에는 최소 6명(학습·검증·시험 각 2명 이상)이 필요하다")
    n_folds = min(a.folds, len(signers) // 2)
    folds = ds.make_folds(signers, n_folds)
    split_hash, class_hash = ds.sha(folds), ds.sha(classes)
    os.makedirs(a.out, exist_ok=True)
    git = git_state()

    for fi, fold in enumerate(folds[:a.max_folds] if a.max_folds else folds):
        todo = [(n, s) for n in a.models.split(",") for s in a.seeds.split(",")
                if not os.path.exists(os.path.join(a.out, "%s-f%d-s%s.json" % (n, fi, s)))]
        if not todo:                                  # 재시작 시 다 끝난 폴드는 데이터 준비(증강본 생성)도 건너뛴다
            print("폴드 %d 건너뜀(모두 끝남)" % fi, flush=True)
            continue
        sel = lambda names: [c for c in clips if c["signer"] in set(fold[names])]
        va = ds.to_arrays(sel("val"), classes, cfg["length"], cfg["hand_rel"], fps_ratio=cfg["test_ratio"], **feat)
        te = ds.to_arrays(sel("test"), classes, cfg["length"], cfg["hand_rel"], fps_ratio=cfg["test_ratio"], **feat)
        bank = ds.augmented_bank(sel("train"), classes, cfg["length"], cfg["hand_rel"], cfg["bank"], seed=fi,
                                 fps_aug=cfg["fps_aug"], pos_aug=cfg["pos_aug"], **feat)
        for name in a.models.split(","):
            for seed in (int(s) for s in a.seeds.split(",")):
                path = os.path.join(a.out, "%s-f%d-s%d.json" % (name, fi, seed))
                if os.path.exists(path):              # 재시작 뒤 다시 실행하면 끝난 실행은 건너뛴다
                    print("건너뜀(이미 있음):", path, flush=True)
                    continue
                t = time.time()
                model, best, hist = train_one(name, bank, va[:2], classes, cfg, seed)
                logits = predict(model, te[0])
                res = {"model": name, "fold": fi, "seed": seed, "params": models.n_params(model),
                       "val_macro_f1": best, "test": metrics(logits, te[1]),
                       "test_true": te[1].tolist(), "test_pred": logits.argmax(1).tolist(),   # 혼동 분석용(클래스 인덱스, classes 정렬 순서)
                       "test_signer": te[2].tolist(),
                       "epochs_run": len(hist), "train_sec": round(time.time() - t, 1), "val_curve": hist,
                       "split": fold, "split_sha": split_hash, "classes_sha": class_hash,
                       "n_classes": len(classes), "config": cfg, "git": git}
                res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                part = path + ".tmp"                      # 쓰는 도중 끊겨도 불완전한 json 이 남지 않게
                with open(part, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False)
                os.replace(part, path)
                print("%s fold%d seed%d: val F1 %.3f | test acc %.3f F1 %.3f | %d에폭 %.0f초 -> %s" % (
                    name, fi, seed, best, res["test"]["acc"], res["test"]["macro_f1"], len(hist), res["train_sec"], path), flush=True)


if __name__ == "__main__":
    main()
