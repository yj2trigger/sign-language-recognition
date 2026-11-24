"""문장 속 단어로 단어 모델 미세조정: AI-Hub 문장 영상에서 어휘 안 단어 구간을 잘라, 저장된 단어 모델을 몇 에폭 더 학습한다.

사용: python src/finetune_sentence_words.py <기본 모델.pt> --train <npz 폴더>,<정답.csv> [...] --out <새 모델.pt>
      [--epochs 15] [--lr 2e-4] [--margin 0] [--bank 4]
시험 수행자의 문장은 --train 에 넣지 않는다(수행자 독립). 원래 학습과 같은 증강(augmented_bank)·특징(.pt 의 config)·손실
(라벨 스무딩 0.1)을 쓰고 학습률만 낮춘다. 검증 세트가 없어 정한 에폭까지 학습한 마지막 가중치를 저장한다.
"""
import argparse
import copy
import os

import numpy as np
import torch
from torch import nn

import dataset as ds
from eval_video_words import load_labels, load_npz, load_word

MIN_FRAMES = 8          # dataset.augment 의 시간 자르기 최소 길이


def sentence_clips(pairs, vocab, margin):
    """(npz 폴더, 정답 CSV) 들에서 어휘 안 단어 구간을 클립(dataset.to_arrays 가 받는 형태)으로 자른다."""
    clips = []
    for npz_dir, labels_csv in pairs:
        for vid, segs in load_labels(labels_csv, "").items():
            path = os.path.join(npz_dir, vid + ".npz")
            if not os.path.exists(path):
                continue
            d, aspect = load_npz(path)
            n = len(d["t"])
            for s, e, w in segs:
                idx = np.flatnonzero((d["t"] >= s - margin) & (d["t"] <= e + margin))
                if w not in vocab or len(idx) == 0:
                    continue
                if len(idx) < MIN_FRAMES:      # 증강의 시간 자르기가 최소 8프레임이라 짧은 단어는 가운데 기준으로 넓힌다
                    lo = min(max(0, (idx[0] + idx[-1]) // 2 - MIN_FRAMES // 2), max(0, n - MIN_FRAMES))
                    idx = np.arange(lo, min(n, lo + MIN_FRAMES))
                clips.append({"xy": d["xy"][idx], "mask": d["mask"][idx], "face": d["face"][idx],
                              "face_mask": d["face_mask"][idx], "aspect": aspect, "word": w, "signer": 0})
    return clips


def finetune(model, clips, classes, cfg, epochs, lr, bank, seed=0):
    feat = {k: cfg.get(k, d) for k, d in (("angles", False), ("traj", False), ("bones", False), ("origin", "shoulder"),
                                          ("hand_z", False), ("folds", False), ("face", False))}
    Xb, y = ds.augmented_bank(clips, classes, cfg["length"], cfg["hand_rel"], bank, seed, fps_aug=cfg.get("fps_aug", False),
                              pos_aug=cfg.get("pos_aug", False), **feat)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(epochs):
        X = Xb[rng.integers(0, len(Xb), len(y)), np.arange(len(y))]      # 표본마다 증강본 한 벌(원래 학습과 같게)
        model.train()
        order, tot = rng.permutation(len(y)), 0.0
        for i in range(0, len(y), cfg["batch"]):
            b = order[i:i + cfg["batch"]]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(X[b]).float()), torch.from_numpy(y[b]))
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        print("에폭 %d 손실 %.4f" % (epoch + 1, tot / len(y)), flush=True)
    model.eval()
    return model, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--train", nargs="+", required=True, help="<npz 폴더>,<정답 CSV> 쌍")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4, help="원래 학습 2e-3 의 10분의 1")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--bank", type=int, default=4)
    ap.add_argument("--replay-dirs", nargs="*", default=[], help="섞을 원래 단어 특징 폴더(기본 모델의 학습 수행자 것만)")
    ap.add_argument("--replay-n", type=int, default=0, help="그 폴더에서 무작위로 뽑아 섞을 단어 클립 수(망각 방지)")
    a = ap.parse_args()
    model, classes, cfg, vocab = load_word(a.base)
    clips = sentence_clips([p.split(",", 1) for p in a.train], vocab, a.margin)
    print("문장 속 단어 클립 %d개(서로 다른 단어 %d개)" % (len(clips), len({c["word"] for c in clips})), flush=True)
    if a.replay_n:
        words = ds.load_clips(a.replay_dirs, classes)
        test = set(torch.load(a.base, weights_only=False)["split"]["test"])
        assert not any(c["signer"] in test for c in words), "재생 데이터에 시험 수행자가 섞임"
        pick = np.random.default_rng(0).choice(len(words), min(a.replay_n, len(words)), replace=False)
        clips += [words[i] for i in pick]
        print("원래 단어 클립 %d개 섞음(후보 %d개, 수행자 %s)" % (len(pick), len(words), sorted({c["signer"] for c in words})), flush=True)
        del words
    model, n = finetune(model, clips, classes, cfg, a.epochs, a.lr, a.bank)
    ck = torch.load(a.base, weights_only=False)
    ck = dict(ck, state=copy.deepcopy(model.state_dict()),
              finetune={"base": a.base, "train": a.train, "epochs": a.epochs, "lr": a.lr, "margin": a.margin, "bank": a.bank,
                        "n_clips": n, "replay_dirs": a.replay_dirs, "replay_n": a.replay_n})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(ck, a.out)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
