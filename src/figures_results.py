"""실험 결과 그림: W3-01 모델 비교(E5), W3-02 어휘 크기 곡선(E5·E6). 결과 json 만 읽는다(원본 데이터 없음).
사용: python src/figures_results.py figures
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load(d, model=None, seeds=None):
    rs = [json.load(open(f, encoding="utf-8")) for f in glob.glob("runs/%s/*.json" % d)]
    return [r for r in rs if (model is None or r["model"] == model) and (seeds is None or r["seed"] in seeds)]


def fold_f1(rs):
    folds = sorted({r["fold"] for r in rs})
    return np.array([np.mean([r["test"]["macro_f1"] for r in rs if r["fold"] == f]) for f in folds])


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)

    # W3-01: 모델 비교 (수행자 18명, 클래스 100, 6폴드, 시드 3)
    e5 = load("e5-18signers-c100")
    names = [("mlp", "MLP 기준선\n(순서 안 씀)"), ("gru", "양방향 GRU"), ("tcn", "TCN")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (m, label) in enumerate(names):
        f = fold_f1([r for r in e5 if r["model"] == m])
        ax.bar(i, f.mean(), color=["#9aa5b1", "#2e86ab", "#d1495b"][i], width=0.6)
        ax.errorbar(i, f.mean(), yerr=f.std(), color="k", capsize=4)
        ax.scatter(np.full(len(f), i) + np.linspace(-0.15, 0.15, len(f)), f, s=12, color="k", zorder=3)
        ax.text(i, max(f.max(), f.mean() + f.std()) + 0.015, "%.3f" % f.mean(), ha="center", fontsize=10)   # 점·오차 막대 위에
    ax.set_xticks(range(3), [n for _, n in names])
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("시험 macro-F1")
    ax.set_title("수행자 독립 비교 (수행자 18명, 클래스 100, 6폴드 × 시드 3)\n점: 폴드별 값, 막대 위 선: 폴드 간 표준편차", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W3-01-model-comparison.png"), dpi=170)
    plt.close(fig)

    # W3-02: 어휘 크기 곡선 (TCN, 같은 분할, 시드 0·1)
    sets = [(50, "e6-vocab-c50"), (100, "e5-18signers-c100"), (200, "e6-vocab-c200"), (300, "e6-vocab-c300")]
    xs, means, sds, folds = [], [], [], []
    for n, d in sets:
        f = fold_f1(load(d, "tcn", (0, 1)))
        xs.append(n); means.append(f.mean()); sds.append(f.std()); folds.append(f)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(xs, means, yerr=sds, marker="o", color="#d1495b", capsize=4, label="폴드 평균 ± 표준편차")
    for x, f in zip(xs, folds):
        ax.scatter(np.full(len(f), x), f, s=10, color="#555", alpha=0.6)
    for x, m in zip(xs, means):
        ax.text(x + 9, m, "%.3f" % m, ha="left", va="center", fontsize=9)   # 오차 막대 오른쪽에
    ax.set_xticks(xs)
    ax.set_ylim(0.94, 1.0)
    ax.set_xlabel("클래스(단어) 수")
    ax.set_ylabel("시험 macro-F1")
    ax.set_title("어휘 크기에 따른 성능 (TCN, 수행자 18명, 같은 6폴드, 시드 2)\n회색 점: 폴드별 값", fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W3-02-vocab-curve.png"), dpi=170)
    plt.close(fig)
    print("그림 2장 저장:", out)


if __name__ == "__main__":
    main()
