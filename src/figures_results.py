"""실험 결과 그림: W3-01 모델 비교(E5), W3-02 어휘 크기 곡선(E5·E6), W3-03 지연(T-016), W3-04 혼동행렬(E6 클래스 300), W3-05 화면비(T-035).
결과 json 과 기록된 실측값만 읽는다(원본 데이터 없음, 단어 이름을 붙일 때만 좌표 폴더를 읽음).
사용: python src/figures_results.py figures [좌표 npz 폴더(혼동행렬에 단어 이름을 붙일 때)]
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
    # W3-03: 지연 3구간 (T-016 실측값, 1080p, CPU, 예열 후 반복 측정의 중앙값)
    parts = [("MediaPipe 추출 (complexity 1)", 43.2), ("특징 계산", 0.3), ("TCN 추론 (64프레임 창)", 1.9)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    left = 0
    for (name, med), color in zip(parts, ["#2e86ab", "#9aa5b1", "#d1495b"]):
        ax.barh(0, med, left=left, color=color, height=0.5, label="%s %.1f ms" % (name, med))
        left += med
    ax.axvline(1000 / 30, color="k", ls="--", lw=1)
    ax.text(1000 / 30 - 0.5, 0.42, "30fps 프레임 간격 33.3 ms", ha="right", fontsize=9)
    ax.axvline(1000 / 20, color="#555", ls=":", lw=1)
    ax.text(1000 / 20 - 0.5, -0.42, "20fps 간격 50 ms", ha="right", fontsize=9)
    ax.text(left + 0.8, 0, "함께 잰 합계\n45.6 ms (21.9 fps)", va="center", fontsize=10)   # 구간 중앙값의 합이 아니라 T-016 동시 측정값
    ax.set_xlim(0, 62); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_xlabel("프레임 1개 처리 시간 (ms, 중앙값)")
    ax.set_title("지연 3구간 분리 측정 (CPU 노트북, 1080p) — 병목은 MediaPipe", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W3-03-latency.png"), dpi=170)
    plt.close(fig)

    # W3-04: 혼동행렬 — 클래스 300, TCN, 시드 4개 × 6폴드의 예측을 합쳐 오류에 가장 많이 얽힌 단어만 보여 준다
    rs = load("e6-vocab-c300", "tcn")
    with open("configs/classes-300.json", encoding="utf-8") as f:
        classes = sorted(json.load(f)["word_ids"])
    t = np.concatenate([r["test_true"] for r in rs]); p = np.concatenate([r["test_pred"] for r in rs])
    cm = np.zeros((len(classes), len(classes)), int)
    np.add.at(cm, (t, p), 1)
    off = cm - np.diag(np.diag(cm))
    top = np.argsort(-(off.sum(0) + off.sum(1)))[:14]
    top = top[np.argsort([classes[i] for i in top])]
    names = gloss_names(sys.argv[2] if len(sys.argv) > 2 else "")
    sub = cm[np.ix_(top, top)] / cm[top].sum(1, keepdims=True)
    labels = ["%d %s" % (classes[i], names.get(classes[i], "")) for i in top]
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(top)):
        for j in range(len(top)):
            if sub[i, j] >= 0.005:
                ax.text(j, i, "%.0f" % (100 * sub[i, j]), ha="center", va="center", fontsize=7,
                        color="w" if sub[i, j] > 0.5 else "k")
    ax.set_xticks(range(len(top)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)), labels, fontsize=8)
    ax.set_xlabel("예측"); ax.set_ylabel("정답")
    fig.colorbar(im, ax=ax, fraction=0.04, label="정답 행 안 비율")
    ax.set_title("오류가 가장 많이 얽힌 단어 %d개의 혼동행렬 (클래스 300, TCN, 시드 4 × 6폴드)\n칸의 숫자: 정답 행 안의 %%, 대각선 = 맞힘, 행 합이 100이 안 되면 나머지는 이 표 밖 단어로 간 오답" % len(top), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W3-04-confusion.png"), dpi=170)
    plt.close(fig)

    # W3-05: 화면비 검증 (T-035, 수행자 3·4·11 × 100클립, 폴드 0 TCN 1개)
    rows = [("원본 16:9", 0.9933, None), ("핸드폰 세로 9:16", 1.0000, 0.4000), ("4:3", 0.9933, 0.9533)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color="#2e86ab", label="16:9 검은 틀(레터박스) 있음")
    ax.bar(x[1:] + 0.18, [r[2] for r in rows[1:]], 0.36, color="#d1495b", label="틀 없이 그대로")
    for i, (_, a, b) in enumerate(rows):
        ax.text(i - 0.18, a + 0.02, "%.3f" % a, ha="center", fontsize=9)
        if b is not None:
            ax.text(i + 0.18, b + 0.02, "%.3f" % b, ha="center", fontsize=9)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0, 1.3)
    ax.set_ylabel("정확도")
    ax.set_title("입력 화면비에 따른 정확도 (재학습 없음, 원본 영상을 잘라 흉내)\n같은 모델, 수행자 3명 × 100단어", fontsize=10)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W3-05-aspect.png"), dpi=170)
    plt.close(fig)
    print("그림 5장 저장:", out)


def gloss_names(feature_dir):
    """좌표 npz 폴더가 주어지면 단어 번호 -> 이름. 없으면 빈 사전(그림에 번호만 표시)."""
    names = {}
    for f in glob.glob(os.path.join(feature_dir, "*.npz")) if feature_dir else []:
        z = np.load(f)
        names.setdefault(int(z["word_id"]), str(z["gloss"]))
    return names


if __name__ == "__main__":
    main()
