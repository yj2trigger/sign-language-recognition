"""1주차 보고서용 그림 3장을 만든다. 영상 프레임은 쓰지 않고 좌표 스켈레톤만 그린다(얼굴 노출·이용 조건 문제를 피함).
사용: python src/figures_w1.py <클립 npz 폴더> <출력 폴더>
만드는 것: W1-01 스켈레톤(한 클립의 네 시점), W1-02 클립 길이 분포, W1-03 손 검출률 분포. 그림마다 원본 수치를 data/*.csv 로 저장한다.
"""
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HAND = list(mp.solutions.hands.HAND_CONNECTIONS)
ARMS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]     # 어깨선, 왼팔(어깨-팔꿈치-손목), 오른팔 (상체 11점 안의 인덱스)
HEAD = [(1, 2), (3, 4)]                              # 눈, 귀 (코=0)


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def draw_frame(ax, xy, mask, title):
    for a, b in ARMS + HEAD:
        if mask[a] and mask[b]:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], "-", color="#555", lw=1.5)
    ax.scatter(xy[0:1, 0], xy[0:1, 1], s=25, color="#555")                 # 코
    for off, color in ((11, "#d1495b"), (32, "#2e86ab")):                  # 왼손 빨강, 오른손 파랑
        if mask[off]:
            for a, b in HAND:
                ax.plot(xy[[off + a, off + b], 0], xy[[off + a, off + b], 1], "-", color=color, lw=1)
            ax.scatter(xy[off:off + 21, 0], xy[off:off + 21, 1], s=3, color=color)
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(1.0, 0.05)       # 이미지 좌표(y가 아래로 증가)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    src, out = sys.argv[1:3]
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    clips = [np.load(f) for f in files]
    names = [os.path.basename(f)[:-4] for f in files]

    # W1-01: 한 클립의 네 시점(시작·1/3·2/3·끝)
    k = len(clips) // 2
    c = clips[k]
    T = len(c["xy"])
    idx = [0, T // 3, 2 * T // 3, T - 1]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    rows = []
    for ax, i in zip(axes, idx):
        draw_frame(ax, c["xy"][i], c["mask"][i], "프레임 %d/%d" % (i + 1, T))
        rows += [(names[k], i, p, float(c["xy"][i, p, 0]), float(c["xy"][i, p, 1]), int(c["mask"][i, p])) for p in range(53)]
    fig.suptitle("추출한 53점 (빨강: 왼손, 파랑: 오른손, 회색: 상체)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W1-01-skeleton.png"), dpi=170)
    plt.close(fig)
    save_csv(os.path.join(out, "data", "W1-01.csv"), ["clip", "frame", "point", "x", "y", "detected"], rows)

    # W1-02: 클립 길이 분포(앞뒤 0.5초 포함, 프레임 수)
    n = np.array([len(c["xy"]) for c in clips])
    p50, p95 = np.percentile(n, [50, 95])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(n, bins=40, color="#2e86ab")
    ax.axvline(p50, color="k", ls="--", lw=1)
    ax.axvline(p95, color="#d1495b", ls="--", lw=1)
    ax.axvline(64, color="#2a9d8f", ls="-", lw=1.5)
    ax.text(p50, ax.get_ylim()[1] * 0.92, " 중앙값 %d" % p50, fontsize=9)
    ax.text(p95, ax.get_ylim()[1] * 0.80, " p95 %d" % p95, color="#d1495b", fontsize=9)
    ax.text(64, ax.get_ylim()[1] * 0.68, "고정 길이 64 ", color="#2a9d8f", fontsize=9, ha="right")
    ax.set_xlabel("클립 프레임 수 (30fps, 단어 구간 앞뒤 0.5초 포함)")
    ax.set_ylabel("클립 수")
    ax.set_title("클립 길이 분포 (수행자 1명, 클립 %d개)" % len(n), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W1-02-clip-length.png"), dpi=170)
    plt.close(fig)
    save_csv(os.path.join(out, "data", "W1-02.csv"), ["clip", "frames"], list(zip(names, n.tolist())))

    # W1-03: 손 검출률 분포(클립별, 검출된 프레임 비율)
    left = np.array([c["mask"][:, 11].mean() for c in clips])
    right = np.array([c["mask"][:, 32].mean() for c in clips])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0.5, 1.0, 26)
    ax.hist(left, bins=bins, alpha=0.6, color="#d1495b", label="왼손")
    ax.hist(right, bins=bins, alpha=0.6, color="#2e86ab", label="오른손")
    ax.set_yscale("log")
    ax.set_xlabel("클립 안에서 손이 검출된 프레임 비율")
    ax.set_ylabel("클립 수 (로그 눈금)")
    ax.set_title("손 검출률 분포 (1.0 = 모든 프레임에서 검출)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W1-03-hand-detection.png"), dpi=170)
    plt.close(fig)
    save_csv(os.path.join(out, "data", "W1-03.csv"), ["clip", "left_rate", "right_rate"],
             [(nm, round(float(a), 4), round(float(b), 4)) for nm, a, b in zip(names, left, right)])
    print("그림 3장 저장:", out)


if __name__ == "__main__":
    main()
