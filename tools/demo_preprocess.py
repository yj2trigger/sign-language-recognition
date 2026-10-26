"""사진·설명용: 클립 하나가 전처리 단계를 거치며 어떻게 바뀌는지 터미널에 출력하고 그림 한 장으로 그린다.
학습과 같은 preprocess 함수를 그대로 부른다. 원본 영상 프레임은 쓰지 않고 좌표 스켈레톤만 그린다.

사용: python tools/demo_preprocess.py [클립 npz] [출력 png]
기본: D:/sign-language-data/tmp/photo-demo 에서 두 손을 가장 높이 든 클립 -> D:/sign-language-data/tmp/preprocess-demo.png
(좌표는 데이터 파생물이라 출력 그림은 레포 밖에 둔다)
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import preprocess as pp  # noqa: E402
from figures_w1 import ARMS, HAND, HEAD  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
LEN = 64


def skeleton(ax, xy, mask, title):
    for a, b in ARMS + HEAD:
        if mask[a] and mask[b]:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], "-", color="#555", lw=1.5)
    for off, color in ((11, "#d1495b"), (32, "#2e86ab")):
        if mask[off]:
            for a, b in HAND:
                ax.plot(xy[[off + a, off + b], 0], xy[[off + a, off + b], 1], "-", color=color, lw=1)
            ax.scatter(xy[off:off + 21, 0], xy[off:off + 21, 1], s=4, color=color)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def raised_frame(xy, mask):
    """두 손이 다 보이는 프레임 중 낮은 쪽 손목도 가장 높이 든 프레임과 그 높이(정규화 y, 작을수록 높음)."""
    norm = pp.normalize(xy, mask)
    both = np.where(mask[:, 11] & mask[:, 32])[0]
    if norm is None or not len(both):
        return len(xy) // 2, np.inf
    low = np.maximum(norm[both, 11, 1], norm[both, 32, 1])
    return int(both[np.argmin(low)]), float(low.min())


def main():
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        clips = sorted(glob.glob("D:/sign-language-data/tmp/photo-demo/*.npz"))
        src = min(clips, key=lambda f: raised_frame(np.load(f)["xy"], np.load(f)["mask"])[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "D:/sign-language-data/tmp/preprocess-demo.png"
    z = np.load(src)
    xy, mask = z["xy"], z["mask"]
    T = len(xy)
    print("입력 클립: %s (단어 번호 %d, 수행자 %d)" % (os.path.basename(src), int(z["word_id"]), int(z["signer"])))

    print("① 추출 좌표       %s  화면 비율 좌표(0~1), 프레임 %d개" % (xy.shape, T))
    miss = [(~mask[:, s]).sum() for s in (11, 32)]
    print("                   손이 안 잡힌 프레임: 왼손 %d, 오른손 %d" % tuple(miss))

    norm = pp.normalize(xy, mask)
    sh = (norm[:, pp.L_SH] + norm[:, pp.R_SH]) / 2
    print("② 어깨 정규화     어깨 중점 -> (%.2f, %.2f), 어깨너비 -> %.2f" % (
        *np.median(sh, 0), np.median(np.linalg.norm(norm[:, pp.L_SH] - norm[:, pp.R_SH], axis=-1))))

    filled = pp.fill_missing(norm, mask)
    changed = int((~mask).any(1).sum())
    print("③ 결측 채우기     빈 점이 있던 프레임 %d개 채움(5프레임 이하는 보간, 그 밖은 직전 값 유지)" % changed)

    hand = pp.hand_relative(filled, mask)
    print("④ 손 상대 좌표    %s 추가: 손목 원점, 손 크기 1(손 모양만 남김)" % (hand.shape,))

    feat = pp.build_features(xy, mask, LEN)
    print("⑤ 64프레임 리샘플 %d -> %d프레임, 모델 입력 %s = 몸 106 + 손 84 + 마스크 53" % (T, LEN, feat.shape))

    t = raised_frame(xy, mask)[0]
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 2])
    shown = xy[t].copy()
    shown[:, 0] *= float(z["wh"][0]) / float(z["wh"][1]) if "wh" in z.files else 16 / 9   # 표시용: 화소 비율로 펴기
    skeleton(fig.add_subplot(gs[0, 0]), shown, mask[t], "① 추출 좌표\n(화면 높이 = 1, 가로는 화소 비율로 폄)")
    ax = fig.add_subplot(gs[0, 1])
    skeleton(ax, filled[t], mask[t], "② 어깨 정규화\n(어깨 중점 0, 어깨너비 1)")
    ax.axhline(0, color="#bbb", lw=0.8); ax.axvline(0, color="#bbb", lw=0.8)
    for k, (s, name, color) in enumerate(((0, "왼손", "#d1495b"), (21, "오른손", "#2e86ab"))):
        ax = fig.add_subplot(gs[0, 2 + k])
        h = hand[t, s:s + 21]
        for a, b in HAND:
            ax.plot(h[[a, b], 0], h[[a, b], 1], "-", color=color, lw=1.2)
        ax.scatter(h[:, 0], h[:, 1], s=8, color=color)
        ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_title("④ %s 상대 좌표\n(손목 원점, 손 크기 1)" % name, fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    w, wname = (11, "왼손목") if (~mask[:, 11]).sum() > (~mask[:, 32]).sum() else (32, "오른손목")   # 결측이 많은 손을 보여 준다
    wy = filled[:, w, 1]                                          # 손목 높이(정규화 좌표)
    ax.plot(np.arange(T), wy, "o-", ms=3, color="#999", label="정규화·결측 채움 후 (%d프레임)" % T)
    ax.plot(np.linspace(0, T - 1, LEN), feat[:, w * 2 + 1], ".-", color="#2e86ab", label="⑤ 64프레임 리샘플")
    gap = np.where(~mask[:, w])[0]
    if len(gap):
        ax.scatter(gap, wy[gap], marker="x", color="#d1495b", zorder=3, label="③ 손이 안 잡혀 채운 프레임")
    ax.axvline(t, color="#bbb", ls="--", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("원래 프레임 번호"); ax.set_ylabel("%s 높이\n(어깨너비 단위, 위가 높음)" % wname)
    ax.legend(fontsize=9, loc="best")
    fig.suptitle("전처리 단계별 변화 — 단어 %d, 수행자 %d, 프레임 %d (점선)" % (int(z["word_id"]), int(z["signer"]), t), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("그림 저장:", out)


if __name__ == "__main__":
    main()
