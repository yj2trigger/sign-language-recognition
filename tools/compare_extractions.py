"""같은 클립을 두 방식으로 추출한 결과(예: 원본 vs 9:16 자른 뒤 16:9 여백)를 점 단위로 비교한다.

사용: python tools/compare_extractions.py <기준 폴더> <비교 폴더>
출력: 손 검출률(기준/비교), 둘 다 검출된 점의 좌표 차이 중앙값·95%(어깨너비 단위).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import preprocess as pp  # noqa: E402


def main():
    ref_dir, cmp_dir = sys.argv[1:3]
    hand = {"ref": [], "cmp": []}
    body, hands, n = [], [], 0
    for f in sorted(glob.glob(os.path.join(cmp_dir, "*.npz"))):
        r = os.path.join(ref_dir, os.path.basename(f))
        if f.endswith(".tmp.npz") or not os.path.exists(r):
            continue
        a, b = np.load(r), np.load(f)
        T = min(len(a["xy"]), len(b["xy"]))
        xa, ma, xb, mb = a["xy"][:T], a["mask"][:T], b["xy"][:T], b["mask"][:T]
        hand["ref"].append(ma[:, [11, 32]].mean())
        hand["cmp"].append(mb[:, [11, 32]].mean())
        width = np.linalg.norm(xa[:, pp.L_SH] - xa[:, pp.R_SH], axis=-1)
        w = np.median(width[ma[:, pp.L_SH] & ma[:, pp.R_SH]]) if (ma[:, pp.L_SH] & ma[:, pp.R_SH]).any() else np.nan
        d = np.linalg.norm(xa - xb, axis=-1) / w
        both = ma & mb
        body.append(d[:, :11][both[:, :11]])
        hands.append(d[:, 11:][both[:, 11:]])
        n += 1
    body, hands = np.concatenate(body), np.concatenate(hands)
    print("클립 %d개" % n)
    print("손 검출률(프레임×손): 기준 %.4f, 비교 %.4f" % (np.mean(hand["ref"]), np.mean(hand["cmp"])))
    for k, v in (("상체", body), ("손", hands)):
        print("%s 좌표 차이(어깨너비 단위): 중앙값 %.4f, 95%% %.4f, 99%% %.4f" % (k, *np.nanpercentile(v, [50, 95, 99])))


if __name__ == "__main__":
    main()
