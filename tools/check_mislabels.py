"""라벨 이름이 다른 단어를 가리키는 클립(select_classes.py --rule v2 --review 출력)을 좌표로 판정한다.

사용: python tools/check_mislabels.py <확인대상.json> <출력.json> <좌표 폴더...>
판정: 클립과 자기 번호(다른 수행자 클립) 사이 거리 d_own, 라벨이 가리킨 번호의 클립과의 거리 d_other 를 잰다.
  d_own 이 같은 단어·다른 수행자 거리의 95% 범위 안이고 d_other 보다 작으면 영상은 맞고 이름만 틀린 것으로 보고 남긴다.
  그 밖은 영상이 다른 단어일 수 있으므로 뺀다(exclude). 클립이나 비교할 같은 번호 클립이 아직 없으면 판정하지 않는다(missing).
거리: build_features(64프레임)의 프레임별 유클리드 거리 평균, 여러 클립과는 중앙값.
영상 전체 npz(build_dataset --full-fps)는 학습과 같게 라벨 구간 ±0.5초로 잘라 비교한다.
출력에는 번호만 담는다(단어 이름은 넣지 않는다).
"""
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dataset as ds  # noqa: E402
import preprocess as pp  # noqa: E402

NAME = re.compile(r"WORD(\d+)_REAL(\d+)_F\.npz$")


def dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def main():
    review, out = sys.argv[1:3]
    items = json.load(open(review, encoding="utf-8"))["items"]
    paths = {}
    for d in sys.argv[3:]:
        for f in glob.glob(os.path.join(d, "*.npz")):
            m = NAME.search(f)
            if m:
                paths[(int(m.group(1)), int(m.group(2)))] = f
    by_word = {}
    for (w, s) in paths:
        by_word.setdefault(w, []).append(s)
    cache = {}

    def feat(w, s):
        if (w, s) not in cache:
            z = np.load(paths[(w, s)])
            seg = ds.segment(z)
            cache[(w, s)] = pp.build_features(z["xy"][seg], z["mask"][seg], 64)
        return cache[(w, s)]

    # 기준: 같은 단어를 다른 수행자가 한 거리. 확인 대상이 아닌 단어에서 수행자 쌍을 뽑는다.
    flagged = {(i["word"], i["signer"]) for i in items}
    rng = random.Random(0)
    same = []
    for w in rng.sample(sorted(by_word), min(200, len(by_word))):
        ss = [s for s in by_word[w] if (w, s) not in flagged]
        for _ in range(min(10, len(ss) * (len(ss) - 1) // 2)):
            a, b = rng.sample(ss, 2)
            same.append(dist(feat(w, a), feat(w, b)))
    if len(same) < 20:
        raise SystemExit("같은 단어·다른 수행자 쌍이 %d개뿐이라 기준을 정할 수 없다(수행자 2명 이상의 좌표 폴더를 줄 것)" % len(same))
    thr = float(np.percentile(same, 95))
    print("같은 단어·다른 수행자 거리: %d쌍, 중앙값 %.2f, 95%% %.2f" % (len(same), np.median(same), thr), flush=True)

    res, exclude = [], []
    for i in items:
        w, s = i["word"], i["signer"]
        if (w, s) not in paths:
            res.append(dict(i, status="missing"))
            continue
        x = feat(w, s)
        own = [dist(x, feat(w, t)) for t in by_word[w] if t != s and (w, t) not in flagged]
        other = [dist(x, feat(v, t)) for v in i["points_to"] for t in by_word.get(v, [])]
        if not own:                                   # 비교할 같은 번호 클립이 아직 없으면 판정하지 않는다
            res.append(dict(i, status="missing"))
            continue
        d_own = float(np.median(own))
        d_other = float(np.median(other)) if other else None
        keep = d_own <= thr and (d_other is None or d_own < d_other)
        res.append(dict(i, d_own=d_own, d_other=d_other, status="keep" if keep else "exclude"))
        if not keep:
            exclude.append([w, s])
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "n_same_pairs": len(same), "items": res, "exclude": sorted(exclude)}, f)
    count = {k: sum(r["status"] == k for r in res) for k in ("keep", "exclude", "missing")}
    print("판정 %s -> %s" % (count, out))


if __name__ == "__main__":
    main()
