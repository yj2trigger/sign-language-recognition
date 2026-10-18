"""runs/<실험>/*.json 을 모아 모델별 요약표와 분할 누수 검사를 출력한다.
사용: python src/summarize_runs.py runs/e1-10signers
- 요약은 폴드 단위로 먼저 시드를 평균한 뒤 폴드 간 평균과 표준편차를 낸다(폴드가 독립 시험 단위).
- 누수 검사: 각 실행의 학습·검증·시험 수행자가 서로 겹치지 않는지, 모든 모델이 같은 분할 해시를 썼는지 확인한다.
"""
import collections
import glob
import json
import sys

import numpy as np

runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(sys.argv[1].rstrip("/\\") + "/*.json"))]
print("실행 %d개, 모델 %s, 분할 해시 %s, 커밋 %s" % (len(runs), sorted({r["model"] for r in runs}),
                                                     sorted({r["split_sha"] for r in runs}), sorted({r["git"] for r in runs})))

bad = [(r["model"], r["fold"]) for r in runs
       if set(r["split"]["train"]) & set(r["split"]["val"]) or set(r["split"]["train"]) & set(r["split"]["test"])
       or set(r["split"]["val"]) & set(r["split"]["test"])]
print("분할 겹침이 있는 실행:", bad if bad else "없음")
tests = collections.Counter(s for f in {r["fold"]: r["split"]["test"] for r in runs}.values() for s in f)
print("폴드별 시험 수행자:", {f: v["test"] for f, v in sorted({r["fold"]: r["split"] for r in runs}.items())})
print("수행자별 시험 횟수:", dict(tests))

print("\n모델 | 파라미터 | 시험 macro-F1 (폴드 평균 ± 표준편차) | 시험 정확도 | top5 | 폴드별 F1 | 평균 학습 시간(초)")
for m in ("mlp", "gru", "tcn"):
    rs = [r for r in runs if r["model"] == m]
    if not rs:
        continue
    folds = sorted({r["fold"] for r in rs})
    f1 = [np.mean([r["test"]["macro_f1"] for r in rs if r["fold"] == f]) for f in folds]
    acc = [np.mean([r["test"]["acc"] for r in rs if r["fold"] == f]) for f in folds]
    t5 = [np.mean([r["test"]["top5"] for r in rs if r["fold"] == f]) for f in folds]
    seed_sd = np.mean([np.std([r["test"]["macro_f1"] for r in rs if r["fold"] == f]) for f in folds])
    print("%s | %d | %.3f ± %.3f | %.3f | %.3f | %s | %.0f  (시드 간 표준편차 평균 %.3f)" % (
        m, rs[0]["params"], np.mean(f1), np.std(f1), np.mean(acc), np.mean(t5), " ".join("%.2f" % x for x in f1),
        np.mean([r["train_sec"] for r in rs]), seed_sd))
