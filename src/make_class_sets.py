"""포개어진 클래스 집합(50 ⊂ 100 ⊂ 200 ⊂ 300)을 만든다. 어휘 크기 곡선에서 집합마다 클래스가 달라 생기는 차이를 없애려는 것이다.

사용: python src/make_class_sets.py <Training 라벨 zip> <Validation 라벨 zip> configs
전제: configs/classes-100.json (select_classes.py 로 만든 기준 100개)이 이미 있다.
  - 100 = 기존 그대로, 200 = 100 + 새로 100개, 300 = 200 + 새로 100개(후보 풀에서 고정 시드로 순서대로 뽑음)
  - 50 = 100 에서 고정 시드로 무작위 50개
"""
import json
import os
import random
import sys

import select_classes as sc

train, val, out = sys.argv[1:4]
base = json.load(open(os.path.join(out, "classes-100.json"), encoding="utf-8"))
b100 = base["word_ids"]
lo, hi = base["range"]
pool = sc.candidates(sc.read_labels(train) + sc.read_labels(val), lo, hi)
rest = [w for w in pool if w not in set(b100)]
new = random.Random(1).sample(rest, 200)                 # 새 단어는 이 순서대로 100개씩 끊어 쓴다
sets = {50: sorted(random.Random(2).sample(b100, 50)),
        100: sorted(b100),
        200: sorted(b100 + new[:100]),
        300: sorted(b100 + new)}
for n, ids in sets.items():
    assert len(ids) == n and len(set(ids)) == n
    if n == 100:
        continue                                          # 기준 파일은 이미 실험에 쓰였으므로 다시 쓰지 않는다
    with open(os.path.join(out, "classes-%d.json" % n), "w", encoding="utf-8") as f:
        json.dump({"word_ids": ids, "n": n, "seed": "nested", "pool_size": len(pool), "range": [lo, hi]}, f)
assert set(sets[50]) <= set(sets[100]) <= set(sets[200]) <= set(sets[300])
print("후보 %d개, 포개짐 확인, 저장: %s" % (len(pool), sorted(sets)))
