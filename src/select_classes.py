"""클래스 부분집합 선정: 후보 단어 번호 중에서 고정 시드로 N개를 뽑는다.

사용: python src/select_classes.py <Training 라벨 zip> <Validation 라벨 zip> <개수> <시드> <출력 json> [최소번호=1501] [최대번호=3000]

후보 조건(정면 F 라벨 기준, 라벨 구간이 비어 있지 않은 항목만 셈):
  1. 단어 전부를 수행한 수행자 18명(번호 1~16 = Training, 17~18 = Validation) 모두에게 라벨이 있다.
  2. 수행자마다 이름 표기가 다르지 않다(번호 하나에 이름 하나).
  3. 그 이름이 다른 단어 번호에 쓰이지 않는다(이름이 같은 서로 다른 수어가 섞이지 않게).
영상을 받은 수행자가 달라도 같은 목록을 쓰도록 라벨만으로 정한다. 출력에는 번호만 담는다(단어 이름은 넣지 않는다).
"""
import collections
import json
import random
import re
import sys
import zipfile

NAME = re.compile(r"WORD(\d+)_REAL(\d+)_F_morpheme\.json$")
FULL_SIGNERS = set(range(1, 19))   # 전 단어를 수행한 수행자 번호(Training 1~16, Validation 17~18)


def read_labels(zip_path):
    """(단어 번호, 수행자, 이름) 목록. 빈 라벨은 제외."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            m = NAME.search(n)
            if not m:
                continue
            data = json.loads(z.read(n).decode("utf-8"))["data"]
            if data:
                out.append((int(m.group(1)), int(m.group(2)), data[0]["attributes"][0]["name"]))
    return out


def candidates(rows, lo, hi):
    per_word = collections.defaultdict(list)
    gloss_words = collections.defaultdict(set)
    for w, s, g in rows:
        per_word[w].append((s, g))
        gloss_words[g].add(w)
    pool = []
    for w, items in per_word.items():
        if not lo <= w <= hi:
            continue
        signers = {s for s, _ in items}
        glosses = {g for _, g in items}
        if FULL_SIGNERS <= signers and len(glosses) == 1 and len(gloss_words[next(iter(glosses))]) == 1:
            pool.append(w)
    return sorted(pool)


def select(rows, n, seed, lo=1501, hi=3000):
    pool = candidates(rows, lo, hi)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool)))), len(pool)


if __name__ == "__main__":
    train, val, n, seed, out = sys.argv[1:6]
    lo = int(sys.argv[6]) if len(sys.argv) > 6 else 1501
    hi = int(sys.argv[7]) if len(sys.argv) > 7 else 3000
    rows = read_labels(train) + read_labels(val)
    words, pool = select(rows, int(n), int(seed), lo, hi)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"word_ids": words, "n": len(words), "seed": int(seed), "pool_size": pool, "range": [lo, hi]}, f)
    print("후보 %d개 중 %d개 선정 -> %s" % (pool, len(words), out))
