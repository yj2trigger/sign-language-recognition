"""클래스 부분집합 선정: 후보 단어 번호 중에서 고정 시드로 N개를 뽑는다.

사용: python src/select_classes.py <Training 라벨 zip> <Validation 라벨 zip> <개수> <시드> <출력 json> [최소번호=1501] [최대번호=3000]
                                  [--rule v1|v2] [--review 확인대상.json]
개수를 0 으로 주면 후보 전부를 쓴다.

후보 조건 v1(기본, 지난 실험의 classes-*.json 을 만든 기준. 정면 F 라벨, 라벨 구간이 비어 있지 않은 항목만 셈):
  1. 단어 전부를 수행한 수행자 18명(번호 1~16 = Training, 17~18 = Validation) 모두에게 라벨이 있다.
  2. 수행자마다 이름 표기가 다르지 않다(번호 하나에 이름 하나).
  3. 그 이름이 다른 단어 번호에 쓰이지 않는다(이름이 같은 서로 다른 수어가 섞이지 않게).
후보 조건 v2: 클래스는 단어 번호이고 이름은 표시에만 쓰므로, 표기 흔들림과 이름 공유로 단어를 버리지 않는다.
  1. v1 의 1과 같다.
  2. 앞뒤 공백을 지운 표기의 다수결 이름(대표 이름)이 그 번호 라벨의 절반을 넘는다.
  소수 표기는 normalize() 가 같은 단어로 볼지 정하고, 그러지 못한 클립은 확인 대상(--review)으로 남긴다.
  확인 대상은 좌표 추출 뒤 tools/check_mislabels.py 가 남길지 뺄지 판정한다.
영상을 받은 수행자가 달라도 같은 목록을 쓰도록 라벨만으로 정한다. 출력에는 번호만 담는다(단어 이름은 넣지 않는다).
"""
import argparse
import collections
import difflib
import json
import random
import re
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


def normalize(rows):
    """v2 표기 정리. 반환: (대표 이름 {번호: 이름}, 확인 대상 [(번호, 수행자, 표기)]).
    소수 표기를 같은 단어로 보는 경우(그 밖은 확인 대상):
      - 앞뒤 공백·줄바꿈만 다름
      - '대표 이름 + 숫자'와 정확히 같음. 데이터에서 이 숫자는 같은 이름을 가진 번호들 사이의 순번이다(예: 자다3 = 자다의 세 번째 번호).
        대표 이름이 다른 번호의 표기 뒤에 붙은 숫자(예: 대표 이름이 종합운동장인 번호의 '잠실1')는 지우지 않는다.
      - 오타·줄임: 대표 이름과 유사도 0.6 이상이고 다른 번호의 대표 이름이 아님"""
    per = collections.defaultdict(collections.Counter)
    for w, _, g in rows:
        per[w][g.strip()] += 1
    major = {w: c.most_common(1)[0][0] for w, c in per.items()}
    others = collections.Counter(major.values())
    review = []
    for w, s, g in rows:
        g, m = g.strip(), major[w]
        if g == m or re.fullmatch(re.escape(m) + r"\d+", g) and not m[-1].isdigit():
            continue
        is_other = others[g] > 0 or others[re.sub(r"\d+$", "", g)] > 0
        if not is_other and difflib.SequenceMatcher(None, g, m).ratio() >= 0.6:
            continue
        review.append((w, s, g))
    return major, review


def candidates_v2(rows, lo, hi):
    """v2 후보 번호와, 후보 번호 안의 확인 대상 [(번호, 수행자, 표기가 가리키는 다른 번호들)]."""
    major, review = normalize(rows)
    per_word = collections.defaultdict(list)
    for w, s, g in rows:
        per_word[w].append((s, g.strip()))
    pool = sorted(w for w, items in per_word.items() if lo <= w <= hi and FULL_SIGNERS <= {s for s, _ in items}
                  and 2 * sum(g == major[w] for _, g in items) > len(items))
    by_name = collections.defaultdict(list)
    for w, m in major.items():
        by_name[m].append(w)
    keep = set(pool)
    points = [(w, s, sorted(set(by_name[g] + by_name[re.sub(r"\d+$", "", g)]) - {w})) for w, s, g in review if w in keep]
    return pool, points


def select(rows, n, seed, lo=1501, hi=3000, rule="v1"):
    pool = candidates(rows, lo, hi) if rule == "v1" else candidates_v2(rows, lo, hi)[0]
    return sorted(random.Random(seed).sample(pool, min(n, len(pool)) if n else len(pool))), len(pool)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("train", "val", "n", "seed", "out"):
        ap.add_argument(name)
    ap.add_argument("lo", nargs="?", type=int, default=1501)
    ap.add_argument("hi", nargs="?", type=int, default=3000)
    ap.add_argument("--rule", default="v1", choices=["v1", "v2"])
    ap.add_argument("--review", default="", help="v2 확인 대상(번호·수행자·가리키는 번호)을 저장할 json")
    a = ap.parse_args()
    rows = read_labels(a.train) + read_labels(a.val)
    words, pool = select(rows, int(a.n), int(a.seed), a.lo, a.hi, a.rule)
    info = {"word_ids": words, "n": len(words), "seed": int(a.seed), "pool_size": pool, "range": [a.lo, a.hi]}
    if a.rule == "v2":
        info["rule"] = "v2"
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(info, f)
    if a.review:
        keep = set(words)
        items = [{"word": w, "signer": s, "points_to": p} for w, s, p in candidates_v2(rows, a.lo, a.hi)[1] if w in keep]
        with open(a.review, "w", encoding="utf-8") as f:
            json.dump({"rule": "v2", "range": [a.lo, a.hi], "items": items}, f)
        print("확인 대상 %d개 -> %s" % (len(items), a.review))
    print("후보 %d개 중 %d개 선정 -> %s" % (pool, len(words), a.out))
