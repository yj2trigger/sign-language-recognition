"""숫자 단어의 자릿값 보조 라벨을 만든다(T-040 설계, 이슈 #3).

사용: python tools/number_labels.py <단어 목록 csv(번호,단어,...)> <클래스 json> <출력 json>
이름이 한자어 수사 글자로만 된 단어를 숫자로 보고(이름 규칙이라 대략), 만 단위로 끊어 읽은 가장 큰 묶음의
[천, 백, 십, 일 자리 숫자, 단위(0 없음·1 만·2 억), 아래 묶음 유무(0/1)] 를 단어 번호별로 적는다.
예: 이만오천 -> [0,0,0,2,1,1], 칠천만 -> [7,0,0,0,1,0], 백칠십칠 -> [0,1,7,7,0,0].
출력에는 단어 이름을 넣지 않지만 원본에서 나온 파생물이므로 레포 밖(D:\\sign-language-data)에 둔다.
"""
import csv
import json
import sys

DIGIT = dict(zip("영일이삼사오육칠팔구", range(10)))     # 영 = 0 (2631번, 일~십 2632~2642번 바로 앞)
SMALL = {"십": 10, "백": 100, "천": 1000}
BIG = {"만": 10 ** 4, "억": 10 ** 8}
CHARS = set(DIGIT) | set(SMALL) | set(BIG)


def value(name):
    total = section = d = 0
    for ch in name:
        if ch in DIGIT:
            d = DIGIT[ch]
        elif ch in SMALL:
            section, d = section + (d or 1) * SMALL[ch], 0
        else:
            total, section, d = total + ((section + d) or 1) * BIG[ch], 0, 0
    return total + section + d


def label(name):
    """숫자 이름이면 6개 라벨, 아니면 None."""
    if not name or set(name) - CHARS:
        return None
    v = value(name)
    unit = 2 if v >= 10 ** 8 else 1 if v >= 10 ** 4 else 0
    top, low = divmod(v, 10 ** (4 * unit))
    if top >= 10 ** 4:                     # 조 단위 이상은 다루지 않는다
        return None
    return [top // 1000, top // 100 % 10, top // 10 % 10, top % 10, unit, int(low > 0)]


def main():
    words, classes, out = sys.argv[1:4]
    with open(classes, encoding="utf-8") as f:
        ids = set(json.load(f)["word_ids"])
    labels = {}
    with open(words, encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if r and r[0].isdigit() and int(r[0]) in ids and (lab := label(r[1].strip())):
                labels[r[0]] = lab
    combos = len({tuple(v) for v in labels.values()})
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rule": "T-040 자릿값 [천,백,십,일,단위,아래묶음]", "labels": labels}, f)
    print("클래스 %d개 중 숫자 %d개, 라벨 조합 %d가지 -> %s" % (len(ids), len(labels), combos, out))


if __name__ == "__main__":
    main()
