"""클래스 선정 조건 확인용 테스트(합성 데이터). 실행: .venv\\Scripts\\python tests\\test_select_classes.py
코드가 조건대로 거르는지만 확인한다. 실제 단어에 대한 결론이 아니다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import select_classes as sc


def rows_for(word, gloss, signers=range(1, 19)):
    return [(word, s, gloss) for s in signers]


def test_candidate_filters():
    rows = (rows_for(1501, "A")                                  # 통과
            + rows_for(1502, "B")[:-1]                           # 수행자 18번 누락 -> 제외
            + [(1503, s, "C" if s < 10 else "C2") for s in range(1, 19)]   # 표기 변형 -> 제외
            + rows_for(1504, "D") + rows_for(1505, "D")          # 이름 공유(서로 다른 번호) -> 둘 다 제외
            + rows_for(1500, "E")                                # 범위 밖 -> 제외
            + rows_for(1506, "F", signers=list(range(1, 18)) + [25]))   # 25번이 끼어도 18번이 없으면 제외
    assert sc.candidates(rows, 1501, 3000) == [1501]


def test_select_is_reproducible_and_sorted():
    rows = sum((rows_for(w, "g%d" % w) for w in range(1501, 1601)), [])
    a, pool = sc.select(rows, 10, seed=0)
    b, _ = sc.select(rows, 10, seed=0)
    c, _ = sc.select(rows, 10, seed=1)
    assert pool == 100 and a == b and a == sorted(a) and len(set(a)) == 10 and a != c


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
