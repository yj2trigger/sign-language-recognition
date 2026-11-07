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


def test_v2_normalize_and_review():
    rows = (rows_for(1501, "하남시청")[:-1] + [(1501, 18, "하남시청\n")]          # 줄바꿈만 다름 -> 통과
            + rows_for(1502, "자다")[:-2] + [(1502, 17, "자다3"), (1502, 18, "자다3")]   # 대표 이름 + 순번 -> 통과
            + rows_for(1503, "잠실")
            + rows_for(1504, "종합운동장")[:-1] + [(1504, 18, "잠실1")]           # 다른 번호의 이름 + 숫자 -> 확인 대상
            + rows_for(1505, "창피하다")
            + rows_for(1506, "동구주민지원센터")[:-1] + [(1506, 18, "창피하다")]   # 다른 번호의 이름 -> 확인 대상
            + rows_for(1507, "완주군수어통역센터")[:-1] + [(1507, 18, "완완주군수어통역센터")]   # 오타 -> 통과
            + rows_for(1508, "막히다") + rows_for(1509, "막히다")                   # 이름 공유 -> 둘 다 후보
            + rows_for(1510, "엄마")[:-1]                                           # 수행자 누락 -> 제외
            + rows_for(1511, "X")[:9] + [(1511, s, "Y") for s in range(10, 19)])  # 다수 이름이 절반 이하 -> 제외
    pool, points = sc.candidates_v2(rows, 1501, 3000)
    assert pool == list(range(1501, 1510))
    assert points == [(1504, 18, [1503]), (1506, 18, [1505])]


def test_v1_unchanged_on_real_labels():
    """실제 라벨이 있으면 v1 후보 수가 지난 실험 기록(1,176개)과 같은지 본다. 라벨이 없으면 건너뛴다."""
    tr = "D:/_대학교/연구과제/데이터셋/수어영상/수어 영상/1.Training/[라벨]01_real_word_morpheme.zip"
    va = "D:/sign-language-data/downloads/test-39478/004.수어영상/2.Validation/라벨링데이터/REAL/WORD/01_real_word_morpheme.zip"
    if not (Path(tr).exists() and Path(va).exists()):
        print("건너뜀: 실제 라벨 없음")
        return
    rows = sc.read_labels(tr) + sc.read_labels(va)
    assert len(sc.candidates(rows, 1501, 3000)) == 1176


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
