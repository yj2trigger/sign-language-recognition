import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_video_words import edit_distance


def test_edit_distance():
    assert edit_distance([], []) == 0
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [1, 3]) == 1          # 삭제
    assert edit_distance([1, 3], [1, 2, 3]) == 1          # 삽입
    assert edit_distance([1, 9, 3], [1, 2, 3]) == 1       # 교체
    assert edit_distance(["FS", None], [-1, 5]) == 2      # 지문자·실패 표시는 어떤 정답과도 다름
    assert edit_distance([1, 2], [3, 4, 5]) == 3
