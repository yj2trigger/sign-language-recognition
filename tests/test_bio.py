"""B/I/O 라벨러·문장 데이터(dataset_bio.py) 확인(합성 데이터, MediaPipe 없음).
실행: python -m pytest tests/test_bio.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset_bio as db

FPS = 15.0


def test_bio_basic():
    bio, typ = db.bio_labels([0.0, 0.1, 0.2, 0.3, 0.4], [(0.1, 0.3, "word")])
    assert bio.tolist() == [0, 1, 2, 2, 0]
    assert typ.tolist() == [0, 1, 1, 1, 0]


def test_bio_adjacent_segments_keep_their_own_begin():
    t = np.arange(10) / 10
    bio, typ = db.bio_labels(t, [(0.45, 0.6, "fingerspelling"), (0.1, 0.4, "word")])   # 순서가 섞여 들어와도 된다
    assert bio.tolist() == [0, 1, 2, 2, 2, 1, 2, 0, 0, 0]
    assert typ.tolist() == [0, 1, 1, 1, 1, 2, 2, 0, 0, 0]


def test_bio_segment_between_frames_gets_one_begin():
    bio, _ = db.bio_labels([0.0, 1 / 15, 2 / 15], [(0.07, 0.09, "word")])   # 프레임 사이에 낀 짧은 구간
    assert bio.tolist() == [0, 1, 0]


def test_bio_rejects_bad_segments():
    with pytest.raises(ValueError):
        db.bio_labels([0, 1], [(1.0, 0.5, "word")])
    with pytest.raises(ValueError):
        db.bio_labels([0, 1], [(0.0, 0.5, "gesture")])


def test_morpheme_segments_drop_names():
    lab = {"data": [{"start": 2.0, "end": 2.5, "attributes": [{"name": "X"}]},
                    {"start": 0.5, "end": 1.2, "attributes": [{"name": "Y"}]}]}
    segs = db.segments_from_morpheme(lab, is_fingerspelling=lambda it: it["attributes"][0]["name"] == "X")
    assert segs == [(0.5, 1.2, "word"), (2.0, 2.5, "fingerspelling")]
    assert db.segments_from_morpheme({"data": []}) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
