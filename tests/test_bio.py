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


def test_concat_sentence_shifts_labels():
    def clip(n, s, e):
        t = np.arange(n) / FPS
        return {"t": t, "xy": np.zeros((n, 53, 2), np.float32), "mask": np.zeros((n, 53), bool),
                "face": np.zeros((n, 33, 2), np.float32), "face_mask": np.zeros((n, 33), bool),
                "seg_start": s, "seg_end": e, "signer": 7}
    s = db.concat_sentence([clip(45, 1.0, 2.0), clip(45, 1.0, 2.0)], np.random.default_rng(0),
                           lead=(0.4, 0.4), gap=(0.3, 0.3), tail=(0.5, 0.5), speed=(1.0, 1.0))
    assert np.allclose(np.diff(s["t"]), 1 / FPS) and s["signer"] == 7
    (a0, a1, _), (b0, b1, _) = s["segments"]
    assert abs((a1 - a0) - 1.0) < 1e-6 and abs((b1 - b0) - 1.0) < 1e-6
    assert abs(a0 - 0.4) < 0.07                          # 첫 단어 앞 여유
    assert abs((b0 - a1) - 0.3) < 0.07                   # 단어 사이 간격
    assert abs((s["t"][-1] - b1) - 0.5) < 0.07           # 마지막 단어 뒤 여유
    fast = db.concat_sentence([clip(45, 1.0, 2.0)] * 2, np.random.default_rng(0), speed=(2.0, 2.0))
    assert all(abs((e - s_) - 0.5) <= 1 / FPS / 2 + 1e-6 for s_, e, _ in fast["segments"])   # 2배속이면 단어 길이가 절반


def test_concat_bridges_words_without_teleport():
    def clip(x):
        n = 45
        xy = np.zeros((n, 53, 2), np.float32)
        xy[:, 32:] = x                                   # 오른손이 단어 내내 x 에 있다
        mask = np.zeros((n, 53), bool)
        mask[:, 32:] = True
        mask[:15, 32:] = mask[31:, 32:] = False          # 단어(1~2초) 앞뒤로는 손을 내려 안 보인다
        return {"t": np.arange(n) / FPS, "xy": xy, "mask": mask, "face": np.zeros((n, 33, 2), np.float32),
                "face_mask": np.zeros((n, 33), bool), "seg_start": 1.0, "seg_end": 2.0, "signer": 1}
    s = db.concat_sentence([clip(0.2), clip(0.8)], np.random.default_rng(0), gap=(0.4, 0.4), speed=(1.0, 1.0))
    (_, a1, _), (b0, _, _) = s["segments"]
    i, j = round(a1 * FPS), round(b0 * FPS)
    x = s["xy"][i:j + 1, 32, 0]
    assert s["mask"][i:j + 1, 32].all()                  # 단어 사이에 손을 내리지 않는다
    assert np.all(np.diff(x) > 0.05) and np.all(np.diff(x) < 0.15), x   # 0.2 에서 0.8 로 고르게 이동(순간 이동 없음)


def test_output_dir_inside_repo_is_refused():
    assert db._inside_repo(db.REPO + "/runs/x")
    assert not db._inside_repo(str(Path(db.REPO).parent / "elsewhere"))


def test_other_drive_is_outside_repo():
    other = "D:" if not db.REPO.upper().startswith("D:") else "E:"
    assert not db._inside_repo(other + r"\sign-language-data\bio")   # 예전에는 commonpath 가 ValueError 로 죽었다


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
