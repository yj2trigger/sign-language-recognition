import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import segment_post as sp


def test_parse_and_default_is_noop():
    c = sp.PostConfig.parse("frag_s=0.3,gap_s=0.1")
    assert c.frag_s == 0.3 and c.gap_s == 0.1 and c.dedup_gap_s == 0.0
    segs = [(0.0, 0.1, "word"), (0.12, 0.5, "word")]
    d = sp.PostConfig()
    assert sp.merge_fragments(segs, d) == segs and sp.dedup([(0, 1, "word", 5), (1.1, 2, "word", 5)], d) == [(0, 1, "word", 5), (1.1, 2, "word", 5)]


def test_merge_fragments():
    c = sp.PostConfig(frag_s=0.3, gap_s=0.1)
    segs = [(0.0, 0.5, "word"), (0.55, 0.7, "word"), (1.5, 1.6, "word"), (1.65, 2.2, "fingerspelling")]
    # 0.15초 조각은 앞에 붙고, 멀리 떨어진 0.1초 조각은 뒤의 긴 조각과 붙는다(종류가 다르면 word)
    assert sp.merge_fragments(segs, c) == [(0.0, 0.7, "word"), (1.5, 2.2, "word")]
    assert sp.merge_fragments([(0, 0.5, "word"), (0.55, 1.2, "word")], c) == [(0, 0.5, "word"), (0.55, 1.2, "word")]   # 둘 다 길면 안 붙임


def test_group_fingerspelling():
    t = np.arange(0, 4, 1 / 15)
    mot = np.zeros((len(t), 2))
    mot[(t >= 1) & (t < 3), 0] = 1.0                       # 1~3초: 손가락만 많이 움직임(지문자)
    mot[:, 1] = 0.1
    segs = [(1.0, 1.4, "word"), (1.5, 1.9, "fingerspelling"), (2.0, 2.9, "word"), (3.5, 3.8, "fingerspelling")]
    out = sp.group_fingerspelling(segs, t, mot, sp.PostConfig(fs_gap_s=0.2, fs_min_s=0.6, fs_ratio=2.0))
    assert out == [(1.0, 2.9, "fingerspelling"), (3.5, 3.8, "word")]                 # 짧은 외톨이는 지문자 아님


def test_dedup():
    c = sp.PostConfig(dedup_gap_s=0.3)
    items = [(0, 0.5, "word", 7), (0.6, 1.0, "word", 7), (1.5, 2.0, "word", 7), (2.1, 2.5, "word", 8)]
    assert sp.dedup(items, c) == [(0, 1.0, "word", 7), (1.5, 2.0, "word", 7), (2.1, 2.5, "word", 8)]
