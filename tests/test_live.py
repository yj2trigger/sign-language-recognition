"""실시간 채점(LiveScorer)과 실시간 후처리(PostStream) 확인(합성, MediaPipe·모델 불필요)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import realtime as rt
import segment_post as sp


def test_word_session_scoring():
    sc = rt.LiveScorer("A1", [5, 6, 7], sentence=False, names={5: "가", 6: "나", 7: "다"})
    assert "다음: 가" in sc.start_line()
    m = sc.on_token(5)
    assert "[맞음]" in m and "다음: 나" in m
    assert "[틀림] 정답: 나" in sc.on_token(5)                                 # 6 이 정답인데 5
    sc.on_token(7)
    assert "대본이 끝난" in sc.on_token(9)
    assert sc.hit == 2 and sc.i == 3 and "2/3" in sc.final_line()


def test_sentence_session_scoring():
    sc = rt.LiveScorer("S1", [5, 6], sentence=True)
    for w in (5, 9, 6):
        assert sc.on_token(w) is None                      # 문장 세션은 반복이 끝나야 채점
    msg = sc.on_sentence_end()
    assert "맞은 단어 2/2" in msg and "오류율 0.50" in msg   # 9 하나가 끼어듦(삽입 1 / 정답 2)
    sc.on_token(6)
    assert "반복 2번, 맞은 단어 3/4" in sc.final_line()        # 문장 끝 없이 멈춰도 마지막 반복을 셈
    assert sc.summary()["reps"][1] == {"pred": [6], "hit": 1, "ed": 1}


def _buf(n=40):
    xy = np.zeros((53, 2), np.float32)
    return [(i / 15, xy, np.zeros(53, bool)) for i in range(n)]


def test_post_stream_holds_and_joins():
    st = rt.PostStream(sp.PostConfig(frag_s=0.3, gap_s=0.2))
    buf = _buf()
    assert st.push((0.0, 0.5, "word"), buf) == []
    assert st.due(0.6, buf) == []                          # 아직 간격 0.2초 안
    assert st.due(0.6, buf, open_start=0.62) == []         # 다음 조각이 이미 시작됨 → 기다림
    assert st.push((0.62, 0.72, "word"), buf) == []        # 0.1초 조각, 간격 0.12 → 같은 묶음
    assert st.due(1.0, buf) == [(0.0, 0.72, "word")]       # 간격이 벌어지자 합쳐서 내보냄
    assert st.flush(buf) == []


def test_post_stream_caps_continuous_runs():
    st = rt.PostStream(sp.PostConfig(frag_s=0.3, gap_s=0.2, fs_gap_s=0.5), max_run_s=2.0)
    buf = _buf(60)
    out = []
    for i in range(8):                                     # 0.3초 간격으로 쉬지 않고 이어지는 0.25초 구간들
        out += st.push((i * 0.3, i * 0.3 + 0.25, "word"), buf)
    assert out and out[-1][1] <= 2.0 + 1e-9                 # 2초를 넘기 전에 한 번 내보냄(무한정 모으지 않는다)
    assert st.due(10.0, buf, open_start=None) != []


def test_post_stream_off_is_passthrough():
    st = rt.PostStream(sp.PostConfig())                    # 모두 꺼짐: hold 0 → 다음 박자에 그대로 나감
    buf = _buf()
    st.push((0.0, 0.5, "word"), buf)
    assert st.due(0.6, buf) == [(0.0, 0.5, "word")]
