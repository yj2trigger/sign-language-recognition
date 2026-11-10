"""숫자 보조 라벨·손실 확인(합성 값). 실행: .venv\\Scripts\\python tests\\test_number_aux.py"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import number_labels as nl
import train


def test_labels():
    cases = {"이만오천": [0, 0, 0, 2, 1, 1], "칠천만": [7, 0, 0, 0, 1, 0], "백칠십칠": [0, 1, 7, 7, 0, 0],
             "십팔": [0, 0, 1, 8, 0, 0], "팔십": [0, 0, 8, 0, 0, 0], "이억": [0, 0, 0, 2, 2, 0],
             "만": [0, 0, 0, 1, 1, 0], "삼천오백": [3, 5, 0, 0, 0, 0], "영": [0, 0, 0, 0, 0, 0]}
    for name, want in cases.items():
        assert nl.label(name) == want, (name, nl.label(name))
    assert nl.label("사과") is None and nl.label("") is None


def test_aux_loss():
    torch.manual_seed(0)
    logits = torch.randn(3, train.AUX_DIM)
    tgt = torch.tensor([[1, 2, 3, 4, 0, 1], [-1] * 6, [0, 0, 0, 5, 2, 0]])
    full = train.aux_loss(logits, tgt)
    only = train.aux_loss(logits[[0, 2]], tgt[[0, 2]])
    assert torch.isclose(full, only), (full, only)           # 숫자 아닌 표본(-1)은 손실에 안 들어간다
    assert train.aux_loss(logits[[1]], tgt[[1]]).item() == 0  # 숫자 표본이 없는 배치는 0(nan 아님)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과:", name)
