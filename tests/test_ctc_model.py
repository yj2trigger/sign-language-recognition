"""Synthetic model contracts. Requires an existing PyTorch installation."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_model import CTCModelConfig, FingerspellingCTC


def model(width=106):
    return FingerspellingCTC(CTCModelConfig(width, 5, 3, hidden_size=8, dropout=0))


def test_variable_batch_padding_and_immutability():
    network = model().eval()
    values = torch.randn(2, 9, 106, requires_grad=True)
    before = values.detach().clone()
    logits, lengths = network(values, [9, 4])
    changed = values.detach().clone()
    changed[1, 4:] = 900
    other, _ = network(changed, [9, 4])
    single, _ = network(values[1:2, :4], [4])
    assert logits.shape == (2, 9, 5)
    assert lengths.tolist() == [9, 4]
    assert torch.equal(logits[1, 4:], torch.zeros(5, 5))
    assert torch.allclose(logits, other)
    assert torch.allclose(logits[1, :4], single[0], atol=1e-6)
    assert torch.equal(values, before)
    logits.sum().backward()
    assert network.rnn.weight_ih_l0.grad.abs().sum() > 0
    assert torch.count_nonzero(values.grad[1, 4:]) == 0


@pytest.mark.parametrize("lengths", [[0], [-1], [10], [2.5], [True], [1, 2]])
def test_bad_lengths(lengths):
    with pytest.raises(ValueError):
        model()(torch.zeros(1, 9, 106), lengths)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_even_in_padding(value):
    values = torch.zeros(1, 3, 106)
    values[0, 2, 0] = value
    with pytest.raises(ValueError):
        model()(values, [1])


@pytest.mark.parametrize("values", [torch.zeros(1, 4, 105), torch.zeros(4, 106),
                                   torch.zeros(0, 4, 106), torch.zeros(1, 0, 106),
                                   torch.zeros(1, 4, 106, dtype=torch.float64)])
def test_bad_input(values):
    with pytest.raises(ValueError):
        model()(values, [4])


@pytest.mark.parametrize("options", [{"blank_id": 5}, {"blank_id": True},
                                    {"num_classes": 1}, {"input_size": 0},
                                    {"dropout": float("nan")}, {"num_layers": 0}])
def test_config(options):
    config = dict(input_size=106, num_classes=5, blank_id=3)
    config.update(options)
    with pytest.raises(ValueError):
        CTCModelConfig(**config)


def test_long_sequence_and_length_transform():
    network = model(2).eval()
    with torch.no_grad():
        logits, lengths = network(torch.zeros(1, 10000, 2), [10000])
    assert logits.shape == (1, 10000, 5)
    assert network.output_lengths(lengths).tolist() == [10000]


def test_mixed_boolean_lengths():
    with pytest.raises(ValueError):
        model()(torch.zeros(2, 3, 106), [True, 2])
    with pytest.raises(ValueError):
        model().output_lengths([True, 2])
