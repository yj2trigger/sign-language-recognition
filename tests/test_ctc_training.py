"""Independent CTC path oracle and synthetic optimization checks."""
import itertools
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_model import CTCModelConfig, FingerspellingCTC
from fingerspelling.ctc_training import CTCTrainer, CTCTrainingConfig, ctc_loss, seed_everything


def batch():
    return dict(features=torch.ones(2, 6, 2), input_lengths=torch.tensor([6, 4]),
                targets=torch.tensor([1, 1], dtype=torch.int64), target_lengths=torch.tensor([1, 1]))


def trainer(dropout=0, device="cpu"):
    seed_everything(7)
    model = FingerspellingCTC(CTCModelConfig(2, 3, 0, hidden_size=8, dropout=dropout))
    return CTCTrainer(model, CTCTrainingConfig(learning_rate=0.02, weight_decay=0, seed=7), device=device)


@pytest.mark.parametrize("target", [[], [1], [1, 1], [1, 2]])
def test_loss_against_enumerated_paths(target):
    logits = torch.tensor([[[0.1, 0.8, -0.4], [0.6, -0.3, 0.7], [0.0, 0.4, 0.9]]],
                          requires_grad=True)
    probabilities = logits.detach().softmax(-1)[0]
    total = 0.0
    for path in itertools.product(range(3), repeat=3):
        collapsed = [key for key, _ in itertools.groupby(path) if key != 0]
        if collapsed == target:
            total += float(torch.stack([probabilities[t, key] for t, key in enumerate(path)]).prod())
    actual = ctc_loss(logits, [3], torch.tensor(target, dtype=torch.int64), [len(target)], blank_id=0)
    expected = -torch.tensor(total).log() / max(len(target), 1)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("targets,lengths,outputs,blank", [
    ([1, 1], [2], [2], 0), ([0], [1], [3], 0), ([3], [1], [3], 0),
    ([-1], [1], [3], 0), ([1], [2], [3], 0), ([1], [1], [4], 0),
    ([1], [-1], [3], 0), ([1], [1], [3], 3)])
def test_loss_rejects_invalid_contract(targets, lengths, outputs, blank):
    with pytest.raises(ValueError):
        ctc_loss(torch.zeros(1, 3, 3), outputs, torch.tensor(targets), lengths, blank_id=blank)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_loss_rejects_nonfinite(value):
    with pytest.raises(ValueError):
        ctc_loss(torch.full((1, 3, 3), value), [3], torch.tensor([1]), [1], blank_id=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_computed_loss_raises(monkeypatch, value):
    monkeypatch.setattr(torch.nn.CTCLoss, "forward", lambda *args: torch.tensor([value]))
    with pytest.raises(FloatingPointError):
        ctc_loss(torch.zeros(1, 3, 3), [3], torch.tensor([1]), [1], blank_id=0)


def test_nonfinite_reduction_raises(monkeypatch):
    monkeypatch.setattr(torch.nn.CTCLoss, "forward", lambda *args: torch.full((4,), 3e38))
    with pytest.raises(FloatingPointError):
        ctc_loss(torch.zeros(4, 3, 3), [3] * 4, torch.tensor([1] * 4), [1] * 4, blank_id=0)


def test_nonzero_blank_and_unpadded_loss():
    logits = torch.randn(1, 5, 3)
    actual = ctc_loss(logits, [3], torch.tensor([0, 0]), [2], blank_id=2)
    expected = torch.nn.CTCLoss(blank=2)(logits[:, :3].log_softmax(-1).transpose(0, 1),
                                        torch.tensor([0, 0]), torch.tensor([3]), torch.tensor([2]))
    assert torch.allclose(actual, expected)


def test_training_clipping_validation_and_loss_decrease(monkeypatch):
    instance = trainer()
    initial = instance.run_epoch([batch()], training=False)
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    def clip(parameters, limit, **kwargs):
        parameters = list(parameters)
        result = original(parameters, limit, **kwargs)
        norm = torch.stack([p.grad.norm() for p in parameters if p.grad is not None]).norm()
        assert norm <= limit + 1e-5
        calls.append(result)
        return result
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    for _ in range(30):
        instance.run_epoch([batch()], training=True)
    final = instance.run_epoch([batch()], training=False)
    assert final < initial
    assert len(calls) == 30 and instance.global_step == 30
    before = [p.clone() for p in instance.model.parameters()]
    instance.run_epoch([batch()], training=False)
    assert all(torch.equal(a, b) for a, b in zip(before, instance.model.parameters()))


def test_seed_reproducibility():
    first = trainer()
    first.run_epoch([batch()], training=True)
    second = trainer()
    second.run_epoch([batch()], training=True)
    assert all(torch.equal(a, b) for a, b in zip(first.model.parameters(), second.model.parameters()))


def test_empty_batches():
    with pytest.raises(ValueError):
        trainer().run_epoch([], training=True)


def test_padding_contract():
    values = batch()
    values["padding_mask"] = torch.ones(2, 6, dtype=torch.bool)
    with pytest.raises(ValueError):
        trainer().run_epoch([values], training=False)


def test_mixed_boolean_targets():
    with pytest.raises(ValueError):
        ctc_loss(torch.zeros(1, 3, 3), [3], [True, 2], [2], blank_id=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training():
    instance = trainer(device="cuda")
    assert instance.run_epoch([batch()], training=True) >= 0
