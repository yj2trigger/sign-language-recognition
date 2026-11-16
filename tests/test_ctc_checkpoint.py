"""Atomic persistence, RNG and exact epoch-boundary restart contracts."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_checkpoint import load_checkpoint, save_checkpoint
from test_ctc_training import batch, trainer


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_exact_resume_and_best_checkpoint(tmp_path, device):
    continuous = trainer(dropout=0.2, device=device)
    continuous.fit(lambda: [batch()], lambda: [batch()], epochs=2, checkpoint_dir=tmp_path / "full")
    expected = {key: value.clone() for key, value in continuous.model.state_dict().items()}
    history = continuous.history
    split = trainer(dropout=0.2, device=device)
    split.fit(lambda: [batch()], lambda: [batch()], epochs=1, checkpoint_dir=tmp_path / "split")
    draws = (random.random(), np.random.random(), torch.rand(2))
    cuda_draw = torch.rand(2, device="cuda") if device == "cuda" else None
    resumed = trainer(dropout=0.2, device=device)
    resumed.resume(tmp_path / "split" / "last.pt")
    assert random.random() == draws[0]
    assert np.random.random() == draws[1]
    assert torch.equal(torch.rand(2), draws[2])
    if cuda_draw is not None:
        assert torch.equal(torch.rand(2, device="cuda"), cuda_draw)
    # Rewind after checking the RNG draws.
    resumed.resume(tmp_path / "split" / "last.pt")
    resumed.fit(lambda: [batch()], lambda: [batch()], epochs=1, checkpoint_dir=tmp_path / "split")
    assert resumed.history == history and resumed.global_step == 2
    assert all(torch.equal(expected[k], v) for k, v in resumed.model.state_dict().items())
    expected_optimizer = continuous.optimizer.state_dict()
    actual_optimizer = resumed.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, state in expected_optimizer["state"].items():
        for name, value in state.items():
            actual = actual_optimizer["state"][parameter_id][name]
            assert torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
    best = load_checkpoint(tmp_path / "split" / "best.pt")
    assert best["best_loss"] == min(row["validation_loss"] for row in history)


def test_atomic_save_failure_preserves_previous(tmp_path, monkeypatch):
    instance = trainer()
    path = tmp_path / "last.pt"
    save_checkpoint(path, instance)
    previous = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_checkpoint(path, instance)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_metadata_mismatch(tmp_path):
    instance = trainer()
    path = tmp_path / "last.pt"
    save_checkpoint(path, instance)
    instance.vocabulary = {"synthetic": 1}
    with pytest.raises(ValueError):
        instance.resume(path)


def test_wrong_checkpoint_format(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"state": {}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_best_preserved_in_new_directory_without_improvement(tmp_path, monkeypatch):
    first = trainer()
    first.fit(lambda: [batch()], lambda: [batch()], epochs=1, checkpoint_dir=tmp_path / "old")
    expected = load_checkpoint(tmp_path / "old" / "best.pt")
    second = trainer()
    second.resume(tmp_path / "old" / "last.pt")
    monkeypatch.setattr(second, "run_epoch", lambda *args, **kwargs: first.best_loss + 1)
    second.fit(lambda: [batch()], lambda: [batch()], epochs=1, checkpoint_dir=tmp_path / "new")
    actual = load_checkpoint(tmp_path / "new" / "best.pt")
    assert actual["best_loss"] == expected["best_loss"]
    assert actual["epoch"] == expected["epoch"]
    assert all(torch.equal(actual["model_state"][key], value)
               for key, value in expected["model_state"].items())


def test_interrupted_best_write_recovers_from_last(tmp_path, monkeypatch):
    from fingerspelling import ctc_checkpoint
    instance = trainer()
    original = ctc_checkpoint.write_checkpoint
    def fail_best(path, payload):
        if Path(path).name == "best.pt":
            raise OSError("simulated best write failure")
        original(path, payload)
    monkeypatch.setattr(ctc_checkpoint, "write_checkpoint", fail_best)
    with pytest.raises(OSError):
        instance.fit(lambda: [batch()], lambda: [batch()], epochs=1, checkpoint_dir=tmp_path)
    assert (tmp_path / "last.pt").exists()
    monkeypatch.setattr(ctc_checkpoint, "write_checkpoint", original)
    resumed = trainer()
    resumed.resume(tmp_path / "last.pt")
    monkeypatch.setattr(resumed, "run_epoch", lambda *args, **kwargs: instance.best_loss + 1)
    resumed.fit(lambda: [batch()], lambda: [batch()], epochs=1, checkpoint_dir=tmp_path)
    assert load_checkpoint(tmp_path / "best.pt")["epoch"] == 1


@pytest.mark.parametrize("corruption", ["shape", "nan", "negative_variance", "step"])
def test_corrupt_optimizer_rejected_without_mutating_trainer(tmp_path, corruption):
    from fingerspelling.ctc_checkpoint import capture_rng
    source = trainer()
    source.run_epoch([batch()], training=True)
    path = tmp_path / "last.pt"
    save_checkpoint(path, source)
    payload = load_checkpoint(path)
    state = next(iter(payload["optimizer_state"]["state"].values()))
    if corruption == "shape":
        state["exp_avg"] = torch.zeros(1)
    elif corruption == "nan":
        state["exp_avg"].fill_(float("nan"))
    elif corruption == "negative_variance":
        state["exp_avg_sq"].fill_(-1)
    else:
        state["step"] = torch.tensor(-1.)
    torch.save(payload, path)
    destination = trainer()
    original_model = destination.model
    original_optimizer = destination.optimizer
    original_rng = capture_rng()
    with pytest.raises(ValueError):
        destination.resume(path)
    assert destination.model is original_model
    assert destination.optimizer is original_optimizer
    assert destination.global_step == 0
    assert torch.equal(torch.get_rng_state(), original_rng["torch"])
    destination.run_epoch([batch()], training=True)
