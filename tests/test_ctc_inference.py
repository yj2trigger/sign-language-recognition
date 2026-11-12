"""Batch inference and checkpoint identity checks."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_checkpoint import save_checkpoint
from fingerspelling.ctc_inference import CTCRecognizer
from test_ctc_training import batch, trainer


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_inference_roundtrip_and_immutability(tmp_path, device):
    instance = trainer(device=device)
    instance.run_epoch([batch()], training=True)
    path = tmp_path / "last.pt"
    save_checkpoint(path, instance)
    recognizer = CTCRecognizer(path, device=device)
    values = batch()
    before = values["features"].clone()
    scores, lengths = recognizer.predict(values["features"], values["input_lengths"])
    instance.model.eval()
    expected, _ = instance.model(values["features"].to(device), values["input_lengths"])
    assert torch.allclose(scores, expected.cpu())
    cpu_scores, _ = CTCRecognizer(path, device="cpu").predict(values["features"], values["input_lengths"])
    assert torch.allclose(scores, cpu_scores, atol=1e-5)
    assert scores.shape == (2, 6, 3) and lengths.tolist() == [6, 4]
    assert not scores.requires_grad and scores.device.type == "cpu"
    assert torch.equal(values["features"], before)
    with pytest.raises(ValueError):
        recognizer.predict(torch.zeros(1, 3, 4), [3])
    with pytest.raises(ValueError):
        CTCRecognizer(path, expected_vocabulary={"different": 1})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_inference_restores_checkpoint_precision_and_caller_flags(tmp_path):
    instance = trainer(device="cuda")
    instance.run_epoch([batch()], training=True)
    path = tmp_path / "precision.pt"
    save_checkpoint(path, instance)
    recognizer = CTCRecognizer(path, device="cuda")
    values = batch()
    expected, _ = recognizer.predict(values["features"], values["input_lengths"])
    original_cudnn = torch.backends.cudnn.allow_tf32
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        actual, _ = recognizer.predict(values["features"], values["input_lengths"])
        assert torch.equal(actual, expected)
        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32
        with pytest.raises(ValueError):
            recognizer.predict(torch.zeros(1, 3, 4), [3])
        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32 = original_cudnn
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
