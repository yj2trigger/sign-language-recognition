"""Strict CTC objective and resumable, epoch-boundary training."""
from dataclasses import dataclass
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ctc_model import integer, lengths_tensor


def seed_everything(seed):
    integer(seed, "seed")
    if seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit uint32")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def ctc_loss(logits, output_lengths, targets, target_lengths, *, blank_id):
    """Mean per-example CTC loss, normalized by max(target length, 1).

    Targets are packed int64 IDs, with no blank. Empty targets are supported.
    Invalid alignments raise before CTCLoss and zero_infinity remains False.
    """
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise ValueError("logits must be float32")
    if logits.ndim != 3 or min(logits.shape) < 1:
        raise ValueError("logits must be nonempty [B,T,C]")
    integer(blank_id, "blank_id")
    if blank_id >= logits.shape[2]:
        raise ValueError("blank_id outside logits")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite logits")
    batch, time, classes = logits.shape
    outputs = lengths_tensor(output_lengths, "output_lengths", batch, time)
    lengths = lengths_tensor(target_lengths, "target_lengths", batch, time, minimum=0)
    if isinstance(targets, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in targets):
        raise ValueError("targets must not contain boolean values")
    ids = torch.as_tensor(targets)
    if ids.dtype != torch.int64 or ids.ndim != 1 or len(ids) != int(lengths.sum()):
        raise ValueError("targets must be packed int64 matching target_lengths")
    ids = ids.detach().to(device="cpu").clone()
    if ((ids < 0) | (ids >= classes) | (ids == blank_id)).any():
        raise ValueError("target IDs contain blank or unknown class")
    offset = 0
    for index, size in enumerate(lengths.tolist()):
        sample = ids[offset:offset + size]
        required = size + int((sample[1:] == sample[:-1]).sum())
        if required > int(outputs[index]):
            raise ValueError(f"impossible CTC alignment at batch index {index}")
        offset += size
    # CPU CTC avoids nondeterministic CUDA CTC backward. The device copy keeps
    # autograd connected to CUDA logits and model parameters.
    log_probs = logits.cpu().log_softmax(-1).transpose(0, 1).contiguous()
    losses = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=False)(
        log_probs, ids, outputs, lengths)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("CTC produced NaN or infinite loss")
    loss = (losses / lengths.clamp_min(1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("CTC reduction produced nonfinite loss")
    return loss


@dataclass(frozen=True)
class CTCTrainingConfig:
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 5.0
    seed: int = 0

    def __post_init__(self):
        integer(self.seed, "seed")
        if self.seed > np.iinfo(np.uint32).max:
            raise ValueError("seed must fit uint32")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"invalid {name}")


def batch_tensors(batch, device):
    """Duck-type PR #16 CTCBatch or accept a mapping with the same fields."""
    def field(name):
        return batch[name] if isinstance(batch, dict) else getattr(batch, name)
    features = torch.as_tensor(field("features"), device=device)
    inputs = field("input_lengths")
    # Validate optional masks/output lengths against the authoritative lengths.
    getter = batch.get if isinstance(batch, dict) else lambda key: getattr(batch, key, None)
    if features.ndim != 3:
        raise ValueError("features must be [B,T,F]")
    lengths = lengths_tensor(inputs, "input_lengths", len(features), features.shape[1])
    mask = getter("padding_mask")
    if mask is not None:
        mask = torch.as_tensor(mask, device="cpu")
        expected = torch.arange(features.shape[1])[None] < lengths[:, None]
        if mask.dtype != torch.bool or mask.shape != expected.shape or not torch.equal(mask, expected):
            raise ValueError("padding_mask must mark the real contiguous prefix True")
    outputs = getter("output_lengths")
    if outputs is not None and not torch.equal(
            lengths_tensor(outputs, "output_lengths", len(features), features.shape[1]), lengths):
        raise ValueError("output_lengths disagree with length-preserving model")
    return features, lengths, field("targets"), field("target_lengths")


class CTCTrainer:
    """Seed before constructing the model for reproducible initial weights.

    fit accepts factories returning fresh train/validation iterables per epoch.
    Factories using global Python/NumPy/Torch RNG resume exactly on the same
    device. Independently seeded DataLoader generators are caller-managed.
    """

    def __init__(self, model, config=CTCTrainingConfig(), *, device="cpu", vocabulary=None,
                 feature_schema=None):
        seed_everything(config.seed)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.vocabulary = vocabulary
        self.feature_schema = feature_schema
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                                            weight_decay=config.weight_decay)
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.history = []
        self.best_checkpoint = None

    def run_epoch(self, batches, *, training):
        self.model.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for batch in batches:
                features, inputs, targets, lengths = batch_tensors(batch, self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits, outputs = self.model(features, inputs)
                loss = ctc_loss(logits, outputs, targets, lengths,
                                blank_id=self.model.config.blank_id)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip,
                                                   error_if_nonfinite=True)
                    self.optimizer.step()
                    self.global_step += 1
                total += float(loss.detach()) * len(features)
                count += len(features)
        if count == 0:
            raise ValueError("epoch requires at least one example")
        return total / count

    def fit(self, train_batches, validation_batches, *, epochs, checkpoint_dir):
        from .ctc_checkpoint import save_checkpoint
        integer(epochs, "epochs", 1)
        directory = Path(checkpoint_dir)
        for _ in range(epochs):
            train_loss = self.run_epoch(train_batches(), training=True)
            validation_loss = self.run_epoch(validation_batches(), training=False)
            self.epoch += 1
            self.history.append({"epoch": self.epoch, "train_loss": train_loss,
                                 "validation_loss": validation_loss})
            improved = validation_loss < self.best_loss
            if improved:
                self.best_loss = validation_loss
                from .ctc_checkpoint import checkpoint_payload
                self.best_checkpoint = checkpoint_payload(self, include_best=False)
            # last.pt is the authoritative completed-epoch state. best.pt can
            # always be reconstructed from it after an interrupted second write.
            save_checkpoint(directory / "last.pt", self)
            from .ctc_checkpoint import write_checkpoint
            write_checkpoint(directory / "best.pt", self.best_checkpoint)
        return list(self.history)

    def resume(self, path):
        from .ctc_checkpoint import restore_trainer
        restore_trainer(path, self)
