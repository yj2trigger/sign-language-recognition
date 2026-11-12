"""Versioned, atomic CTC checkpoints with optimizer and global RNG state."""
from dataclasses import asdict
import copy
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .ctc_model import CTCModelConfig, FingerspellingCTC, integer


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("exact resume requires the saved CUDA device topology")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    torch.use_deterministic_algorithms(state["deterministic"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = state.get("cudnn_allow_tf32", False)
    torch.backends.cuda.matmul.allow_tf32 = state.get("matmul_allow_tf32", False)


def checkpoint_payload(trainer, *, include_best=True):
    payload = {"format": "fingerspelling-ctc", "version": 1,
               "model_config": trainer.model.configuration(),
               "model_state": trainer.model.state_dict(),
               "training_config": asdict(trainer.config),
               "optimizer_state": trainer.optimizer.state_dict(),
               "epoch": trainer.epoch, "global_step": trainer.global_step,
               "best_loss": trainer.best_loss, "history": trainer.history,
               "vocabulary": trainer.vocabulary, "feature_schema": trainer.feature_schema,
               "device_type": trainer.device.type, "rng": capture_rng()}
    if include_best:
        payload["best_checkpoint"] = trainer.best_checkpoint
    return copy.deepcopy(payload)


def save_checkpoint(path, trainer):
    """Replace one file atomically. Save only at a successfully completed epoch.

    Metadata must contain weights-only-loader-compatible primitives/tensors.
    vocabulary should be the externally supplied token->ID mapping, and
    feature_schema should specify channel order, normalization and timing.
    """
    write_checkpoint(path, checkpoint_payload(trainer))


def write_checkpoint(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                            dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    """Load on CPU with the restricted weights-only unpickler."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "fingerspelling-ctc" or payload.get("version") != 1:
        raise ValueError("unsupported CTC checkpoint")
    required = {"model_config", "model_state", "training_config", "optimizer_state",
                "epoch", "global_step", "best_loss", "history", "rng", "device_type",
                "vocabulary", "feature_schema"}
    if not required <= payload.keys():
        raise ValueError("incomplete CTC checkpoint")
    CTCModelConfig(**payload["model_config"])
    integer(payload["epoch"], "checkpoint epoch")
    integer(payload["global_step"], "checkpoint global_step")
    history = payload["history"]
    if not isinstance(history, list) or len(history) != payload["epoch"]:
        raise ValueError("invalid checkpoint history")
    for index, row in enumerate(history, 1):
        if (not isinstance(row, dict) or row.get("epoch") != index or
                any(not isinstance(row.get(key), (int, float)) or
                    not math.isfinite(row[key]) for key in ("train_loss", "validation_loss"))):
            raise ValueError("invalid checkpoint history entry")
    expected_best = min((row["validation_loss"] for row in history), default=float("inf"))
    if payload["best_loss"] != expected_best:
        raise ValueError("invalid checkpoint best_loss")
    if any(not torch.isfinite(value).all() for value in payload["model_state"].values()):
        raise ValueError("checkpoint has nonfinite model weights")
    return payload


def restore_trainer(path, trainer):
    payload = load_checkpoint(path)
    if payload["model_config"] != trainer.model.configuration():
        raise ValueError("checkpoint model configuration mismatch")
    if payload["training_config"] != asdict(trainer.config):
        raise ValueError("checkpoint training configuration mismatch")
    if payload["vocabulary"] != trainer.vocabulary or payload["feature_schema"] != trainer.feature_schema:
        raise ValueError("checkpoint vocabulary or feature schema mismatch")
    if payload["device_type"] != trainer.device.type:
        raise ValueError("exact training resume requires the saved device type")
    # Validate weights/optimizer on temporary objects before changing live state.
    current_rng = capture_rng()
    try:
        candidate = FingerspellingCTC(trainer.model.config).to(trainer.device)
        candidate.load_state_dict(payload["model_state"], strict=True)
        optimizer = torch.optim.AdamW(candidate.parameters())
        optimizer.load_state_dict(payload["optimizer_state"])
        validate_optimizer_state(optimizer)
        restore_rng(payload["rng"])
    except Exception:
        restore_rng(current_rng)
        raise
    trainer.model = candidate
    trainer.optimizer = optimizer
    trainer.epoch = payload["epoch"]
    trainer.global_step = payload["global_step"]
    trainer.best_loss = payload["best_loss"]
    trainer.history = payload["history"]
    trainer.best_checkpoint = payload.get("best_checkpoint")
    if trainer.best_checkpoint is None and trainer.epoch:
        # A best.pt is itself a complete restart point.
        trainer.best_checkpoint = copy.deepcopy(payload)


def validate_optimizer_state(optimizer):
    """Reject malformed AdamW moments before applying a resumed optimizer."""
    for group in optimizer.param_groups:
        for name in ("lr", "eps", "weight_decay"):
            value = group[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid optimizer {name}")
        if any(not isinstance(value, (int, float)) or not 0 <= value < 1 for value in group["betas"]):
            raise ValueError("invalid optimizer betas")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                required.add("max_exp_avg_sq")
            if not required <= state.keys():
                raise ValueError("incomplete optimizer state")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1 or
                    not torch.isfinite(step).all() or step.item() < 0 or step.item() != int(step.item())):
                raise ValueError("invalid optimizer step")
            for name in required - {"step"}:
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or
                        value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                    raise ValueError(f"invalid optimizer {name}")
                if name != "exp_avg" and (value < 0).any():
                    raise ValueError(f"negative optimizer {name}")
