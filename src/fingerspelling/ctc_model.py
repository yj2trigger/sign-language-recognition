"""Length-preserving packed BiGRU for externally defined CTC vocabularies."""
from dataclasses import asdict, dataclass
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def lengths_tensor(value, name, batch_size, maximum, minimum=1):
    if isinstance(value, (list, tuple)) and any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError(f"{name} must contain integers without booleans")
    result = torch.as_tensor(value)
    if result.dtype not in (torch.int32, torch.int64) or result.shape != (batch_size,):
        raise ValueError(f"{name} must be an integer vector of batch size")
    result = result.detach().to(device="cpu", dtype=torch.int64).clone()
    if ((result < minimum) | (result > maximum)).any():
        raise ValueError(f"{name} outside supported range")
    return result


@dataclass(frozen=True)
class CTCModelConfig:
    input_size: int
    num_classes: int
    blank_id: int
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self):
        for name in ("input_size", "hidden_size", "num_layers"):
            integer(getattr(self, name), name, 1)
        integer(self.num_classes, "num_classes", 2)
        integer(self.blank_id, "blank_id")
        if self.blank_id >= self.num_classes:
            raise ValueError("blank_id outside num_classes")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError("dropout must be a finite number")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class FingerspellingCTC(nn.Module):
    """Accept float32 [B,T,F], return finite logits [B,T,C] and CPU lengths.

    F is supplied by the feature producer. Raw 53x2 coordinates can be flattened
    to F=106 by the caller. Missing-joint processing belongs to PR #16.
    All input cells must be finite, including right padding. Padding logits are
    zero and callers must still pass output_lengths to the decoder/loss.
    """

    def __init__(self, config: CTCModelConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size), nn.GELU(), nn.Dropout(config.dropout),
        )
        self.rnn = nn.GRU(
            config.hidden_size, config.hidden_size, config.num_layers,
            batch_first=True, bidirectional=True,
            # cuDNN's internal RNN dropout state is not exposed in state_dict.
            # Explicit projection/head dropout uses the checkpointed Torch RNG.
            dropout=0,
        )
        self.head = nn.Sequential(nn.Dropout(config.dropout),
                                  nn.Linear(2 * config.hidden_size, config.num_classes))

    def output_lengths(self, input_lengths):
        """Identity transform, also accepts PR #16 NumPy length vectors."""
        values = torch.as_tensor(input_lengths)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("input_lengths must be a nonempty vector")
        return lengths_tensor(input_lengths, "input_lengths", len(values), torch.iinfo(torch.int64).max)

    def forward(self, features, input_lengths):
        if not isinstance(features, torch.Tensor) or features.dtype != torch.float32:
            raise ValueError("features must be a float32 tensor")
        if features.ndim != 3 or min(features.shape[:2]) < 1:
            raise ValueError("features must have nonempty [B,T,F] shape")
        if features.shape[2] != self.config.input_size:
            raise ValueError("feature width differs from input_size")
        if features.device != next(self.parameters()).device:
            raise ValueError("features and model must use the same device")
        if not torch.isfinite(features).all():
            raise ValueError("features contain nonfinite values")
        lengths = lengths_tensor(input_lengths, "input_lengths", len(features), features.shape[1])
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        # Project only real frames, preserving the PackedSequence bookkeeping.
        projected = packed._replace(data=self.projection(packed.data))
        hidden, _ = self.rnn(projected)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=features.shape[1])
        logits = self.head(hidden)
        valid = torch.arange(features.shape[1], device=features.device)[None] < lengths.to(features.device)[:, None]
        logits = logits.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model produced nonfinite logits")
        return logits, lengths

    def configuration(self):
        return asdict(self.config)
