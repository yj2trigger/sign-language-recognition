"""Checkpoint-backed batch inference returning PR #15-compatible scores."""
import torch

from .ctc_checkpoint import load_checkpoint
from .ctc_model import CTCModelConfig, FingerspellingCTC


class CTCRecognizer:
    def __init__(self, checkpoint, *, device="cpu", expected_vocabulary=None,
                 expected_feature_schema=None):
        payload = load_checkpoint(checkpoint)
        if expected_vocabulary is not None and payload["vocabulary"] != expected_vocabulary:
            raise ValueError("checkpoint vocabulary mismatch")
        if expected_feature_schema is not None and payload["feature_schema"] != expected_feature_schema:
            raise ValueError("checkpoint feature schema mismatch")
        self.device = torch.device(device)
        self.model = FingerspellingCTC(CTCModelConfig(**payload["model_config"]))
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.vocabulary = payload["vocabulary"]
        self.feature_schema = payload["feature_schema"]
        self.cudnn_allow_tf32 = payload["rng"].get("cudnn_allow_tf32", False)
        self.matmul_allow_tf32 = payload["rng"].get("matmul_allow_tf32", False)

    @torch.inference_mode()
    def predict(self, features, input_lengths):
        """Return CPU float32 [B,T,C] logits and CPU int64 [B] lengths.

        Input must already follow the checkpoint feature contract. Pass
        logits.tolist()/lengths.tolist() to PR #15's Python-sequence decoder.
        No token decoding or raw landmark preprocessing is performed here.
        """
        values = torch.as_tensor(features, device=self.device)
        self.model.eval()
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = self.matmul_allow_tf32
            with torch.backends.cudnn.flags(
                    enabled=torch.backends.cudnn.enabled,
                    benchmark=torch.backends.cudnn.benchmark,
                    deterministic=torch.backends.cudnn.deterministic,
                    allow_tf32=self.cudnn_allow_tf32):
                logits, lengths = self.model(values, input_lengths)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        return logits.cpu(), lengths
