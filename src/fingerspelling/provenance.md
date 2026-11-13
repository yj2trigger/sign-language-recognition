- `8f23314`: jamo
- `b37954a`: tokens, ctc_targets
- `5a29221`: ctc_decode, cer_metrics
- `9e22ab2`: sequence_features, sequence_preprocessing, sequence_validation
- `05acbd2`: ctc_model, ctc_checkpoint, ctc_training, ctc_inference
Byte-identical reuse; no Git history dependency at runtime.

The 12 modules above are needed by the MediaPipe runner/reader. The unused
ctc_batching.py remains local and is excluded from this PR. Batching in the
experiment runner uses its validated features and the existing trainer mapping contract.
