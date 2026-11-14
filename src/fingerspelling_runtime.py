"""15fps MediaPipe 53-point CTC interface, independent of the realtime loop."""
from functools import lru_cache
import hashlib
import inspect
import os
from pathlib import Path

import numpy as np
from fingerspelling import jamo
from fingerspelling.tokens import TokenVocabulary
from fingerspelling.ctc_targets import TargetEncoder
from fingerspelling.sequence_features import build_sequence_features
from fingerspelling.sequence_validation import validate_sequence
from fingerspelling.ctc_inference import CTCRecognizer
from fingerspelling.ctc_decode import greedy_decode


def vocabulary():
    return TokenVocabulary({token: i + 1 for i, token in enumerate(sorted(jamo.JAMO))}, blank_id=0)


def feature_schema():
    base = Path(__file__).parent
    paths = [*(base / 'fingerspelling' / (name + '.py') for name in
             ('sequence_features', 'sequence_preprocessing', 'sequence_validation'))]
    return dict(version='mediapipe53-ctc-v1', input_size=424, fps=15,
                normalization='shoulder', motion='delta', max_gap=3, aspect=16/9,
                feature_function_sha256=hashlib.sha256(inspect.getsource(features).encode()).hexdigest(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def features(xy, mask):
    """Keep all time steps and original observed mask; interpolate <=3 frames."""
    result = build_sequence_features(xy, mask, fps=15, motion='delta',
                                     normalization='shoulder', max_gap=3, aspect=16/9)
    if not result.prepared.available[:, 11:].any():
        raise ValueError('no_usable_hands')
    return result.values


def tokens_for(ids):
    inverse = {i: token for token, i in vocabulary().token_ids.items()}
    return [inverse[i] for i in ids]


def render(ids,mode='strict'):
    tokens = tokens_for(ids)
    if mode in ('prefix','resync'):
        from fingerspelling.partial_compose import compose_partial
        return compose_partial(tokens,mode=mode)
    if mode!='strict':raise ValueError('invalid_render_mode')
    try:
        return jamo.compose_hangul(tokens)
    except ValueError:
        # Preserve undecodable jamo; never invent missing consonants/vowels.
        return ''.join(tokens)


class FingerspellingReader:
    def __init__(self, checkpoint, *, device='cpu',render_mode='strict'):
        if render_mode not in ('strict','prefix','resync'):raise ValueError('invalid_render_mode')
        self.render_mode=render_mode
        vocab = vocabulary()
        self.recognizer = CTCRecognizer(checkpoint, device=device,
            expected_vocabulary=dict(token_ids=dict(vocab.token_ids), blank_id=0),
            expected_feature_schema=feature_schema())
        config = self.recognizer.model.config
        if config.input_size != 424 or config.num_classes != vocab.num_classes or config.blank_id != 0:
            raise ValueError('checkpoint_input_or_classes_mismatch')

    def __call__(self, xy, mask):
        xy, mask = np.asarray(xy), np.asarray(mask)
        if xy.shape == (0, 53, 2) and mask.shape == (0, 53) and mask.dtype == np.bool_:
            return ''
        validate_sequence(xy, mask, allow_all_missing=True)
        if len(xy) < 2 or not mask[:, 11:].any():
            return ''
        try:
            values = features(xy, mask)
        except ValueError as exc:
            if str(exc) == 'no_usable_hands':
                return ''
            raise
        scores, lengths = self.recognizer.predict(values[None], [len(values)])
        ids = greedy_decode(scores[0].tolist(), blank_id=0, num_classes=52, length=int(lengths[0]))
        return render(ids,self.render_mode)


@lru_cache(maxsize=2)
def load_reader(checkpoint, device='cpu',render_mode='strict'):
    """Cache one reader per immutable checkpoint path/device; no per-call loads."""
    return FingerspellingReader(checkpoint, device=device,render_mode=render_mode)


def read_fingerspelling(xy, mask) -> str:
    """Set FINGERSPELLING_CHECKPOINT once to a matching MediaPipe checkpoint."""
    checkpoint = os.environ.get('FINGERSPELLING_CHECKPOINT')
    if not checkpoint:
        raise RuntimeError('set FINGERSPELLING_CHECKPOINT to a MediaPipe CTC checkpoint')
    return load_reader(str(Path(checkpoint).resolve()),render_mode=os.environ.get('FINGERSPELLING_RENDER_MODE','strict'))(xy, mask)
