"""Dependency-free, batch-major CTC greedy inference and explicit text rendering."""
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType


class DecodeError(ValueError):
    """Stable error codes, without input text or score values."""


def _integer(value, code):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise DecodeError(code)
    return int(value)


def _items(value, code):
    if isinstance(value, (str, bytes, Mapping)):
        raise DecodeError(code)
    try:
        return iter(value)
    except TypeError:
        raise DecodeError(code) from None


def greedy_decode(scores, *, blank_id, num_classes, length=None):
    """Decode [time, classes]; merge adjacent IDs BEFORE removing blank.

    Raw logits, probabilities and log probabilities are accepted. All scores
    must be finite (including padding); NaN and +/-inf are rejected. Exact ties
    choose the lowest class ID. length counts model OUTPUT frames, not video
    frames. Even ignored padding must have the declared width. No mutation.
    """
    width = _integer(num_classes, "invalid_num_classes")
    blank = _integer(blank_id, "invalid_blank")
    if width == 0 or blank >= width:
        raise DecodeError("invalid_class_contract")
    limit = None if length is None else _integer(length, "invalid_length")
    result, previous, count = [], None, 0
    for time, row in enumerate(_items(scores, "invalid_scores")):
        best_id, best_score, size = None, None, 0
        for index, value in enumerate(_items(row, "invalid_row")):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise DecodeError("invalid_score")
            try:
                finite = isfinite(value)
            except (OverflowError, TypeError, ValueError):
                raise DecodeError("invalid_score") from None
            if not finite:
                raise DecodeError("nonfinite_score")
            if best_score is None or value > best_score:
                best_id, best_score = index, value
            size += 1
        if size != width:
            raise DecodeError("class_count_mismatch")
        if limit is None or time < limit:
            if best_id != previous and best_id != blank:
                result.append(best_id)
            previous = best_id
        count += 1
    if limit is not None and limit > count:
        raise DecodeError("length_exceeds_time")
    return tuple(result)


def greedy_decode_batch(scores, *, blank_id, num_classes, lengths=None):
    """Decode [batch, time, classes], permitting ragged time dimensions.

    Empty batches are supported. Pass an explicit empty lengths iterable for an
    empty batch. No layout guessing or automatic tensor/device conversion.
    """
    # Validate the contract even for an empty batch.
    greedy_decode((), blank_id=blank_id, num_classes=num_classes)
    rows = _items(scores, "invalid_batch")
    if lengths is None:
        return tuple(greedy_decode(row, blank_id=blank_id, num_classes=num_classes)
                     for row in rows)
    limits = _items(lengths, "invalid_lengths")
    sentinel, output = object(), []
    for row in rows:
        limit = next(limits, sentinel)
        if limit is sentinel:
            raise DecodeError("batch_length_mismatch")
        # None is valid for the single decoder but not within explicit lengths.
        limit = _integer(limit, "invalid_length")
        output.append(greedy_decode(row, blank_id=blank_id,
                                   num_classes=num_classes, length=limit))
    if next(limits, sentinel) is not sentinel:
        raise DecodeError("batch_length_mismatch")
    return tuple(output)


class TokenRenderer:
    """Render an external token->ID mapping; never invent a token inventory.

    Sparse IDs are supported. Unknown IDs (including holes) raise errors.
    A composer receives a fresh list[str], compatible with PR #11's
    compose_hangul. Without a composer, tokens are joined literally.
    """

    def __init__(self, token_ids, *, blank_id, compose=None):
        if not isinstance(token_ids, Mapping) or not token_ids:
            raise DecodeError("invalid_vocabulary")
        blank = _integer(blank_id, "invalid_blank")
        inverse = {}
        for token, index in token_ids.items():
            if not isinstance(token, str) or not token:
                raise DecodeError("invalid_token")
            index = _integer(index, "invalid_token_id")
            if index == blank or index in inverse:
                raise DecodeError("duplicate_or_blank_id")
            inverse[index] = token
        if compose is not None and not callable(compose):
            raise DecodeError("invalid_composer")
        self._tokens = MappingProxyType(inverse)
        self._blank_id = blank
        self._num_classes = max(blank, *inverse) + 1
        self._compose = compose

    @property
    def blank_id(self):
        return self._blank_id

    @property
    def num_classes(self):
        return self._num_classes

    @classmethod
    def from_vocabulary(cls, vocabulary, *, compose=None):
        """Duck-type PR #13 TokenVocabulary, without importing its code."""
        try:
            mapping, blank, width = (vocabulary.token_ids, vocabulary.blank_id,
                                     vocabulary.num_classes)
        except AttributeError:
            raise DecodeError("invalid_vocabulary_interface") from None
        renderer = cls(mapping, blank_id=blank, compose=compose)
        if _integer(width, "invalid_num_classes") != renderer.num_classes:
            raise DecodeError("class_count_mismatch")
        return renderer

    def tokens(self, ids):
        result = []
        for index in _items(ids, "invalid_ids"):
            index = _integer(index, "invalid_token_id")
            if index == self.blank_id:
                continue
            if index not in self._tokens:
                raise DecodeError("unregistered_token_id")
            result.append(self._tokens[index])
        return tuple(result)

    def render(self, ids):
        tokens = self.tokens(ids)
        if self._compose is None:
            return "".join(tokens)
        try:
            result = self._compose(list(tokens))
        except (ValueError, TypeError, KeyError):
            raise DecodeError("composition_rejected") from None
        if not isinstance(result, str):
            raise DecodeError("invalid_composer_result")
        return result
