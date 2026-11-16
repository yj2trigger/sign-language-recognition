"""Explicit, immutable vocabulary. No official fingerspelling inventory inferred."""
from collections.abc import Mapping
from types import MappingProxyType


class TokenError(ValueError):
    """Fixed codes only; never expose source labels or tokens."""


class TokenVocabulary:
    def __init__(self, token_ids: Mapping[str, int], *, blank_id: int):
        if not isinstance(token_ids, Mapping) or not token_ids:
            raise TokenError("invalid_vocabulary")
        copied = dict(token_ids)
        if any(not isinstance(t, str) or not t for t in copied):
            raise TokenError("invalid_token")
        if type(blank_id) is not int or blank_id < 0:
            raise TokenError("invalid_blank")
        if any(type(i) is not int or i < 0 for i in copied.values()):
            raise TokenError("invalid_id")
        if len(set(copied.values())) != len(copied):
            raise TokenError("duplicate_id")
        if blank_id in copied.values():
            raise TokenError("blank_collision")
        self._token_ids = MappingProxyType(copied)
        self._blank_id = blank_id

    @property
    def token_ids(self):
        return self._token_ids

    @property
    def blank_id(self):
        return self._blank_id

    @property
    def num_classes(self):
        """Required logits width; sparse explicit IDs are allowed."""
        return max(self.blank_id, *self.token_ids.values()) + 1

    def encode(self, tokens):
        result = []
        for token in tokens:
            if not isinstance(token, str) or token not in self.token_ids:
                raise TokenError("unregistered_token")
            result.append(self.token_ids[token])
        return tuple(result)
