"""CTC targets with injected inventory and PR #11 decomposition dependency.

No Unicode decomposition or official 37-class definition is implemented here.
Input length always means model OUTPUT time steps, after any subsampling.
"""
from collections import Counter
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
import unicodedata

from .tokens import TokenError, TokenVocabulary


class CTCError(ValueError):
    """Sanitized reason code; no source text in exception messages."""


@dataclass(frozen=True)
class CTCTarget:
    original_text: str = field(repr=False)
    normalized_text: str = field(repr=False)
    tokens: tuple[str, ...] = field(repr=False)
    ids: tuple[int, ...] = field(repr=False)
    minimum_input_length: int
    input_length: int | None


class TargetEncoder:
    def __init__(self, vocabulary: TokenVocabulary, *, decompose=None,
                 literal_mapping=None):
        self.vocabulary = vocabulary
        self._decompose = decompose
        mapping = {} if literal_mapping is None else dict(literal_mapping)
        for char, tokens in mapping.items():
            # Literal overrides apply only outside Hangul. No silent deletions.
            if (not isinstance(char, str) or len(char) != 1
                    or unicodedata.category(char).startswith("C")
                    or self._hangul(char) or char == "\ufffd"
                    or unicodedata.normalize("NFC", char) != char):
                raise CTCError("invalid_literal_mapping")
            if not isinstance(tokens, tuple) or not tokens:
                raise CTCError("invalid_literal_mapping")
            vocabulary.encode(tokens)
        self.literal_mapping = MappingProxyType(mapping)

    @staticmethod
    def _hangul(char):
        return ("\uac00" <= char <= "\ud7a3" or "\u3131" <= char <= "\u318e")

    def _split(self, char):
        splitter = self._decompose
        if splitter is None:
            try:
                splitter = import_module(".jamo", __package__).decompose_hangul
            except ModuleNotFoundError as exc:
                if exc.name != __package__ + ".jamo":
                    raise
                raise CTCError("missing_pr11_dependency") from None
        try:
            tokens = splitter(char)
        except (TypeError, ValueError):
            raise CTCError("decomposition_rejected") from None
        if not isinstance(tokens, (list, tuple)) or not tokens:
            raise CTCError("invalid_decomposition")
        return tuple(tokens)

    def encode(self, text: str, *, input_length: int | None = None) -> CTCTarget:
        if not isinstance(text, str):
            raise CTCError("invalid_text_type")
        if input_length is not None and (type(input_length) is not int or input_length < 0):
            raise CTCError("invalid_input_length")
        normalized = unicodedata.normalize("NFC", text)
        if not normalized:
            raise CTCError("empty_text")
        tokens = []
        for char in normalized:
            if char in self.literal_mapping:
                tokens.extend(self.literal_mapping[char])
            elif self._hangul(char):
                tokens.extend(self._split(char))
            elif char.isdecimal():
                raise CTCError("digit_policy_required")
            elif char.isspace():
                raise CTCError("whitespace_policy_required")
            else:
                raise CTCError("unsupported_character")
        ids = self.vocabulary.encode(tokens)
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if input_length is not None and input_length < minimum:
            raise CTCError("insufficient_input_length")
        return CTCTarget(text, normalized, tuple(tokens), ids, minimum, input_length)


def audit_crowd(report, encoder: TargetEncoder) -> dict:
    """Aggregate adapter records only. Never return labels, filenames or IDs.

    Character type counts are occurrences in NFC text, including every rejected
    character, even if conversion stops at an earlier error. No video length is
    available here; success does not imply CTC temporal feasibility.
    """
    errors, unsupported = Counter(), Counter()
    converted = empty = digits = digit_converted = 0
    for record in report.records:
        text = record.target.text
        normalized = unicodedata.normalize("NFC", text)
        has_digit = any(c.isdecimal() for c in normalized)
        digits += has_digit
        empty += not normalized
        for char in normalized:
            rejected = not encoder._hangul(char) and char not in encoder.literal_mapping
            if encoder._hangul(char):
                try:
                    encoder._split(char)
                except CTCError as exc:
                    rejected = str(exc) == "decomposition_rejected"
            if rejected:
                kind = ("decimal" if char.isdecimal() else "whitespace" if char.isspace()
                        else unicodedata.category(char))
                unsupported[kind] += 1
        try:
            encoder.encode(text)
        except (CTCError, TokenError) as exc:
            errors[str(exc)] += 1
        else:
            converted += 1
            digit_converted += has_digit
    return {"total_labels": report.json_count, "adapter_records": len(report.records),
            "adapter_errors": sum(n for _, n in report.errors),
            "converted": converted, "unconvertible": len(report.records) - converted,
            "empty_strings": empty, "digit_strings": digits,
            "digit_converted": digit_converted, "digit_rejected": digits - digit_converted,
            "unsupported_character_occurrences": dict(sorted(unsupported.items())),
            "errors": dict(sorted(errors.items()))}
