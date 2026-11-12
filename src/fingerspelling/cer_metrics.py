"""Unicode code-point CER with deterministic edit counts and linear DP memory."""
from dataclasses import dataclass
from math import inf
import unicodedata


class MetricError(ValueError):
    """Invalid evaluation text or policy."""


@dataclass(frozen=True)
class TextPolicy:
    """NFC by default; preserve digits, punctuation, case and all whitespace.

    normalize_nfc=False compares raw Unicode code points. No NFKC folding,
    stripping, digit conversion or grapheme clustering is performed.
    """
    normalize_nfc: bool = True

    def __post_init__(self):
        if type(self.normalize_nfc) is not bool:
            raise MetricError("invalid_normalization_policy")

    def apply(self, text):
        if not isinstance(text, str):
            raise MetricError("invalid_text")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
            raise MetricError("invalid_unicode")
        return unicodedata.normalize("NFC", text) if self.normalize_nfc else text


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_characters: int = 0

    def __post_init__(self):
        values = (self.substitutions, self.deletions, self.insertions,
                  self.reference_characters)
        if any(type(v) is not int or v < 0 for v in values):
            raise MetricError("invalid_counts")
        if self.substitutions + self.deletions > self.reference_characters:
            raise MetricError("invalid_counts")

    @property
    def errors(self):
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self):
        """Empty reference: 0 for no errors, +inf for any insertions."""
        if not self.reference_characters:
            return inf if self.errors else 0.0
        return self.errors / self.reference_characters

    def __add__(self, other):
        if not isinstance(other, ErrorCounts):
            return NotImplemented
        return ErrorCounts(self.substitutions + other.substitutions,
                           self.deletions + other.deletions,
                           self.insertions + other.insertions,
                           self.reference_characters + other.reference_characters)


def edit_counts(reference, prediction, *, policy=TextPolicy()):
    """Levenshtein counts, O(N*M) time, O(M) memory.

    Each cell stores (distance,S,D,I); ties choose diagonal (match/substitute),
    then deletion, then insertion. Several optimal alignments may exist: this
    tie policy makes the error-type breakdown reproducible. Input is unchanged.
    """
    if not isinstance(policy, TextPolicy):
        raise MetricError("invalid_policy")
    reference, prediction = policy.apply(reference), policy.apply(prediction)
    if reference == prediction:
        return ErrorCounts(reference_characters=len(reference))
    if not reference:
        return ErrorCounts(insertions=len(prediction))
    if not prediction:
        return ErrorCounts(deletions=len(reference), reference_characters=len(reference))
    previous = [(j, 0, 0, j) for j in range(len(prediction) + 1)]
    for i, truth in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, predicted in enumerate(prediction, 1):
            d, s, delete, insert = previous[j - 1]
            mismatch = truth != predicted
            diagonal = (d + mismatch, s + mismatch, delete, insert)
            d, s, delete, insert = previous[j]
            deletion = (d + 1, s, delete + 1, insert)
            d, s, delete, insert = current[j - 1]
            insertion = (d + 1, s, delete, insert + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda x: x[0]))
        previous = current
    _, s, d, ins = previous[-1]
    return ErrorCounts(int(s), d, ins, len(reference))


def character_error_rate(reference, prediction, *, policy=TextPolicy()):
    return edit_counts(reference, prediction, policy=policy).cer
