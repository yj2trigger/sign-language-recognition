"""Lossless partial rendering with the project's pinned Hangul grammar.

No reference text, vocabulary frequency or dictionary enters these functions.
"""
import unicodedata

from . import jamo as _jamo


def validate_tokens(tokens):
    jamo = _jamo
    if not isinstance(tokens, list):
        raise TypeError("tokens must be a list")
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("tokens must contain strings")
        if token not in jamo.JAMO:
            raise ValueError("unsupported_token")
    return jamo


def syllable_span(tokens, start, jamo):
    """Apply the existing default composer's two/three-token lookahead."""
    if tokens[start] not in jamo.INITIALS:
        return None, "expected_initial"
    if start + 1 >= len(tokens):
        return None, "missing_medial"
    if tokens[start + 1] not in jamo.MEDIALS:
        return None, "expected_medial"
    end = start + 2
    if end < len(tokens) and tokens[end] in jamo.FINALS[1:]:
        next_initial = (tokens[end] in jamo.INITIALS and end + 1 < len(tokens)
                        and tokens[end + 1] in jamo.MEDIALS)
        if not next_initial:
            end += 1
    return end, None


def strict_diagnostics(tokens):
    jamo = validate_tokens(tokens)
    try:
        text = jamo.compose_hangul(tokens)
    except ValueError:
        prefix, start = [], 0
        while start < len(tokens):
            end, reason = syllable_span(tokens, start, jamo)
            if end is None:
                return dict(success=False, text="".join(tokens), failure_index=start,
                            failure_reason=reason, valid_prefix_tokens=start,
                            valid_prefix_text="".join(prefix))
            prefix.append(jamo.compose_hangul(tokens[start:end], preserve_boundaries=True))
            start = end
        raise AssertionError("strict_and_segment_grammar_disagree")
    return dict(success=True, text=unicodedata.normalize("NFC", text),
                failure_index=None, failure_reason=None,
                valid_prefix_tokens=len(tokens), valid_prefix_text=text)


def compose_partial(tokens, *, mode="resync"):
    """Render valid spans, preserving every original token in order.

    prefix stops composition at the first malformed span. resync preserves
    one malformed token and resumes scanning. Both use the original syllable
    boundary rule, without merging/splitting compound jamo or inserting ㅇ.
    """
    if mode not in ("prefix", "resync"):
        raise ValueError("invalid_partial_mode")
    jamo = validate_tokens(tokens)
    original = strict_diagnostics(tokens)
    if original["success"]:
        return original["text"]
    pieces, start = [], 0
    while start < len(tokens):
        end, _ = syllable_span(tokens, start, jamo)
        if end is None:
            if mode == "prefix":
                pieces.extend(tokens[start:])
                break
            pieces.append(tokens[start])
            start += 1
        else:
            pieces.append(jamo.compose_hangul(tokens[start:end], preserve_boundaries=True))
            start = end
    text = unicodedata.normalize("NFC", "".join(pieces))
    if jamo.decompose_hangul(text) != tokens:
        raise AssertionError("partial_composition_changed_tokens")
    return text
