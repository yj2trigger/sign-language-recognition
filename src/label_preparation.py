"""Strict parser for the morpheme subset consumed by build_dataset.py.

This is NOT a verified AI-Hub road-name adapter. No filename-to-road-name
mapping, jamo vocabulary, normalization, or tokenization is inferred here.
Only synthetic fixtures are distributed. Errors never include source values.
"""
from dataclasses import dataclass, field
import json
import math
import unicodedata
from collections.abc import Iterable


class LabelError(ValueError):
    """Invalid or ambiguous input; messages contain field positions only."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str = field(repr=False)


@dataclass(frozen=True)
class Target:
    """Internal per-video target. Identity must be supplied by a verified adapter.

    signer_id is a local pseudonymous grouping key, not a person's name.
    text is preserved exactly; segments are morphemes, never jamo alignments.
    """
    video_id: str = field(repr=False)
    signer_id: str = field(repr=False)
    text: str = field(repr=False)
    segments: tuple[Segment, ...] = field(default=(), repr=False)


def _text(value, location):
    if not isinstance(value, str) or not value.strip():
        raise LabelError(f"{location}: expected nonempty string")
    # Reject invalid Unicode, hidden controls and replacement decoding artifacts.
    if any(unicodedata.category(c).startswith("C") or c == "\ufffd" for c in value):
        raise LabelError(f"{location}: invalid Unicode/control character")
    return value


def _time(value, location):
    # The public table/examples use decimal strings; existing code uses numbers.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise LabelError(f"{location}: expected decimal seconds")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise LabelError(f"{location}: invalid seconds") from None
    if not math.isfinite(result) or result < 0:
        raise LabelError(f"{location}: expected finite nonnegative seconds")
    return result


def parse_morphemes(document: dict) -> tuple[Segment, ...]:
    """Parse {data: [{start, end, attributes: [{name}]}]} into all segments.

    Extra metadata is ignored. Empty data, multiple name attributes, overlaps,
    and nonchronological intervals are rejected instead of silently selecting
    the first annotation. Missing keys raise LabelError, not KeyError.
    """
    if not isinstance(document, dict):
        raise LabelError("root: expected object")
    data = document.get("data")
    if not isinstance(data, list) or not data:
        raise LabelError("data: expected nonempty array")
    result = []
    for i, item in enumerate(data):
        loc = f"data[{i}]"
        if not isinstance(item, dict):
            raise LabelError(f"{loc}: expected object")
        start = _time(item.get("start"), f"{loc}.start")
        end = _time(item.get("end"), f"{loc}.end")
        if end <= start or (result and start < result[-1].end):
            raise LabelError(f"{loc}: invalid interval order/overlap")
        attrs = item.get("attributes")
        if not isinstance(attrs, list) or len(attrs) != 1 or not isinstance(attrs[0], dict):
            raise LabelError(f"{loc}.attributes: expected one name attribute")
        text = _text(attrs[0].get("name"), f"{loc}.attributes[0].name")
        result.append(Segment(start, end, text))
    return tuple(result)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise LabelError("JSON: duplicate object key")
        obj[key] = value
    return obj


def decode_morphemes(raw: bytes) -> tuple[Segment, ...]:
    """Decode UTF-8 JSON bytes (optional BOM); return validated morphemes.

    No encoding fallback or lossy replacement is performed. No file is read.
    """
    if not isinstance(raw, bytes):
        raise LabelError("input: expected UTF-8 bytes")
    try:
        document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except LabelError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise LabelError("input: invalid UTF-8 JSON") from None
    return parse_morphemes(document)


def make_target(*, video_id: str, signer_id: str, text: str) -> Target:
    """Create a target from externally verified video/signer/text association.

    Numbers, punctuation, spacing and normalization form remain unchanged.
    This validates internal representation, not AI-Hub provenance or vocabulary.
    """
    return Target(_text(video_id, "video_id"), _text(signer_id, "signer_id"), _text(text, "text"))


def target_from_morphemes(document: dict, *, video_id: str, signer_id: str) -> Target:
    """Accept exactly one morpheme as one target; never concatenate glosses.

    Caller must verify that the single name is a full transcription. A general
    morpheme gloss is not automatically a road-name spelling.
    """
    segments = parse_morphemes(document)
    if len(segments) != 1:
        raise LabelError("data: single-target conversion requires exactly one segment")
    target = make_target(video_id=video_id, signer_id=signer_id, text=segments[0].text)
    return Target(target.video_id, target.signer_id, target.text, segments)


def validate_targets(targets: Iterable[Target]) -> tuple[Target, ...]:
    """Return validated batch, rejecting duplicate video IDs, even identical rows.

    Same text across different videos is allowed. No deduplication by text.
    """
    try:
        iterator = iter(targets)
    except TypeError:
        raise LabelError("targets: expected iterable") from None
    result, seen = [], set()
    for item in iterator:
        if not isinstance(item, Target):
            raise LabelError("targets: expected Target records")
        make_target(video_id=item.video_id, signer_id=item.signer_id, text=item.text)
        if not isinstance(item.segments, tuple) or len(item.segments) > 1:
            raise LabelError("targets: expected zero or one target segment")
        if item.segments:
            seg = item.segments[0]
            if not isinstance(seg, Segment):
                raise LabelError("targets: expected Segment record")
            start, end = _time(seg.start, "segment.start"), _time(seg.end, "segment.end")
            if end <= start or _text(seg.text, "segment.text") != item.text:
                raise LabelError("targets: invalid segment or target text mismatch")
        if item.video_id in seen:
            raise LabelError("targets: duplicate video_id")
        seen.add(item.video_id)
        result.append(item)
    return tuple(result)


def validate_signer_splits(splits: dict[str, Iterable[Target]]) -> None:
    """Reject repeated video IDs or signer IDs between named dataset splits.

    No random clip split is generated. Empty splits are permitted. This checks
    only supplied identities; wrong or missing source identities need an audit.
    """
    if not isinstance(splits, dict):
        raise LabelError("splits: expected object")
    videos, signers = set(), set()
    for name, rows in splits.items():
        _text(name, "split name")
        batch = validate_targets(rows)
        current_videos = {r.video_id for r in batch}
        current_signers = {r.signer_id for r in batch}
        if videos & current_videos or signers & current_signers:
            raise LabelError("splits: video or signer leakage")
        videos.update(current_videos)
        signers.update(current_signers)
