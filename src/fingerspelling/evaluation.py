"""Strict, model-independent evaluation with explicit identity provenance."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .cer_metrics import ErrorCounts, TextPolicy, edit_counts
from .ctc_decode import TokenRenderer, greedy_decode_batch


class EvaluationError(ValueError):
    """Stable validation errors, without private labels or identities."""


def _identity(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise EvaluationError("invalid_identity")
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise EvaluationError("invalid_identity")


@dataclass(frozen=True)
class Reference:
    video_id: str = field(repr=False)
    text: str = field(repr=False)
    performer_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class VideoResult:
    video_id: str = field(repr=False)
    performer_id: str | None = field(repr=False)
    counts: ErrorCounts


@dataclass(frozen=True)
class EvaluationReport:
    videos: tuple[VideoResult, ...] = field(repr=False)
    corpus: ErrorCounts
    by_performer: Mapping = field(repr=False)
    identity_status: str
    unverified_reasons: tuple[str, ...]
    policy: TextPolicy

    @property
    def mean_video_cer(self):
        """Unweighted mean; may be +inf. Empty corpus returns None."""
        return sum(v.counts.cer for v in self.videos) / len(self.videos) if self.videos else None


def _references(records):
    if isinstance(records, (str, bytes, Mapping)):
        raise EvaluationError("invalid_references")
    try:
        rows = tuple(records)
    except TypeError:
        raise EvaluationError("invalid_references") from None
    seen = set()
    for row in rows:
        if not isinstance(row, Reference):
            raise EvaluationError("invalid_reference")
        _identity(row.video_id)
        if row.video_id in seen:
            raise EvaluationError("duplicate_video_id")
        seen.add(row.video_id)
        if row.text is None:
            raise EvaluationError("missing_reference")
        TextPolicy().apply(row.text)
        if row.performer_id is not None:
            _identity(row.performer_id)
    return rows


def validate_evaluation_data(references, *, training_performer_ids=None,
                             training_video_ids=None, identities_verified=False):
    """Return (validated records, reasons for unverified held-out status).

    Identity strings alone are not proof of real people. identities_verified
    asserts an independently audited, shared performer-ID namespace, including
    the training inventory. CROWD IDs must not be assumed to provide that.
    Explicit empty training IDs mean 'known no training people'; None is unknown.
    """
    if type(identities_verified) is not bool:
        raise EvaluationError("invalid_identity_verification")
    rows = _references(references)
    reasons = []
    if not rows:
        reasons.append("empty_evaluation")
    if not identities_verified:
        reasons.append("identity_provenance_unverified")
    if any(row.performer_id is None for row in rows):
        reasons.append("missing_performer_identity")
    for inventory, code, values in (
        (training_performer_ids, "performer_leakage", {r.performer_id for r in rows if r.performer_id is not None}),
        (training_video_ids, "video_leakage", {r.video_id for r in rows}),
    ):
        if inventory is None:
            if code == "performer_leakage":
                reasons.append("training_performers_unknown")
            continue
        if isinstance(inventory, (str, bytes, Mapping)):
            raise EvaluationError("invalid_training_inventory")
        try:
            items = tuple(inventory)
        except TypeError:
            raise EvaluationError("invalid_training_inventory") from None
        for item in items:
            _identity(item)
        if values.intersection(items):
            raise EvaluationError(code)
    return rows, tuple(reasons)


def _keyed_predictions(predictions, expected):
    # Pair iterables preserve duplicate detection; mappings are also accepted.
    source = predictions.items() if isinstance(predictions, Mapping) else predictions
    if isinstance(source, (str, bytes)):
        raise EvaluationError("invalid_predictions")
    try:
        iterator = iter(source)
    except TypeError:
        raise EvaluationError("invalid_predictions") from None
    result = {}
    for pair in iterator:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise EvaluationError("invalid_prediction_pair")
        key, value = pair
        _identity(key)
        if key in result:
            raise EvaluationError("duplicate_prediction_id")
        if value is None:
            raise EvaluationError("missing_prediction")
        result[key] = value
    if expected - result.keys():
        raise EvaluationError("missing_prediction")
    if result.keys() - expected:
        raise EvaluationError("unexpected_prediction")
    return result


def _prepare(references, policy, training_performer_ids, training_video_ids,
             identities_verified, require_verified):
    if not isinstance(policy, TextPolicy) or type(require_verified) is not bool:
        raise EvaluationError("invalid_evaluation_policy")
    rows, reasons = validate_evaluation_data(
        references, training_performer_ids=training_performer_ids,
        training_video_ids=training_video_ids, identities_verified=identities_verified)
    if require_verified and reasons:
        raise EvaluationError("held_out_unverified")
    return rows, reasons


def _report(rows, predictions, policy, reasons):
    predicted = _keyed_predictions(predictions, {row.video_id for row in rows})
    total, grouped, results = ErrorCounts(), {}, []
    for row in rows:
        counts = edit_counts(row.text, predicted[row.video_id], policy=policy)
        total = total + counts
        grouped[row.performer_id] = grouped.get(row.performer_id, ErrorCounts()) + counts
        results.append(VideoResult(row.video_id, row.performer_id, counts))
    return EvaluationReport(tuple(results), total, MappingProxyType(grouped),
                            "verified" if not reasons else "unverified", reasons, policy)


def evaluate_texts(references, predictions, *, policy=TextPolicy(),
                   training_performer_ids=None, training_video_ids=None,
                   identities_verified=False, require_verified=False):
    """Compare exact video-ID sets; aggregate counts, never average CER as corpus.

    Missing values differ from valid empty strings. Unknown performers are
    grouped under None and are always marked unverified. No raw text in reports.
    """
    rows, reasons = _prepare(references, policy, training_performer_ids,
                             training_video_ids, identities_verified, require_verified)
    return _report(rows, predictions, policy, reasons)


def evaluate_scores(references, scores, *, renderer, lengths=None, policy=TextPolicy(),
                    training_performer_ids=None, training_video_ids=None,
                    identities_verified=False, require_verified=False):
    """Video-keyed [time,classes] scores -> IDs -> text -> CER report.

    lengths, when given, must be a video-keyed mapping with the same exact IDs.
    A model adapter must convert tensors to CPU lists/arrays, reorder to batch
    major and supply post-subsampling lengths; this module never loads a model.
    """
    if not isinstance(renderer, TokenRenderer):
        raise EvaluationError("invalid_renderer")
    rows, reasons = _prepare(references, policy, training_performer_ids,
                             training_video_ids, identities_verified, require_verified)
    expected = {r.video_id for r in rows}
    keyed = _keyed_predictions(scores, expected)
    if lengths is not None:
        if not isinstance(lengths, Mapping) or set(lengths) != expected:
            raise EvaluationError("invalid_length_inventory")
        limits = [lengths[r.video_id] for r in rows]
    else:
        limits = None
    ids = greedy_decode_batch((keyed[r.video_id] for r in rows),
                              blank_id=renderer.blank_id,
                              num_classes=renderer.num_classes, lengths=limits)
    predictions = [(row.video_id, renderer.render(tokens)) for row, tokens in zip(rows, ids)]
    return _report(rows, predictions, policy, reasons)
