"""Read-only source diagnostics. These aggregates never authorize conversion.

Confidence outside [0,1] is retained for measurement only. The strict training
parser and adapter continue to reject it, including errors in unused face slots.
"""
from collections import Counter
import hashlib
import zipfile
import zlib

from .crowd_keypoints import (
    CrowdKeypointArchive, GROUPS, KeypointError, _parse_keypoint_json,
)


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ConfidenceDiagnostics:
    """Aggregate finite, structurally validated frames without exposing payloads."""

    def __init__(self):
        self.groups = {}
        for name, count in GROUPS:
            self.groups[name] = dict(joints=0, empty_frames=0, negative=0, above_one=0,
                                     zero=0, below_half=0, invalid_frames=0,
                                     minimum=None, maximum=None,
                                     invalid_by_joint=[0] * count,
                                     coordinate_min=None, coordinate_max=None)

    def inspect(self, raw):
        # All shape/type/finite/version/3D/duplicate-key checks are shared with
        # the strict parser. The only bypass is the measured confidence range.
        frame = _parse_keypoint_json(raw, check_confidence=False)
        invalid = False
        for (name, _), points in zip(GROUPS, (
                frame.pose, frame.face, frame.left_hand, frame.right_hand)):
            item = self.groups[name]
            if not len(points):
                item["empty_frames"] += 1
                continue
            confidence = points[:, 2]
            bad = (confidence < 0) | (confidence > 1)
            invalid |= bool(bad.any())
            item["joints"] += len(points)
            for key, values in (("negative", confidence < 0),
                                ("above_one", confidence > 1),
                                ("zero", confidence == 0),
                                ("below_half", confidence < .5)):
                item[key] += int(values.sum())
            item["invalid_frames"] += int(bad.any())
            item["invalid_by_joint"] = [a + int(b) for a, b in
                                         zip(item["invalid_by_joint"], bad)]
            for key, value, operation in (
                    ("minimum", float(confidence.min()), min),
                    ("maximum", float(confidence.max()), max),
                    ("coordinate_min", float(points[:, :2].min()), min),
                    ("coordinate_max", float(points[:, :2].max()), max)):
                item[key] = value if item[key] is None else operation(item[key], value)
        return "confidence" if invalid else None


def audit_source(path, *, expected_video_ids=None):
    """Full CRC scan, confidence distribution and anonymous sequence histogram.

    No source spec is fabricated. Valid means strict schema/range only, never
    verified anatomy, geometry, temporal correctness or training suitability.
    expected_video_ids must come from the existing PR #12 label parser.
    """
    before = _digest(path)
    diagnostics = ConfidenceDiagnostics()
    errors, lengths, invalid_per_sequence = Counter(), Counter(), Counter()
    position_totals = [0] * 10
    position_invalid = [0] * 10
    strict_sequences = missing_indices = total = strict_frames = 0
    with CrowdKeypointArchive(path) as archive:
        ids = {sequence.video_id for sequence in archive.sequences}
        for sequence in archive.sequences:
            lengths[len(sequence.entries)] += 1
            missing_indices += sequence.missing_frames
            sequence_errors = invalid_frames = 0
            for position, (_, member) in enumerate(sequence.entries):
                position_bin = min(9, position * 10 // len(sequence.entries))
                position_totals[position_bin] += 1
                total += 1
                try:
                    with archive.archive.open(member) as stream:
                        raw = stream.read(archive.max_json_bytes + 1)
                    if len(raw) != member.file_size:
                        raise KeypointError("member_size")
                    error = diagnostics.inspect(raw)
                except KeypointError as exc:
                    error = str(exc)
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile,
                        EOFError, zlib.error, NotImplementedError):
                    error = "member_read"
                if error:
                    errors[error] += 1
                    sequence_errors += 1
                    invalid_frames += int(error == "confidence")
                    position_invalid[position_bin] += int(error == "confidence")
                else:
                    strict_frames += 1
            strict_sequences += int(sequence_errors == 0)
            invalid_per_sequence[invalid_frames] += 1
    report = dict(
        frames=total, strict_frames=strict_frames, errors=dict(errors),
        sequences=len(ids), strict_sequences=strict_sequences,
        missing_frame_indices=missing_indices,
        frame_count_histogram=dict(sorted(lengths.items())),
        confidence_invalid_frames_per_sequence=dict(sorted(invalid_per_sequence.items())),
        relative_frame_deciles=dict(total=position_totals, confidence_invalid=position_invalid),
        confidence_groups=diagnostics.groups,
        source_sha256=before, source_hash_unchanged=before == _digest(path),
        trustworthy_geometry_sequences=0, verified_conversion_success=0,
        conversion_blocked_sequences=len(ids), ctc_usable_samples=0,
        conversion_attempted_sequences=0, conversion_failed_sequences=0,
        conversion_failure_reasons={"source_spec_unverified": len(ids),
                                    "frame_time_unverified": len(ids)},
        missing_joint_ratio=None, fps15_verified=False,
        confidence_root_cause="unresolved", coordinate_system="unverified",
        anatomical_handedness="unverified", per_video_timing="unverified",
    )
    if expected_video_ids is not None:
        expected = set(expected_video_ids)
        report.update(label_video_matches=len(ids & expected),
                      keypoint_without_label=len(ids - expected),
                      label_without_keypoint=len(expected - ids))
    return report
