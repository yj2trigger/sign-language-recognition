"""Print aggregate CROWD keypoint validation, without extracting private data.

Label providers must reuse the PR #12 parser and supply validated exact video
IDs. No label parser is embedded here. A source spec is required for conversion.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_dataset_validation import InventoryError, inventory_dataset
from fingerspelling.crowd_keypoints import CrowdKeypointArchive, KeypointError
from fingerspelling.crowd_keypoint_adapter import conversion_blockers


def file_digest(path):
    try:
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").digest()
    except OSError:
        raise KeypointError("archive_read") from None


def audit_archive(path, *, expected_video_ids=None):
    """Full sequential JSON/CRC scan, returning aggregates only.

    No metadata is inferred for conversion. File-level problems are counted and
    scanning continues. A failed frame excludes its sequence from valid counts.
    No derived coordinates, source IDs or labels are written to disk.
    """
    before = file_digest(path)
    errors = Counter()
    counts = Counter(json_total=0, json_valid=0, frames_with_extra_fields=0)
    lengths = Counter()
    group_shapes = Counter()
    valid_sequences = 0
    missing_frames = 0
    confidence_zero = 0
    total_joints = 0
    coordinate_min = float("inf")
    coordinate_max = float("-inf")
    with CrowdKeypointArchive(path) as archive:
        ids = {sequence.video_id for sequence in archive.sequences}
        for sequence in archive.sequences:
            valid = True
            lengths[len(sequence.entries)] += 1
            missing_frames += sequence.missing_frames
            for frame, error in archive.iter_frame_results(sequence):
                counts["json_total"] += 1
                if error:
                    errors[error] += 1
                    valid = False
                    continue
                counts["json_valid"] += 1
                counts["frames_with_extra_fields"] += int(frame.extra_fields > 0)
                groups = (frame.pose, frame.face, frame.left_hand, frame.right_hand)
                group_shapes[tuple(len(group) for group in groups)] += 1
                for group in groups:
                    if not len(group):
                        continue
                    total_joints += len(group)
                    confidence_zero += int(np.count_nonzero(group[:, 2] == 0))
                    observed = group[group[:, 2] > 0, :2]
                    if observed.size:
                        coordinate_min = min(coordinate_min, float(observed.min()))
                        coordinate_max = max(coordinate_max, float(observed.max()))
            valid_sequences += int(valid)
        report = {
            **counts, "errors": dict(sorted(errors.items())), "sequences": len(ids),
            "valid_sequences": valid_sequences,
            "uncompressed_bytes": archive.uncompressed_bytes,
            "frame_count_min": min(lengths, default=0),
            "frame_count_max": max(lengths, default=0),
            "frame_count_histogram": dict(sorted(lengths.items())),
            "missing_frame_indices": missing_frames,
            "group_shape_counts": {str(key): value for key, value in group_shapes.items()},
            "confidence_zero_joints": confidence_zero, "total_joints": total_joints,
            "confidence_zero_ratio": confidence_zero / total_joints if total_joints else None,
            "observed_coordinate_min": coordinate_min if np.isfinite(coordinate_min) else None,
            "observed_coordinate_max": coordinate_max if np.isfinite(coordinate_max) else None,
            "conversion_blockers": list(conversion_blockers(None)),
            "verified_conversion_success": 0, "conversion_blocked_sequences": len(ids),
            "official_performer_split": "unverified",
        }
        if expected_video_ids is not None:
            expected = set(expected_video_ids)
            report.update(label_video_matches=len(ids & expected),
                          keypoint_without_label=len(ids - expected),
                          label_without_keypoint=len(expected - ids))
    report["source_hash_unchanged"] = before == file_digest(path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--inventory-root")
    args = parser.parse_args(argv)
    try:
        if args.inventory_root:
            print(json.dumps({"inventory": inventory_dataset(args.inventory_root).summary()}), flush=True)
        print(json.dumps(audit_archive(args.archive), sort_keys=True, allow_nan=False))
    except (KeypointError, InventoryError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
