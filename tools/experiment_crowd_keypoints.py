"""Read-only geometry experiment using the pinned PR #19 parser.

Run from any directory. Only aggregate measurements are emitted. No source
specification, transformed landmarks or frame timestamps are inferred.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f7631c32b734abc1027329b8d29a9d520fc82987"
EDGES = np.array([-np.inf, 0, 1, 256, 512, 768, 1024, 1280, 1920, 2560, 4096, np.inf])


def load_source():
    """Reuse reviewed code directly from Git without changing the checkout."""
    name = "_crowd_experiment_source"
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    for leaf in ("crowd_keypoints", "crowd_keypoint_adapter"):
        path = f"src/fingerspelling/{leaf}.py"
        code = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT)
        module = types.ModuleType(f"{name}.{leaf}")
        module.__package__ = name
        sys.modules[module.__name__] = module
        exec(compile(code, f"{SOURCE_COMMIT}:{path}", "exec"), module.__dict__)
    return sys.modules[f"{name}.crowd_keypoints"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Coordinates:
    """All finite coordinates and separate positive-confidence joint moments."""

    def __init__(self, joints):
        self.minimum = np.full(2, np.inf)
        self.maximum = np.full(2, -np.inf)
        self.histogram = np.zeros((2, len(EDGES) - 1), np.int64)
        self.count = np.zeros(joints, np.int64)
        self.total = np.zeros((joints, 2))
        self.low = np.full((joints, 2), np.inf)
        self.high = np.full((joints, 2), -np.inf)
        self.confidence_min = np.inf
        self.confidence_max = -np.inf
        self.invalid_confidence = 0

    def add(self, points):
        if not len(points):
            return
        xy = points[:, :2]
        self.minimum = np.minimum(self.minimum, xy.min(0))
        self.maximum = np.maximum(self.maximum, xy.max(0))
        for axis in range(2):
            self.histogram[axis] += np.histogram(xy[:, axis], EDGES)[0]
        observed = points[:, 2] > 0
        self.count += observed
        self.total += np.where(observed[:, None], xy, 0)
        self.low = np.minimum(self.low, np.where(observed[:, None], xy, np.inf))
        self.high = np.maximum(self.high, np.where(observed[:, None], xy, -np.inf))
        self.confidence_min = min(self.confidence_min, float(points[:, 2].min()))
        self.confidence_max = max(self.confidence_max, float(points[:, 2].max()))
        self.invalid_confidence += int(((points[:, 2] < 0) | (points[:, 2] > 1)).sum())

    def result(self):
        def finite(values):
            return [float(v) if np.isfinite(v) else None for v in values]
        return dict(min_xy=finite(self.minimum), max_xy=finite(self.maximum),
                    histogram_xy=self.histogram.tolist(),
                    confidence_range=finite([self.confidence_min, self.confidence_max]),
                    invalid_confidence_values=self.invalid_confidence,
                    positive_confidence_joints=[
                        dict(count=int(n), mean_xy=finite(self.total[i] / n) if n else [None, None],
                             min_xy=finite(self.low[i]), max_xy=finite(self.high[i]))
                        for i, n in enumerate(self.count)])


def experiment(path):
    source = load_source()
    before = digest(path)
    stats = {key: Coordinates(n) for key, n in source.GROUPS}
    errors = Counter()
    lengths = []
    strict_sequences = gaps = total = starts_zero = 0
    top_keys, people_keys, camera_types = Counter(), Counter(), Counter()
    wrist_distances = [[], []]
    with source.CrowdKeypointArchive(path) as archive:
        for sequence in archive.sequences:
            lengths.append(len(sequence.entries))
            starts_zero += sequence.frame_indices[0] == 0
            gaps += sequence.missing_frames
            clean = True
            for _, member in sequence.entries:
                total += 1
                with archive.archive.open(member) as stream:
                    raw = stream.read(archive.max_json_bytes + 1)
                try:
                    frame = source._parse_keypoint_json(raw, check_confidence=False)
                except source.KeypointError as exc:
                    errors[str(exc)] += 1
                    clean = False
                    continue
                document = json.loads(raw.decode("utf-8-sig"))
                top_keys.update(document.keys())
                people_keys.update(document["people"].keys())
                camera = document.get("camparam")
                camera_types[f"{type(camera).__name__}:empty={not bool(camera)}"] += 1
                points = (frame.pose, frame.face, frame.left_hand, frame.right_hand)
                invalid = any(((p[:, 2] < 0) | (p[:, 2] > 1)).any() for p in points)
                if invalid:
                    errors["confidence"] += 1
                    clean = False
                for (key, _), p in zip(source.GROUPS, points):
                    stats[key].add(p)
                # Distances only describe field agreement, never anatomical truth.
                for slot, hand in enumerate((frame.left_hand, frame.right_hand)):
                    if len(frame.pose) and len(hand) and hand[0, 2] > 0:
                        wrists = frame.pose[[7, 4]]
                        if (wrists[:, 2] > 0).all():
                            wrist_distances[slot].append(np.linalg.norm(wrists[:, :2] - hand[0, :2], axis=1))
            strict_sequences += clean
            if len(lengths) % 100 == 0:
                print(f"groups={len(lengths)} frames={total}", file=sys.stderr, flush=True)
    after = digest(path)
    if before != after:
        raise RuntimeError("source_changed")
    return dict(source_commit=SOURCE_COMMIT, sha256_before=before, sha256_after=after,
                source_bytes=Path(path).stat().st_size, frames=total, groups=len(lengths),
                errors=dict(errors), strict_groups=strict_sequences, missing_indices=gaps,
                groups_starting_zero=int(starts_zero),
                frame_count_percentiles=np.percentile(lengths, [0, 25, 50, 75, 100]).tolist() if lengths else [],
                frame_count_histogram=dict(Counter(lengths)),
                histogram_edges=["-inf", *EDGES[1:-1].tolist(), "+inf"],
                coordinates={key: value.result() for key, value in stats.items()},
                top_keys=dict(top_keys), people_keys=dict(people_keys), camparam_types=dict(camera_types),
                wrist_field_distance=[dict(count=len(values), median_to_pose7_pose4=
                                          np.median(values, axis=0).tolist() if values else [])
                                      for values in wrist_distances],
                converted_ctc_inputs=0, coordinate_system="unverified", fps="unverified",
                anatomical_handedness="unverified")


def label_metadata(path, keypoint_path):
    """Inspect metadata fields and exact video basenames, without target parsing."""
    source = load_source()
    before = digest(path)
    fields, errors = Counter(), Counter()
    ids, durations = [], []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if member.file_size > 65_536:
                raise ValueError("label_metadata_size")
            with archive.open(member) as stream:
                document = json.loads(stream.read(65_537).decode("utf-8-sig"))
            metadata = document.get("metaData", {})
            fields.update(metadata.keys())
            name = metadata.get("name")
            if not isinstance(name, str) or not source.VIDEO.fullmatch(name):
                errors["metadata_name"] += 1
            else:
                ids.append(name)
            duration = metadata.get("duration")
            try:
                valid_duration = type(duration) in (int, float) and math.isfinite(duration) and duration > 0
            except OverflowError:
                valid_duration = False
            if valid_duration:
                durations.append(duration)
            else:
                errors["duration"] += 1
    with source.CrowdKeypointArchive(keypoint_path) as archive:
        keypoint_ids = {sequence.video_id for sequence in archive.sequences}
    after = digest(path)
    if before != after:
        raise RuntimeError("source_changed")
    return dict(sha256_before=before, sha256_after=after,
                source_bytes=Path(path).stat().st_size, metadata_fields=dict(fields), errors=dict(errors),
                records=len(ids), duplicate_names=len(ids) - len(set(ids)),
                exact_keypoint_matches=len(set(ids) & keypoint_ids),
                label_without_keypoint=len(set(ids) - keypoint_ids),
                keypoint_without_label=len(keypoint_ids - set(ids)),
                duration_range_seconds=[min(durations), max(durations)] if durations else [],
                fps="unverified", target_semantics="not_evaluated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keypoint_zip", type=Path)
    parser.add_argument("--morpheme-zip", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only and args.morpheme_zip is None:
        parser.error("--metadata-only requires --morpheme-zip")
    result = {} if args.metadata_only else experiment(args.keypoint_zip)
    if args.morpheme_zip:
        result["morpheme_metadata"] = label_metadata(args.morpheme_zip, args.keypoint_zip)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
