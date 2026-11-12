"""Bounded readers for the observed CROWD per-frame JSON schema.

Source identifiers and coordinates are private. Exceptions contain fixed codes.
Archive payloads are read sequentially, with ZIP CRC checking by zipfile.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import zipfile
import zlib

import numpy as np

VIDEO = re.compile(r"NIA_SL_FS(?P<fs>[0-9]{4})_CROWD(?P<crowd>[0-9]{2})_F\.mp4")
MEMBER = re.compile(
    r"keypoint/(?P<folder>[0-9]{2})/"
    r"(?P<base>NIA_SL_FS[0-9]{4}_CROWD(?P<crowd>[0-9]{2})_F)/"
    r"(?P=base)_(?P<frame>[0-9]{12})_keypoints\.json"
)
GROUPS = (("pose_keypoints_2d", 25), ("face_keypoints_2d", 70),
          ("hand_left_keypoints_2d", 21), ("hand_right_keypoints_2d", 21))


class KeypointError(ValueError):
    """Fixed error code without source values."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KeypointError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise KeypointError("nonfinite")


@dataclass(frozen=True, repr=False)
class KeypointFrame:
    pose: np.ndarray
    face: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    extra_fields: int = 0


def parse_keypoint_json(raw, *, max_bytes=65_536):
    """Parse x,y,confidence triples. Empty groups retain shape (0,3).

    The observed people value is one object. No person tracking or handedness,
    image size, coordinate units or frame time is inferred from this object.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise KeypointError("invalid_limit")
    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise KeypointError("json_size")
    try:
        document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
    except KeypointError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise KeypointError("json_document") from None
    if not isinstance(document, dict) or type(document.get("version")) not in (int, float):
        raise KeypointError("schema")
    if document["version"] != 1.3:
        raise KeypointError("version")
    people = document.get("people")
    if not isinstance(people, dict) or type(people.get("person_id")) is not int:
        raise KeypointError("people")
    arrays = []
    for name, count in GROUPS:
        values = people.get(name)
        if not isinstance(values, list) or len(values) not in (0, count * 3):
            raise KeypointError("joint_count")
        if any(type(value) not in (int, float) for value in values):
            raise KeypointError("coordinate_type")
        try:
            points = np.array(values, dtype=np.float64).reshape(-1, 3)
        except (ValueError, OverflowError):
            raise KeypointError("coordinate_value") from None
        if not np.isfinite(points).all():
            raise KeypointError("nonfinite")
        if ((points[:, 2] < 0) | (points[:, 2] > 1)).any():
            raise KeypointError("confidence")
        points.setflags(write=False)
        arrays.append(points)
    for name, _ in GROUPS:
        values = people.get(name.replace("_2d", "_3d"))
        if values != []:
            raise KeypointError("unsupported_3d")
    expected_people = {"person_id", *(name for name, _ in GROUPS),
                       *(name.replace("_2d", "_3d") for name, _ in GROUPS)}
    extra_fields = len(set(document) - {"version", "people", "camparam"})
    extra_fields += len(set(people) - expected_people)
    return KeypointFrame(*arrays, extra_fields=extra_fields)


@dataclass(frozen=True)
class SequenceIndex:
    video_id: str = field(repr=False)
    entries: tuple = field(repr=False)

    @property
    def frame_indices(self):
        return tuple(index for index, _ in self.entries)

    @property
    def missing_frames(self):
        indices = self.frame_indices
        return indices[-1] + 1 - len(indices)


class CrowdKeypointArchive:
    """Context-managed ZIP index, with bounded payload reads and no extraction.

    Limits are checked before payload reads. Python zipfile loads the central
    directory first, so max_members does not bound that allocation. Limits are
    configurable for trusted datasets. Duplicate frame/member IDs fail closed.
    """

    def __init__(self, path, *, max_members=1_000_000, max_json_bytes=65_536,
                 max_total_bytes=8 * 1024**3, max_ratio=100):
        for value in (max_members, max_json_bytes, max_total_bytes, max_ratio):
            if type(value) is not int or value < 1:
                raise KeypointError("invalid_limit")
        self.path = Path(path)
        self.max_members = max_members
        self.max_json_bytes = max_json_bytes
        self.max_total_bytes = max_total_bytes
        self.max_ratio = max_ratio
        self.archive = None
        self.sequences = ()
        self.uncompressed_bytes = 0

    def __enter__(self):
        try:
            self.archive = zipfile.ZipFile(self.path)
            groups = defaultdict(list)
            total = count = 0
            for info in self.archive.infolist():
                if info.is_dir():
                    continue
                count += 1
                if count > self.max_members:
                    raise KeypointError("member_limit")
                if info.file_size > self.max_json_bytes:
                    raise KeypointError("member_size")
                total += info.file_size
                if total > self.max_total_bytes:
                    raise KeypointError("total_size")
                if info.file_size > max(1, info.compress_size) * self.max_ratio:
                    raise KeypointError("compression_ratio")
                if info.flag_bits & 1:
                    raise KeypointError("encrypted_member")
                match = MEMBER.fullmatch(info.filename)
                if not match or match["folder"] != match["crowd"]:
                    raise KeypointError("member_name")
                groups[match["base"] + ".mp4"].append((int(match["frame"]), info))
            sequences = []
            for video, entries in sorted(groups.items()):
                entries.sort(key=lambda entry: entry[0])
                indices = [index for index, _ in entries]
                if len(set(indices)) != len(indices):
                    raise KeypointError("duplicate_frame")
                sequences.append(SequenceIndex(video, tuple(entries)))
            self.sequences = tuple(sequences)
            self.uncompressed_bytes = total
            return self
        except KeypointError:
            self.close()
            raise
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError):
            self.close()
            raise KeypointError("archive_read") from None

    def read_frames(self, sequence):
        """Yield validated frames and verify each member CRC when fully read."""
        for frame, error in self.iter_frame_results(sequence):
            if error:
                raise KeypointError(error)
            yield frame

    def iter_frame_results(self, sequence):
        """Read every member, yielding (frame, None) or (None, fixed_code)."""
        if self.archive is None:
            raise KeypointError("archive_closed")
        if not any(sequence is item for item in self.sequences):
            raise KeypointError("foreign_sequence")
        for _, info in sequence.entries:
            try:
                with self.archive.open(info) as stream:
                    raw = stream.read(self.max_json_bytes + 1)
                if len(raw) != info.file_size or len(raw) > self.max_json_bytes:
                    raise KeypointError("member_size")
                frame = parse_keypoint_json(raw, max_bytes=self.max_json_bytes)
            except KeypointError as error:
                yield None, str(error)
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError,
                    zlib.error, NotImplementedError):
                yield None, "member_read"
            else:
                yield frame, None

    def close(self):
        if self.archive is not None:
            self.archive.close()
        self.archive = None
        self.sequences = ()

    def __exit__(self, *exc):
        self.close()
