"""Read-only CROWD adapter. Never extracts ZIP members or emits source values.

Records and review IDs are private local data. Only summary() is suitable for
public reporting. This maps supplied labels, not verified video transcriptions.
"""
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
import re
from urllib.parse import unquote, urlsplit
import zipfile
import zlib

from label_preparation import (
    LabelError, Target, decode_label_document, target_from_morphemes, validate_targets,
)

MEMBER = re.compile(
    r"morpheme/(?P<folder>[0-9]{2})/"
    r"(?P<base>NIA_SL_FS(?P<fs>[0-9]{4})_CROWD(?P<crowd>[0-9]{2})_F)_morpheme\.json"
)
VIDEO = re.compile(r"NIA_SL_FS[0-9]{4}_CROWD[0-9]{2}_F\.mp4")


class CrowdLabelError(LabelError):
    """A fixed error code only, without filenames, URLs or source text."""


@dataclass(frozen=True)
class CrowdLabel:
    fs_id: str = field(repr=False)
    crowd_id: str = field(repr=False)
    target: Target = field(repr=False)


@dataclass(frozen=True)
class CrowdReport:
    records: tuple[CrowdLabel, ...] = field(repr=False)
    review_fs_ids: tuple[str, ...] = field(repr=False)
    json_count: int
    errors: tuple[tuple[str, int], ...]

    def summary(self) -> dict[str, int]:
        """Return aggregate counts only; suffix count is a heuristic, not a class."""
        texts = [r.target.text for r in self.records]
        failures = dict(self.errors)
        return {
            "json_count": self.json_count,
            "mapped_count": len(self.records),
            "error_count": sum(failures.values()),
            "filename_mismatch_count": sum(failures.get(k, 0) for k in (
                "member_name", "folder_id", "metadata_name", "url_name")),
            "fs_count": len({r.fs_id for r in self.records}),
            "crowd_count": len({r.crowd_id for r in self.records}),
            "conflict_group_count": len(self.review_fs_ids),
            "hangul_count": sum(bool(re.search(r"[\uac00-\ud7a3]", t)) for t in texts),
            "digit_count": sum(bool(re.search(r"[0-9]", t)) for t in texts),
            "road_suffix_candidate_count": sum(t.endswith(("로", "길")) for t in texts),
        }


def expected_video_name(member_name: str) -> str:
    """Validate a ZIP member path and return its expected video basename."""
    if not isinstance(member_name, str):
        raise CrowdLabelError("member_name")
    match = MEMBER.fullmatch(member_name)
    if not match:
        raise CrowdLabelError("member_name")
    if match["folder"] != match["crowd"]:
        raise CrowdLabelError("folder_id")
    return match["base"] + ".mp4"


def parse_crowd_label(member_name: str, raw: bytes) -> CrowdLabel:
    """Validate name/URL association and one morpheme; preserve original text.

    CROWD ID is a grouping key from filenames, not a verified person identity.
    """
    expected = expected_video_name(member_name)
    match = MEMBER.fullmatch(member_name)
    try:
        document = decode_label_document(raw)
    except LabelError:
        raise CrowdLabelError("json_document") from None
    metadata = document.get("metaData")
    if not isinstance(metadata, dict):
        raise CrowdLabelError("metadata")
    if metadata.get("name") != expected:
        raise CrowdLabelError("metadata_name")
    url = metadata.get("url")
    if not isinstance(url, str) or not url or any(ord(c) < 33 for c in url):
        raise CrowdLabelError("metadata_url")
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError
        basename = unquote(parsed.path.rsplit("/", 1)[-1], errors="strict")
    except (ValueError, UnicodeError):
        raise CrowdLabelError("metadata_url") from None
    if basename != expected:
        raise CrowdLabelError("url_name")
    try:
        target = target_from_morphemes(
            document, video_id=expected, signer_id="CROWD" + match["crowd"],
        )
    except LabelError:
        raise CrowdLabelError("morpheme") from None
    return CrowdLabel(match["fs"], match["crowd"], target)


def find_conflicts(records: Iterable[CrowdLabel]) -> tuple[str, ...]:
    """Return private review FS IDs with unequal exact text; never normalize."""
    groups = defaultdict(set)
    for record in records:
        groups[record.fs_id].add(record.target.text)
    return tuple(sorted(fs for fs, texts in groups.items() if len(texts) > 1))


def read_crowd_zip(path, *, max_json_bytes: int = 1_048_576,
                   max_members: int = 100_000, max_total_bytes: int = 128_000_000) -> CrowdReport:
    """Read members sequentially, returning mappings and fixed-code error counts.

    File-level errors are counted and processing continues. Duplicate member
    names are all excluded (no first/last wins). Archive/limit failures raise a
    sanitized exception and return no partial report. Limits bound decompressed
    input; central-directory loading is still handled by Python zipfile.
    """
    for limit in (max_json_bytes, max_members, max_total_bytes):
        if type(limit) is not int or limit <= 0:
            raise CrowdLabelError("invalid_limit")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if len(members) > max_members:
                raise CrowdLabelError("member_limit")
            files = [m for m in members if not m.is_dir()]
            counts = Counter(m.filename for m in files)
            jsons = [m for m in files if m.filename.lower().endswith(".json")]
            if sum(m.file_size for m in jsons) > max_total_bytes:
                raise CrowdLabelError("total_size_limit")
            records, errors = [], Counter()
            for member in jsons:
                if counts[member.filename] > 1:
                    errors["duplicate_json"] += 1
                    continue
                try:
                    expected_video_name(member.filename)
                    if member.file_size > max_json_bytes:
                        raise CrowdLabelError("json_size_limit")
                    with archive.open(member) as stream:
                        raw = stream.read(max_json_bytes + 1)
                    if len(raw) > max_json_bytes:
                        raise CrowdLabelError("json_size_limit")
                    records.append(parse_crowd_label(member.filename, raw))
                except CrowdLabelError as exc:
                    errors[str(exc)] += 1
                except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError, EOFError, zlib.error):
                    errors["member_read"] += 1
            validate_targets(r.target for r in records)
            return CrowdReport(tuple(records), find_conflicts(records), len(jsons), tuple(sorted(errors.items())))
    except CrowdLabelError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError, LabelError):
        raise CrowdLabelError("archive_read") from None


@dataclass(frozen=True)
class VideoCheck:
    """Aggregate comparison of supplied basenames, not filesystem/video checks."""
    status: str
    expected_count: int
    supplied_count: int = 0
    matched_count: int = 0
    missing_count: int = 0
    unexpected_count: int = 0
    duplicate_count: int = 0


def compare_video_names(report: CrowdReport, video_names: Iterable[str] | None = None,
                        *, max_names: int = 100_000) -> VideoCheck:
    """Compare exact video basenames; None means not provided, [] means empty.

    Duplicate count counts extra occurrences, unexpected counts distinct names.
    No supplied list is presented as proof of file existence or content.
    """
    expected = {r.target.video_id for r in report.records}
    if video_names is None:
        return VideoCheck("not_provided", len(expected))
    if type(max_names) is not int or max_names <= 0 or isinstance(video_names, (str, bytes)):
        raise CrowdLabelError("video_inventory")
    try:
        iterator = iter(video_names)
    except TypeError:
        raise CrowdLabelError("video_inventory") from None
    counts = Counter()
    for i, name in enumerate(iterator):
        if i >= max_names:
            raise CrowdLabelError("video_inventory_limit")
        if not isinstance(name, str) or not VIDEO.fullmatch(name):
            raise CrowdLabelError("video_name")
        counts[name] += 1
    supplied = set(counts)
    missing, unexpected = len(expected - supplied), len(supplied - expected)
    duplicates = sum(n - 1 for n in counts.values())
    status = "matched" if not (missing or unexpected or duplicates or report.errors) else "mismatch"
    return VideoCheck(status, len(expected), sum(counts.values()), len(expected & supplied),
                      missing, unexpected, duplicates)
