"""Read-only inventory, without assuming an AI-Hub keypoint schema.

Categories describe filenames only, never verified content or compatibility.
No source names, labels, URLs or identifiers are returned or printed.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import zipfile


class InventoryError(ValueError):
    """Sanitized inventory failure; no partial result is returned."""


def _category(name):
    name = name.replace("\\", "/").lower()
    suffix = Path(name).suffix
    if suffix == ".json" and ("/morpheme/" in "/" + name or name.endswith("_morpheme.json")):
        return "morpheme_json"
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        return "video_candidate"
    if suffix in {".json", ".npy", ".npz", ".csv", ".pkl", ".pickle"}:
        return "unclassified_data_candidate"
    return "other"


@dataclass(frozen=True)
class InventoryReport:
    files: int
    archives: int
    archive_members: int
    duplicate_members: int
    loose_categories: tuple[tuple[str, int], ...]
    archive_categories: tuple[tuple[str, int], ...]

    def summary(self):
        return {
            "files": self.files, "archives": self.archives,
            "archive_members": self.archive_members,
            "duplicate_members": self.duplicate_members,
            "loose_categories": dict(self.loose_categories),
            "archive_categories": dict(self.archive_categories),
            "keypoint_schema": "not_inspected",
            "ctc_compatibility": "not_established",
        }


def inventory_dataset(root, *, max_files=100_000, max_members=1_000_000):
    """Count loose files and ZIP directory entries without extracting/reading payloads.

    Limits apply across the whole scan. ZIP central directories are loaded by
    zipfile before the member limit can be checked. Symlinks/junctions are
    rejected, including the root. Corrupt payloads/CRC are NOT checked here.
    Duplicate count means extra occurrences of an exact member name per ZIP.
    """
    for limit in (max_files, max_members):
        if type(limit) is not int or limit < 1:
            raise InventoryError("invalid_limit")
    loose, archived = Counter(), Counter()
    files = archives = members = duplicates = 0
    try:
        root = Path(root)
        if root.is_symlink() or root.is_junction() or not root.is_dir():
            raise InventoryError("invalid_root")
        def walk_error(error):
            raise InventoryError("directory_read")
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            for name in dirs + names:
                path = Path(directory) / name
                if path.is_symlink() or path.is_junction():
                    raise InventoryError("linked_path")
            for name in names:
                files += 1
                if files > max_files:
                    raise InventoryError("file_limit")
                path = Path(directory) / name
                if path.suffix.lower() != ".zip":
                    loose[_category(path.relative_to(root).as_posix())] += 1
                    continue
                archives += 1
                with zipfile.ZipFile(path) as archive:
                    counts = Counter()
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        members += 1
                        if members > max_members:
                            raise InventoryError("member_limit")
                        counts[member.filename] += 1
                        archived[_category(member.filename)] += 1
                    duplicates += sum(count - 1 for count in counts.values())
    except InventoryError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError):
        raise InventoryError("inventory_read") from None
    return InventoryReport(files, archives, members, duplicates,
                           tuple(sorted(loose.items())), tuple(sorted(archived.items())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args(argv)
    try:
        report = inventory_dataset(args.root)
    except InventoryError as error:
        print(json.dumps({"error": str(error)}))
        return 2
    print(json.dumps(report.summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
