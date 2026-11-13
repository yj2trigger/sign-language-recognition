"""Print anonymous source diagnostics; never write training arrays or labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_keypoints import KeypointError
from fingerspelling.crowd_source_validation import audit_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    args = parser.parse_args(argv)
    try:
        report = audit_source(args.archive)
    except (KeypointError, OSError):
        print(json.dumps({"error": "source_audit_failed"}))
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["source_hash_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
