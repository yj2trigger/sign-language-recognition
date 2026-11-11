"""Read-only diagnostic, NOT official 37-class compatibility validation.

Uses a specified local PR #11 Git object in memory without installing/copying
jamo.py. Only use a reviewed, trusted ref: its Python code will be executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.crowd_labels import read_crowd_zip
from fingerspelling.ctc_targets import TargetEncoder, audit_crowd
from fingerspelling.tokens import TokenVocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--reviewed-pr11-ref", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.reviewed_pr11_ref + "^{commit}"], cwd=root,
    ).decode("ascii").strip()
    source = subprocess.check_output(
        ["git", "show", ref + ":src/fingerspelling/jamo.py"], cwd=root,
    ).decode("utf-8")
    module = types.ModuleType("reviewed_pr11_diagnostic")
    exec(compile(source, "<reviewed-pr11>", "exec"), module.__dict__)
    # Diagnostic Unicode vocabulary only, sorted for deterministic IDs.
    symbols = sorted(module.JAMO)
    vocab = TokenVocabulary({t: i + 1 for i, t in enumerate(symbols)}, blank_id=0)
    before = hashlib.sha256(args.zip_path.read_bytes()).hexdigest()
    report = read_crowd_zip(args.zip_path)
    result = audit_crowd(report, TargetEncoder(vocab, decompose=module.decompose_hangul))
    after = hashlib.sha256(args.zip_path.read_bytes()).hexdigest()
    print(json.dumps({"scope": "diagnostic_unicode_not_official_37",
                      "pr11_ref": ref, "diagnostic_token_count": len(symbols),
                      "blank_id": 0, "zip_unchanged": before == after,
                      "length_validation": "not_performed_no_model_output_lengths",
                      "adapter_summary": report.summary(), "audit": result},
                     ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
