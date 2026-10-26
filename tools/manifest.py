"""데이터 목록(manifests/*.csv)을 만들거나, 지금 디스크의 데이터가 목록과 같은지 검사한다.

원본 영상·좌표·모델은 AI-Hub 이용 조건 때문에 레포에 올리지 않는다. 대신 무엇이 있어야 하는지와 그 해시를 레포에 남겨
팀원이 각자 받은·추출한 데이터가 같은지 확인할 수 있게 한다.

사용: python tools/manifest.py build [--data D:/sign-language-data] [--pre D:/_대학교/연구과제/데이터셋]
      python tools/manifest.py check [--data ...] [--pre ...]
- sources.csv: AI-Hub 원본 zip. 파일당 수십 GB 라 전체 해시(파일당 수 분) 대신 크기 + 앞 1MB 의 SHA-256 을 쓴다.
- features.csv: 추출한 클립 npz 전부. 파일 전체 SHA-256 과 단어 번호·수행자 번호·프레임 수(단어 이름 글자는 넣지 않는다).
- models.csv: 저장한 모델 파일의 SHA-256 과 만든 커밋.
- configs/split-6fold.json 이 dataset.make_folds 결과와 같은지도 검사한다.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "src"))
import dataset as ds  # noqa: E402

MAN = os.path.join(REPO, "manifests")
SPLIT = os.path.join(REPO, "configs", "split-6fold.json")
SIGNERS = list(range(1, 19))

# (AI-Hub 파일 키, 뿌리, 상대 경로, 내용). 키 규칙: 단어 영상 Training 홀수 파일 = 39546 + 2×(수행자−1), 수행자별 단어 1501~3000(T-012).
SOURCES = [("사용자 사전 다운로드", "pre", "수어영상/수어 영상/1.Training/[라벨]01_real_word_morpheme.zip",
            "Training 단어 형태소 라벨(구간·이름), 전체 수행자")]
SOURCES += [(str(39546 + 2 * (s - 1)), "data",
             "downloads/train-word-video-%02d/004.수어영상/1.Training/원천데이터/REAL/WORD/%02d_real_word_video.zip" % (s, 2 * s - 1),
             "Training 단어 영상, 수행자 %d, 단어 1501~3000, 5시점" % s) for s in range(1, 17)]
SOURCES += [("39483", "data", "downloads/val-word-video/004.수어영상/2.Validation/원천데이터/REAL/01_real_word_video.zip",
             "Validation 단어 영상, 수행자 17·18, 전체 단어"),
            ("39478", "data", "downloads/test-39478/004.수어영상/2.Validation/라벨링데이터/REAL/WORD/01_real_word_morpheme.zip",
             "Validation 단어 형태소 라벨")]
FEATURE_GLOBS = ["features/word-*", "features-z/word-*", "features-aspect/*/word-*"]
MODEL_GLOBS = ["models/**/*.pt", "models/*.task"]


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit:
            h.update(f.read(limit))
        else:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def rel(root, path):
    return os.path.relpath(path, root).replace("\\", "/")


def source_rows(roots):
    rows = []
    for key, root, path, what in SOURCES:
        p = os.path.join(roots[root], path)
        ok = os.path.exists(p)
        rows.append({"aihub_key": key, "root": root, "path": path, "contents": what,
                     "bytes": os.path.getsize(p) if ok else "", "sha256_first_1mb": sha256(p, 1 << 20) if ok else ""})
    return rows


def feature_rows(data):
    rows = []
    for g in FEATURE_GLOBS:
        for d in sorted(glob.glob(os.path.join(data, g))):
            for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
                if f.endswith(".tmp.npz"):
                    continue
                z = np.load(f)
                rows.append({"path": rel(data, f), "word_id": int(z["word_id"]), "signer": int(z["signer"]),
                             "frames": len(z["xy"]), "complexity": int(z["complexity"]),
                             "wh": "x".join(str(int(v)) for v in z["wh"]) if "wh" in z.files else "",
                             "has_z": int("z" in z.files), "sha256": sha256(f)})
    return rows


def model_rows(data):
    import torch
    rows = []
    for g in MODEL_GLOBS:
        for f in sorted(glob.glob(os.path.join(data, g), recursive=True)):
            row = {"path": rel(data, f), "bytes": os.path.getsize(f), "sha256": sha256(f), "git": "", "model": "",
                   "n_classes": "", "train_signers": ""}
            if f.endswith(".pt"):
                ck = torch.load(f, weights_only=False)
                row.update(git=ck["git"], model=ck["model"], n_classes=len(ck["classes"]),
                           train_signers=" ".join(map(str, ck["split"]["train"])))
            rows.append(row)
    return rows


def write(name, rows):
    with open(os.path.join(MAN, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("%s: %d행" % (name, len(rows)))


def read(name):
    with open(os.path.join(MAN, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def diff(name, want, have, key):
    """목록(want)과 지금 디스크(have)를 key 로 맞춰 빠진 것·바뀐 것·목록에 없는 것을 센다. 문제 수를 돌려준다."""
    w = {r[key]: r for r in want}
    h = {r[key]: {k: str(v) for k, v in r.items()} for r in have}
    missing = sorted(set(w) - set(h))
    extra = sorted(set(h) - set(w))
    changed = sorted(k for k in set(w) & set(h) if w[k] != h[k])
    print("%s: 목록 %d, 디스크 %d | 없음 %d, 바뀜 %d, 목록에 없음 %d" % (name, len(w), len(h), len(missing), len(changed), len(extra)))
    for label, ks in (("없음", missing), ("바뀜", changed), ("목록에 없음", extra)):
        for k in ks[:10]:
            print("  %s: %s" % (label, k))
        if len(ks) > 10:
            print("  ... 외 %d개" % (len(ks) - 10))
    return len(missing) + len(changed) + len(extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--data", default="D:/sign-language-data")
    ap.add_argument("--pre", default="D:/_대학교/연구과제/데이터셋", help="사용자가 미리 받아 둔 라벨 파일의 뿌리")
    a = ap.parse_args()
    roots = {"data": a.data, "pre": a.pre}
    folds = ds.make_folds(SIGNERS, 6)
    split = {"signers": SIGNERS, "n_folds": 6, "seed": 0, "sha": ds.sha(folds), "folds": folds}
    if a.cmd == "build":
        os.makedirs(MAN, exist_ok=True)
        write("sources.csv", source_rows(roots))
        write("features.csv", feature_rows(a.data))
        write("models.csv", model_rows(a.data))
        with open(SPLIT, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=1)
        print("split-6fold.json: 해시 %s" % split["sha"])
        return
    bad = diff("sources.csv", read("sources.csv"), source_rows(roots), "path")
    bad += diff("features.csv", read("features.csv"), feature_rows(a.data), "path")
    bad += diff("models.csv", read("models.csv"), model_rows(a.data), "path")
    with open(SPLIT, encoding="utf-8") as f:
        same = json.load(f) == json.loads(json.dumps(split))
    print("split-6fold.json:", "make_folds 결과와 같음" if same else "다름(분할 코드나 수행자 목록이 바뀜)")
    bad += 0 if same else 1
    print("결과:", "모두 일치" if not bad else "불일치 %d건" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
