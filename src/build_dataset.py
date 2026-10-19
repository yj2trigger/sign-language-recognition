"""AI-Hub 103 단어 영상 zip 의 정면(F) 영상에서 라벨 구간의 좌표를 뽑아 클립별 npz 로 저장한다.

사용: python src/build_dataset.py <영상zip> <라벨zip> <출력폴더> [프로세스 수=4] [complexity=1] [limit=0] [클래스 json]
이미 만든 클립은 건너뛰므로 중단 후 다시 실행해도 이어서 진행한다. limit>0 이면 앞에서 그만큼만 처리한다.
클래스 json(select_classes.py 출력)을 주면 그 단어 번호의 클립만 처리한다.
npz 내용: xy (T,53,2) float32, mask (T,53) bool, fps, gloss(문자열), word_id, signer, complexity, z (T,53), wh(화면 너비·높이).
출력은 원본에서 만든 가공 데이터이므로 레포에 올리지 않고 D:\\sign-language-data 아래에만 둔다.
"""
import json
import os
import re
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np

import landmarks

NAME = re.compile(r"NIA_SL_WORD(\d+)_REAL(\d+)_F$")
_zv = _zl = None
_out = _tmp = ""
_cx = 1


def _init(vid, lab, out, cx):
    global _zv, _zl, _out, _tmp, _cx
    _zv, _zl, _out, _cx = zipfile.ZipFile(vid), zipfile.ZipFile(lab), out, cx
    _tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "tmp", "build_%d" % os.getpid())
    os.makedirs(_tmp, exist_ok=True)


def _work(name):
    base = os.path.basename(name)[:-4]
    dst = os.path.join(_out, base + ".npz")
    if os.path.exists(dst):
        return base, "skip"
    word, signer = NAME.match(base).groups()
    lab = json.loads(_zl.read("morpheme/%s/%s_morpheme.json" % (signer, base)).decode("utf-8"))
    if not lab["data"]:
        return base, "nolabel"
    seg = lab["data"][0]
    path = _zv.extract(name, _tmp)
    try:
        res = landmarks.extract_clip(path, seg["start"], seg["end"], complexity=_cx, with_z=True)
    finally:
        os.remove(path)
    if res is None:
        return base, "empty"
    xy, mask, fps, z, wh = res
    part = os.path.join(_out, base + ".tmp.npz")   # 쓰는 도중 중단돼도 불완전한 파일이 남지 않게
    np.savez(part, xy=xy, mask=mask, fps=fps, gloss=seg["attributes"][0]["name"],
             word_id=int(word), signer=int(signer), complexity=_cx,
             z=z, wh=np.array(wh))                  # z: MediaPipe 깊이 값, wh: 화면 (너비, 높이) — 화면비 보정용(C-16)
    os.replace(part, dst)
    return base, "ok"


def main():
    vid, lab, out = sys.argv[1:4]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    cx = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    words = None
    if len(sys.argv) > 7:       # select_classes.py 가 만든 json: 이 단어 번호의 클립만 처리한다
        with open(sys.argv[7], encoding="utf-8") as f:
            words = set(json.load(f)["word_ids"])
    os.makedirs(out, exist_ok=True)
    # 워커 초기화에서 예외가 나면 Pool 이 오류를 보이지 않고 워커를 계속 다시 만든다. 부모에서 먼저 열어 즉시 실패시킨다.
    zipfile.ZipFile(lab).close()
    names = sorted(n for n in zipfile.ZipFile(vid).namelist() if n.endswith("_F.mp4"))
    if words is not None:
        names = [n for n in names if int(NAME.match(os.path.basename(n)[:-4]).group(1)) in words]
    if limit:
        names = names[:limit]
    print("대상 %d개, 프로세스 %d, complexity %d" % (len(names), procs, cx), flush=True)
    counts, t0 = {}, time.time()
    with Pool(procs, _init, (vid, lab, out, cx)) as pool:
        for i, (base, status) in enumerate(pool.imap_unordered(_work, names, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "skip"):
                print("주의:", base, status, flush=True)
            if i % 25 == 0 or i == len(names):
                el = time.time() - t0
                print("%d/%d %s 경과 %.0f초 남은 예상 %.0f초" % (i, len(names), counts, el, el / i * (len(names) - i)), flush=True)
    print("완료", counts, flush=True)


if __name__ == "__main__":
    main()
