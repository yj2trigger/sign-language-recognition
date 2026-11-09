"""AI-Hub 103 단어 영상 zip 의 정면(F) 영상에서 라벨 구간의 좌표를 뽑아 클립별 npz 로 저장한다.

사용: python src/build_dataset.py <영상zip> <라벨zip> <출력폴더> [프로세스 수=4] [complexity=1] [limit=0] [클래스 json] [자르기 비율]
이미 만든 클립은 건너뛰므로 중단 후 다시 실행해도 이어서 진행한다. limit>0 이면 앞에서 그만큼만 처리한다.
클래스 json(select_classes.py 출력)을 주면 그 단어 번호의 클립만 처리한다.
자르기 비율(예: 9/16)을 주면 프레임 가운데를 그 비율로 잘라 다른 카메라를 흉내 낸다(화면비 검증용, 클래스 json 자리에 - 가능).
npz 내용: xy (T,53,2) float32, mask (T,53) bool, fps, gloss(문자열), word_id, signer, complexity, z (T,53), wh(화면 너비·높이).
--full-fps N 을 붙이면 라벨 구간이 아니라 영상 전체 시간을 N fps 카메라가 찍었을 프레임만 처리하고(위상은 클립 이름으로 고정),
  얼굴 33점(face, face_mask), 프레임 시각 t, 원본 fps(src_fps), 위상(phase), 라벨 시각(seg_start, seg_end), full=True 를 더 저장한다.
  학습 때는 dataset.load_clips 가 라벨 시각으로 잘라 기존 클립과 같은 모양으로 만든다.
출력은 원본에서 만든 가공 데이터이므로 레포에 올리지 않고 D:\\sign-language-data 아래에만 둔다.
"""
import json
import os
import re
import sys
import time
import zipfile
import zlib
from multiprocessing import Pool

import numpy as np

import landmarks
from winpower import keep_awake, no_power_throttling

NAME = re.compile(r"NIA_SL_WORD(\d+)_REAL(\d+)_F$")
_zv = _zl = None
_out = _tmp = ""
_cx = 1
_crop = None
_full_fps = 0.0


def _init(vid, lab, out, cx, crop, full_fps=0.0):
    global _zv, _zl, _out, _tmp, _cx, _crop, _full_fps
    _crop, _full_fps = crop, full_fps
    _zv, _zl, _out, _cx = zipfile.ZipFile(vid), zipfile.ZipFile(lab), out, cx
    _tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "tmp", "build_%d" % os.getpid())
    os.makedirs(_tmp, exist_ok=True)


def _work(name):
    base = os.path.basename(name)[:-4]
    dst = os.path.join(_out, base + ".npz")
    if os.path.exists(dst):
        return base, "skip"
    word, signer = NAME.match(base).groups()
    try:
        raw = _zl.read("morpheme/%s/%s_morpheme.json" % (signer, base))
    except KeyError:                               # 라벨 파일 이름이 틀린 클립(예: WORD1536_REAL10_U_F)은 넘긴다. 예외가 나면 pool 전체가 멈춘다
        return base, "nojson"
    lab = json.loads(raw.decode("utf-8"))
    if not lab["data"]:
        return base, "nolabel"
    seg = lab["data"][0]
    path = _zv.extract(name, _tmp)
    try:
        if _full_fps:
            res = landmarks.extract_video(path, _full_fps, phase_seed=zlib.crc32(base.encode()), complexity=_cx)
        else:
            res = landmarks.extract_clip(path, seg["start"], seg["end"], complexity=_cx, with_z=True, crop_aspect=_crop)
    finally:
        os.remove(path)
    if res is not None and _full_fps:
        part = os.path.join(_out, base + ".tmp.npz")
        np.savez(part, xy=res["xy"], mask=res["mask"], z=res["z"], face=res["face"], face_mask=res["face_mask"],
                 t=res["t"], fps=_full_fps, src_fps=res["src_fps"], phase=res["phase"], full=True,
                 seg_start=seg["start"], seg_end=seg["end"], gloss=seg["attributes"][0]["name"],
                 word_id=int(word), signer=int(signer), complexity=_cx, wh=np.array(res["wh"]))
        os.replace(part, dst)
        return base, "ok"
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
    argv, full_fps = list(sys.argv), 0.0
    if "--full-fps" in argv:
        i = argv.index("--full-fps")
        full_fps = float(argv[i + 1])
        del argv[i:i + 2]
    vid, lab, out = argv[1:4]
    procs = int(argv[4]) if len(argv) > 4 else 4
    cx = int(argv[5]) if len(argv) > 5 else 1
    limit = int(argv[6]) if len(argv) > 6 else 0
    words = None
    if len(argv) > 7 and argv[7] != "-":       # select_classes.py 가 만든 json: 이 단어 번호의 클립만 처리한다
        with open(argv[7], encoding="utf-8") as f:
            words = set(json.load(f)["word_ids"])
    crop = None
    if len(argv) > 8:
        num, den = argv[8].split("/")
        crop = float(num) / float(den)
    if full_fps and crop:
        raise SystemExit("--full-fps 와 자르기 비율은 함께 쓸 수 없다")
    os.makedirs(out, exist_ok=True)
    # 워커 초기화에서 예외가 나면 Pool 이 오류를 보이지 않고 워커를 계속 다시 만든다. 부모에서 먼저 열어 즉시 실패시킨다.
    zipfile.ZipFile(lab).close()
    names = sorted(n for n in zipfile.ZipFile(vid).namelist() if n.endswith("_F.mp4"))
    if words is not None:
        names = [n for n in names if int(NAME.match(os.path.basename(n)[:-4]).group(1)) in words]
    if limit:
        names = names[:limit]
    print("대상 %d개, 프로세스 %d, complexity %d, %s | 효율 모드 제외 %s, 절전 방지 %s" % (
        len(names), procs, cx, "영상 전체 %.0ffps" % full_fps if full_fps else "라벨 구간", no_power_throttling(), keep_awake()), flush=True)
    counts, t0 = {}, time.time()
    with Pool(procs, _init, (vid, lab, out, cx, crop, full_fps)) as pool:
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
