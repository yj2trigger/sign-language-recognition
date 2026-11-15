"""AI-Hub 문장 영상(정면 _F.mp4)에서 53점·얼굴 33점을 15fps 로 뽑아, 형태소 라벨의 단어 구간 시각과 함께 npz 로 저장한다.

결과는 원본 가공물이라 레포 밖에만 쓴다. 단어명·문장 원문은 저장하지 않는다(segments_from_morpheme 이 시각·종류만 넘긴다).
이미 있는 npz 는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 한다.

사용: python src/extract_sentences.py "<[원천]NN_real_sen_video.zip>" "<[라벨]01_real_sen_morpheme.zip>" --out <레포 밖 폴더> [--procs 6]
결과: <out>/video_F/*.mp4 (zip 에서 꺼낸 정면 영상), <out>/features/NIA_SL_SEN0001_REAL01_F.npz
      npz 키: t (T,), xy (T,53,2), mask (T,53), face (T,33,2), face_mask (T,33), seg (N,2) 단어 시작·끝 초, duration
이 결과를 train_boundary.py --real-train / --real-test, eval_boundary.py 에 넣는다.
"""
import argparse
import functools
import json
import os
import zipfile
import zlib
from multiprocessing import Pool

import numpy as np

import dataset_bio as db


def label_name(base):
    """NIA_SL_SEN0001_REAL01_F -> 라벨 zip 안의 morpheme/01/NIA_SL_SEN0001_REAL01_F_morpheme.json"""
    return "morpheme/%s/%s_morpheme.json" % (base.split("REAL")[1][:2], base)


def sentence_npz(res, lab):
    """extract_video 결과와 morpheme json 으로 저장할 배열을 만든다. 단어명은 쓰지 않는다."""
    segs = db.segments_from_morpheme(lab)
    return dict(t=res["t"], xy=res["xy"], mask=res["mask"], face=res["face"], face_mask=res["face_mask"],
                seg=np.array([(s, e) for s, e, _ in segs], np.float32).reshape(-1, 2),
                duration=float(lab["metaData"]["duration"]))


def work(path, label_zip, fdir):
    import landmarks                                          # 작업 프로세스마다 MediaPipe 를 따로 연다
    base = os.path.basename(path)[:-4]
    dst = os.path.join(fdir, base + ".npz")
    if os.path.exists(dst):
        return base, "skip"
    res = landmarks.extract_video(path, 15.0, phase_seed=zlib.crc32(base.encode()))
    if res is None:
        return base, "fail"
    with zipfile.ZipFile(label_zip) as z:
        lab = json.loads(z.read(label_name(base)))
    np.savez(dst + ".tmp.npz", **sentence_npz(res, lab))
    os.replace(dst + ".tmp.npz", dst)
    return base, "ok"


def unpack_front(video_zip, vdir):
    """zip 에서 정면 영상(_F.mp4)만 꺼낸다. 크기가 같은 파일은 다시 꺼내지 않는다."""
    with zipfile.ZipFile(video_zip) as z:
        names = [n for n in z.namelist() if n.endswith("_F.mp4")]
        for n in names:
            dst = os.path.join(vdir, os.path.basename(n))
            if not os.path.exists(dst) or os.path.getsize(dst) != z.getinfo(n).file_size:
                with z.open(n) as src, open(dst, "wb") as f:
                    while chunk := src.read(1 << 22):
                        f.write(chunk)
    return sorted(os.path.join(vdir, os.path.basename(n)) for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_zip", help="[원천]NN_real_sen_video.zip")
    ap.add_argument("label_zip", help="[라벨]01_real_sen_morpheme.zip")
    ap.add_argument("--out", required=True, help="결과 폴더(레포 밖)")
    ap.add_argument("--procs", type=int, default=6, help="MediaPipe 작업 프로세스 수")
    a = ap.parse_args()
    if db._inside_repo(a.out):
        raise SystemExit("원본 가공물이라 레포 안(%s)에는 쓰지 않는다" % db.REPO)
    vdir, fdir = os.path.join(a.out, "video_F"), os.path.join(a.out, "features")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    paths = unpack_front(a.video_zip, vdir)
    print("정면 영상", len(paths), flush=True)
    stat = {}
    with Pool(a.procs) as pool:
        for i, (_, s) in enumerate(pool.imap_unordered(functools.partial(work, label_zip=a.label_zip, fdir=fdir), paths)):
            stat[s] = stat.get(s, 0) + 1
            if i % 50 == 0:
                print(i, stat, flush=True)
    print("끝", stat, flush=True)


if __name__ == "__main__":
    main()
