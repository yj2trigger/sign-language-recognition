"""문장 영상의 라벨 시각으로 프레임별 B/I/O(Begin/Inside/Outside) 정답을 만든다.

라벨 구간 = [(시작 초, 끝 초, 종류)], 종류는 'word' 또는 'fingerspelling'. 구간마다 첫 프레임이 B, 나머지가 I, 구간 밖은 O.
입력 세 가지:
  1. AI-Hub 문장 라벨(morpheme json: data[].start/end) -> segments_from_morpheme
  2. build_dataset.py --full-fps 15 로 뽑은 단어 영상 전체 npz(seg_start/seg_end 1개) -> segments_from_npz
     단어 영상만 있으므로, 같은 수행자의 단어 영상 여러 개를 이어 붙여 '의사 문장'을 만든다(concat_sentence).
  3. 실제 문장 영상에서 뽑은 npz(seg = 형태소 라벨 시각) -> load_real_sentences
  4. 데이터 없이 동작을 확인하는 합성 문장(synthetic_sentence, 테스트·스모크 실행용)
문장 원문·단어명(gloss)은 읽지도 저장하지도 않는다. 출력 npz 는 원본 가공물이라 레포 밖(D:\\sign-language-data)에만 쓴다.

사용: python src/dataset_bio.py <full-fps npz 폴더>... --out <레포 밖 폴더> [--words 2,5] [--n 1000] [--seed 0]
"""
import argparse
import glob
import os
import re

import numpy as np

O, B, I = 0, 1, 2
TYPES = {"word": 1, "fingerspelling": 2}       # seg_type 배열 값(0 = 구간 밖)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bio_labels(t, segments, b_frames=1):
    """프레임 시각 t 와 라벨 구간으로 B/I/O 를 만든다.

    Args:
        t (array-like): (T,) 프레임 시각(초), 오름차순.
        segments (list[tuple[float, float, str]]): (시작, 끝, 종류). 종류는 TYPES 의 키.
        b_frames (int): 구간 앞에서 B 로 둘 프레임 수(1 이상).

    Returns:
        tuple[np.ndarray, np.ndarray]: bio (T,) int8 (O=0, B=1, I=2), seg_type (T,) int8 (0, 1=word, 2=fingerspelling).
        구간이 두 프레임 사이에 끼어 걸리는 프레임이 없으면 가운데에 가장 가까운 프레임 하나를 B 로 둔다(짧은 단어를 잃지 않게).

    Raises:
        ValueError: 끝이 시작보다 앞서거나, 종류가 TYPES 에 없거나, b_frames < 1 일 때.

    Examples:
        >>> bio_labels([0.0, 0.1, 0.2, 0.3, 0.4], [(0.1, 0.3, "word")])[0].tolist()
        [0, 1, 2, 2, 0]
    """
    if b_frames < 1:
        raise ValueError("b_frames 는 1 이상이어야 한다")
    t = np.asarray(t, float)
    bio, typ = np.zeros(len(t), np.int8), np.zeros(len(t), np.int8)
    for start, end, kind in sorted(segments):
        if end < start:
            raise ValueError("구간 끝(%.3f)이 시작(%.3f)보다 앞선다" % (end, start))
        if kind not in TYPES:
            raise ValueError("모르는 구간 종류: %r" % (kind,))
        idx = np.where((t >= start) & (t <= end))[0]
        if len(idx) == 0:
            if len(t) == 0:
                continue
            idx = np.array([int(np.argmin(np.abs(t - (start + end) / 2)))])
        bio[idx] = I
        bio[idx[:b_frames]] = B
        typ[idx] = TYPES[kind]
    return bio, typ


def segments_from_morpheme(lab, is_fingerspelling=None):
    """AI-Hub morpheme json(딕셔너리)의 data[] 에서 라벨 구간을 뽑는다. 단어명은 is_fingerspelling 판정에만 넘기고 버린다.

    Args:
        lab (dict): json.load 한 morpheme 라벨. data[i] 에 start, end(초)가 있다.
        is_fingerspelling (callable | None): data[i] 를 받아 지문자면 True. None 이면 모두 'word'.

    Returns:
        list[tuple[float, float, str]]: 시작 순으로 정렬한 (시작, 끝, 종류).
    """
    segs = []
    for item in lab.get("data") or []:
        kind = "fingerspelling" if is_fingerspelling and is_fingerspelling(item) else "word"
        segs.append((float(item["start"]), float(item["end"]), kind))
    return sorted(segs)


def segments_from_npz(z):
    """build_dataset.py --full-fps 결과 npz 의 라벨 구간(단어 1개). 라벨 구간만 뽑은 예전 npz 는 영상 전체 시각이 없어 쓸 수 없다."""
    if "full" not in z.files:
        raise ValueError("영상 전체 npz(--full-fps)가 아니다")
    return [(float(z["seg_start"]), float(z["seg_end"]), "word")]


def load_full_clips(feature_dirs, signers=None):
    """영상 전체 npz 를 읽는다. gloss(단어명)는 읽지 않는다. 반환: [{t, xy, mask, face, face_mask, seg_start, seg_end, signer, word}]."""
    keep = None if signers is None else set(signers)
    clips = []
    for d in feature_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            if "full" not in z.files or (keep is not None and int(z["signer"]) not in keep):
                continue
            clips.append({k: z[k].copy() for k in ("t", "xy", "mask", "face", "face_mask")} |
                         {"seg_start": float(z["seg_start"]), "seg_end": float(z["seg_end"]),
                          "signer": int(z["signer"]), "word": int(z["word_id"])})
    return clips


def load_real_sentences(feature_dirs, signers=None):
    """실제 문장 영상에서 뽑은 npz(t, xy, mask, face, face_mask, seg (N,2) 시각만)를 읽는다. 단어명은 파일에 없다.
    수행자 번호는 파일 이름의 REALxx 에서 읽는다. 반환: concat_sentence 와 같은 딕셔너리 목록(+ name)."""
    keep = None if signers is None else set(signers)
    out = []
    for d in feature_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            name = os.path.basename(f)[:-4]
            m = re.search(r"REAL(\d+)", name)
            if m is None or (keep is not None and int(m.group(1)) not in keep):
                continue
            z = np.load(f)
            if "seg" not in z.files or not len(z["seg"]):
                continue
            s = {k: z[k].copy() for k in ("t", "xy", "mask", "face", "face_mask")}
            s["segments"] = [(float(a), float(b), "word") for a, b in z["seg"]]
            s["signer"], s["name"] = int(m.group(1)), name
            out.append(s)
    return out


# 실제 문장 라벨(AI-Hub Validation 문장 형태소, 정면 4,000개, 수행자 17·18)의 p5~p95 범위(초).
# 첫 단어 앞 여유 / 단어 사이 간격 / 마지막 단어 뒤 여유. 0.8초 이상인 단어 사이 간격은 없었다(문장 끝 멈춤과 겹치지 않음).
SEN_LEAD, SEN_GAP, SEN_TAIL = (1.1, 2.6), (0.07, 0.38), (0.94, 1.94)
# 단어 재생 속도 배율. 단어 영상의 단어 길이 중앙값 1.4초를 문장 속 0.7초 근처로 줄이는 범위(1배도 남겨 둔다).
SEN_SPEED = (1.0, 2.2)


def _take(c, times):
    """클립에서 times(초)에 가장 가까운 프레임들을 뽑는다(빨리 재생하기 위한 다시 뽑기)."""
    j = np.abs(c["t"][None, :] - np.asarray(times)[:, None]).argmin(1)
    return {k: c[k][j] for k in ("xy", "mask", "face", "face_mask")}


def _bridge(a, b, n, rng, jitter=0.003):
    """단어 a 의 마지막 프레임에서 단어 b 의 첫 프레임까지 n 프레임을 직선으로 잇는다(단어 사이 이동).
    두 프레임 모두에서 보인 점만 잇고, 나머지는 앞 절반은 a, 뒤 절반은 b 를 따른다. 얼굴도 절반씩 나눈다.
    직선 이동이 너무 매끈한 단서가 되지 않게 MediaPipe 흔들림만큼 작은 잡음을 더한다."""
    w = (np.arange(1, n + 1) / (n + 1))[:, None, None]
    both = a["mask"] & b["mask"]
    half = np.arange(n) < n / 2
    pick = lambda k: np.where(half.reshape((-1,) + (1,) * a[k].ndim), a[k][None], b[k][None])
    out = {k: pick(k) for k in ("xy", "mask", "face", "face_mask")}
    line = (1 - w) * a["xy"][None] + w * b["xy"][None] + rng.normal(0, jitter, (n,) + a["xy"].shape)
    out["xy"] = np.where(both[None, :, None], line, out["xy"]).astype(a["xy"].dtype)
    return out


def concat_sentence(clips, rng, lead=SEN_LEAD, gap=SEN_GAP, tail=SEN_TAIL, speed=SEN_SPEED, fps=15.0):
    """단어 영상 여러 개를 이어 붙여 의사 문장 하나를 만든다.

    - 첫 단어 앞(lead)·마지막 단어 뒤(tail)는 원본 클립의 쉬는 자세·손 올림/내림 프레임을 그대로 쓴다.
      클립에 그만큼의 여유가 없으면 있는 만큼만 쓴다.
    - 단어 사이(gap)는 원본의 손 내림·올림 대신 앞 단어 끝 자세에서 다음 단어 첫 자세까지 직선으로 잇는다.
      따로 찍은 영상을 그냥 붙이면 이음매에서 손이 순간 이동해, 모델이 실제 경계 대신 그것을 배웠다(뉴스 영상에서 확인).
    - 단어(라벨 구간)는 speed 배로 빨리 재생한다. 단어 영상의 단어가 문장 속 단어보다 길어서다(중앙값 1.4초 대 0.7초).
    반환: {t, xy, mask, face, face_mask, segments, signer}. t 는 0 부터 1/fps 간격."""
    parts, segs, n, prev = [], [], 0, None
    valid = [c for c in clips if np.any((c["t"] >= c["seg_start"]) & (c["t"] <= c["seg_end"]))]
    if not valid:
        raise ValueError("이어 붙일 프레임이 없다")
    for i, c in enumerate(valid):
        if i == 0:                                           # 첫 단어 앞: 원본 그대로
            pre = c["t"][(c["t"] >= c["seg_start"] - rng.uniform(*lead)) & (c["t"] < c["seg_start"])]
            if len(pre):
                parts.append(_take(c, pre)); n += len(pre)
        sp = rng.uniform(*speed)
        m = max(1, int(round((c["seg_end"] - c["seg_start"]) / sp * fps)) + 1)
        core = _take(c, np.minimum(c["seg_start"] + np.arange(m) * sp / fps, c["seg_end"]))
        if prev is not None:                                 # 단어 사이: 직선 이동
            g = max(0, int(round(rng.uniform(*gap) * fps)) - 1)   # 끝 프레임~다음 시작 프레임 간격이 gap 이 되게
            if g:
                last = {k: v[-1] for k, v in prev.items()}
                first = {k: v[0] for k, v in core.items()}
                parts.append(_bridge(last, first, g, rng)); n += g
        segs.append((n / fps, (n + m - 1) / fps, "word"))
        parts.append(core); n += m
        prev = core
    c = valid[-1]                                            # 마지막 단어 뒤: 원본 그대로
    post = c["t"][(c["t"] > c["seg_end"]) & (c["t"] <= c["seg_end"] + rng.uniform(*tail))]
    if len(post):
        parts.append(_take(c, post)); n += len(post)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["t"] = np.arange(n) / fps
    out["segments"] = segs
    out["signer"] = int(valid[0]["signer"])
    return out


def make_sentences(clips, n, words=(2, 5), seed=0, fps=15.0):
    """수행자마다 단어 영상을 무작위로 골라 의사 문장 n 개를 만든다(한 문장 = 한 수행자)."""
    rng = np.random.default_rng(seed)
    by = {}
    for c in clips:
        by.setdefault(c["signer"], []).append(c)
    signers = sorted(by)
    out = []
    for i in range(n):
        pool = by[signers[i % len(signers)]]
        k = int(rng.integers(words[0], words[1] + 1))
        pick = [pool[j] for j in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
        out.append(concat_sentence(pick, rng, fps=fps))
    return out


# ---- 합성 문장: 실제 데이터 없이 라벨러·탐지기·학습 루프가 도는지 확인한다 ----

def _hand(center, spread, wiggle):
    """손 21점: 손목 center, 손가락 끝을 spread 만큼 위로 펼치고 wiggle 로 흔든다."""
    pts = np.tile(center, (21, 1)).astype(np.float32)
    for f, tip in enumerate((4, 8, 12, 16, 20)):
        for j, k in enumerate(range(tip - 3, tip + 1)):
            pts[k] = center + np.array([(f - 2) * 0.01, -spread * (j + 1) / 4]) + wiggle[f] * (j + 1) / 4
    return pts


def synthetic_sentence(rng, n_words=3, fps=15.0, fingerspell_prob=0.0, end_face=True, signer=0):
    """손 올림·움직임으로 단어를, 손 내림 또는 손 멈춤으로 단어 사이를 흉내 낸 합성 문장.

    좌표 단위는 MediaPipe 정규화(0~1). 어깨너비 0.2, 쉬는 손은 화면 아래(손 마스크 꺼짐).
    지문자 구간은 손목을 고정하고 손가락만 빠르게 움직인다. end_face 이면 마지막 단어 끝에서 눈썹을 올린다.
    반환: concat_sentence 와 같은 딕셔너리.
    """
    frames, segs = [], []

    def add(kind, dur, word=None):
        for k in range(max(1, int(round(dur * fps)))):
            frames.append((kind, k / fps, word))

    add("rest", rng.uniform(0.6, 1.0))
    kinds = []
    for w in range(n_words):
        kinds.append("fingerspelling" if rng.random() < fingerspell_prob else "word")
        start = len(frames)
        dur = rng.uniform(0.6, 1.0) if w < n_words - 1 else rng.uniform(0.9, 1.3)   # 마지막 단어는 길게(문장 끝 늘임)
        add(kinds[-1], dur + (0.4 if kinds[-1] == "fingerspelling" else 0.0), w)    # 지문자는 글자 여러 개라 더 길다
        segs.append((start, len(frames) - 1, kinds[-1]))
        if w < n_words - 1:
            add("hold" if rng.random() < 0.5 else "rest", rng.uniform(0.35, 0.6))
    add("rest", rng.uniform(1.8, 2.4))

    T = len(frames)
    xy, mask = np.zeros((T, 53, 2), np.float32), np.zeros((T, 53), bool)
    face, face_mask = np.zeros((T, 33, 2), np.float32), np.ones((T, 33), bool)
    phase = rng.uniform(0, 2 * np.pi, 2)
    last_up = np.array([[0.62, 0.5], [0.38, 0.5]], np.float32)
    end_t = segs[-1][1] / fps
    for i, (kind, tau, word) in enumerate(frames):
        xy[i, 0] = (0.5, 0.2)
        xy[i, 5], xy[i, 6] = (0.6, 0.4), (0.4, 0.4)
        mask[i, [0, 5, 6, 9, 10]] = True
        if kind == "rest":
            wrists = np.array([[0.62, 0.95], [0.38, 0.95]], np.float32)
        elif kind == "hold":
            wrists = last_up
        else:
            amp = 0.004 if kind == "fingerspelling" else 0.05
            base = np.array([[0.62, 0.5 + 0.03 * (word % 2)], [0.38, 0.5]], np.float32)
            wrists = base + amp * np.stack([np.sin(2 * np.pi * 1.5 * tau + phase), np.cos(2 * np.pi * 1.5 * tau + phase)], 1)
            last_up = wrists.astype(np.float32)
        xy[i, 9], xy[i, 10] = wrists
        if kind != "rest":
            fast = 0.015 if kind == "fingerspelling" else 0.002
            for h, start in enumerate((11, 32)):
                wig = fast * np.stack([np.sin(2 * np.pi * 4 * tau + np.arange(5) + h),
                                       np.cos(2 * np.pi * 4 * tau + np.arange(5))], 1) if kind != "hold" else np.zeros((5, 2))
                xy[i, start:start + 21] = _hand(wrists[h], 0.06, wig)
                mask[i, start:start + 21] = True
        brow = 0.02 if end_face and end_t - 0.4 <= i / fps else 0.0
        face[i, :10, 1] = 0.12 - brow            # 눈썹(위로 갈수록 y 가 작다)
        face[i, 10:18, 1] = 0.15                  # 눈
        face[i, 18:30, 1] = 0.25                  # 입
        face[i, 26, 1], face[i, 27, 1] = 0.248, 0.252
        face[i, 30, 1], face[i, 31, 1], face[i, 32, 1] = 0.2, 0.3, 0.05
        face[i, :, 0] = 0.5
    t = np.arange(T) / fps
    return {"t": t, "xy": xy, "mask": mask, "face": face, "face_mask": face_mask,
            "segments": [(t[a], t[b], k) for a, b, k in segs], "signer": signer}


def _inside_repo(path):
    p = os.path.abspath(path)
    try:
        return os.path.commonpath([p, REPO]) == REPO
    except ValueError:            # 드라이브가 다르면(D: 등) 레포 밖이다
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="+", help="build_dataset.py --full-fps 결과 폴더")
    ap.add_argument("--out", required=True, help="B/I/O npz 를 쓸 폴더(레포 밖)")
    ap.add_argument("--n", type=int, default=1000, help="만들 의사 문장 수")
    ap.add_argument("--words", default="2,5", help="문장당 단어 수 범위")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if _inside_repo(a.out):
        raise SystemExit("원본 가공물이라 레포 안(%s)에는 쓰지 않는다. D:\\sign-language-data 아래를 지정할 것" % REPO)
    lo, hi = (int(x) for x in a.words.split(","))
    clips = load_full_clips(a.features)
    if not clips:
        raise SystemExit("영상 전체 npz(--full-fps)가 없다")
    os.makedirs(a.out, exist_ok=True)
    for i, s in enumerate(make_sentences(clips, a.n, (lo, hi), a.seed)):
        bio, typ = bio_labels(s["t"], s["segments"])
        np.savez(os.path.join(a.out, "sent%05d.npz" % i), t=s["t"], xy=s["xy"], mask=s["mask"], face=s["face"],
                 face_mask=s["face_mask"], bio=bio, seg_type=typ, signer=s["signer"],
                 segments=np.array([(a0, b0, TYPES[k]) for a0, b0, k in s["segments"]], np.float32))
    print("의사 문장 %d개 -> %s (단어 영상 %d개에서)" % (a.n, a.out, len(clips)))


if __name__ == "__main__":
    main()
