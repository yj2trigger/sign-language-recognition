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

import numpy as np

O, B, I = 0, 1, 2
TYPES = {"word": 1, "fingerspelling": 2}       # seg_type 배열 값(0 = 구간 밖)


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
