"""구간 탐지기 출력 후처리. 탐지기 모델(B)은 그대로 두고, 낸 구간만 다듬는다(T-068).

① 짧은 조각 합치기: frag_s 보다 짧은 조각을 간격 gap_s 이내의 이웃에 붙인다(한 단어를 여러 조각으로 자르는 과분할).
② 지문자 묶음 판정: 간격 fs_gap_s 이내로 이어진 조각들을 묶고, 묶음 전체가 fs_min_s 이상이며 손가락 움직임 / 손목 움직임이
   fs_ratio 이상이면 지문자 구간 하나로 본다. 탐지기는 조각마다 판정해 지문자 하나(약 3.7초)를 5~6조각으로 잘라 놓쳤다(T-066).
   켜면 탐지기의 조각별 지문자 판정은 쓰지 않는다. 움직임은 탐지기와 같은 FeatureExtractor 값(프레임마다 finger, wrist_speed).
③ 같은 단어 연속 합치기(분류 뒤): 같은 번호가 간격 dedup_gap_s 이내로 이어지면 하나로 센다.
값이 0 이면 그 단계는 끈다(기본값 = 후처리 없음, 예전 결과와 같음).
"""
from dataclasses import dataclass, fields

import numpy as np

import boundary as bd


@dataclass
class PostConfig:
    frag_s: float = 0.0
    gap_s: float = 0.15
    fs_gap_s: float = 0.0
    fs_min_s: float = 0.6
    fs_ratio: float = 2.0
    dedup_gap_s: float = 0.0

    @classmethod
    def parse(cls, spec):
        """'frag_s=0.3,gap_s=0.1' -> PostConfig. 빈 문자열이면 기본값."""
        names = {f.name for f in fields(cls)}
        kw = {}
        for part in filter(None, (spec or "").split(",")):
            k, v = part.split("=")
            if k.strip() not in names:
                raise ValueError("모르는 후처리 값: %s (가능: %s)" % (k, sorted(names)))
            kw[k.strip()] = float(v)
        return cls(**kw)


def motion(t, xy, mask):
    """프레임마다 (손가락 움직임, 손목 속도). 탐지기가 구간 동안 더하는 값과 같다."""
    fe = bd.FeatureExtractor()
    out = np.zeros((len(t), 2))
    for i in range(len(t)):
        fe.step(float(t[i]), xy[i], mask[i])
        out[i] = fe.last["finger"], fe.last["wrist_speed"]
    return out


def merge_fragments(segs, cfg):
    """segs: 시간순 [(시작, 끝, 종류)]. 짧은 조각을 가까운 앞 조각(없거나 멀면 뒤 조각)에 붙인다."""
    if not cfg.frag_s or not segs:
        return list(segs)
    out = [list(segs[0])]
    for s, e, k in segs[1:]:
        p = out[-1]
        if s - p[1] <= cfg.gap_s and (e - s < cfg.frag_s or p[1] - p[0] < cfg.frag_s):
            p[1], p[2] = e, (k if k == p[2] else "word")
        else:
            out.append([s, e, k])
    return [tuple(x) for x in out]


def group_fingerspelling(segs, t, mot, cfg):
    """이어진 조각 묶음을 지문자 하나로 볼지 판정. mot = motion() 결과. 지문자가 아닌 조각은 'word' 로 둔다."""
    if not cfg.fs_gap_s or not segs:
        return list(segs)
    groups, cur = [], [segs[0]]
    for sg in segs[1:]:
        if sg[0] - cur[-1][1] <= cfg.fs_gap_s:
            cur.append(sg)
        else:
            groups.append(cur)
            cur = [sg]
    groups.append(cur)
    out = []
    for g in groups:
        s, e = g[0][0], g[-1][1]
        sel = (t >= s) & (t <= e)
        finger, wrist = mot[sel, 0].sum(), mot[sel, 1].sum()
        if e - s >= cfg.fs_min_s and finger >= cfg.fs_ratio * max(wrist, 1e-6):
            out.append((s, e, "fingerspelling"))
        else:
            out += [(a, b, "word" if k == "fingerspelling" else k) for a, b, k in g]
    return out


def dedup(items, cfg):
    """items: 시간순 [(시작, 끝, 종류, 단어 번호)]. 같은 단어가 바로 이어지면 하나로."""
    if not cfg.dedup_gap_s or not items:
        return list(items)
    out = [list(items[0])]
    for s, e, k, w in items[1:]:
        p = out[-1]
        if k == "word" and p[2] == "word" and w is not None and w == p[3] and s - p[1] <= cfg.dedup_gap_s:
            p[1] = e
        else:
            out.append([s, e, k, w])
    return [tuple(x) for x in out]


def apply(segs, t, xy, mask, cfg, mot=None):
    """분류 전 단계(①②). segs 는 문장 끝을 뺀 [(시작, 끝, 종류)]."""
    segs = merge_fragments(segs, cfg)
    if cfg.fs_gap_s:
        segs = group_fingerspelling(segs, t, motion(t, xy, mask) if mot is None else mot, cfg)
    return segs
