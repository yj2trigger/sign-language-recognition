"""Resumable, one-video-at-a-time MediaPipe CTC experiment (no ZIP expansion)."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import unicodedata
import zipfile
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fingerspelling_runtime import features, feature_schema, vocabulary, render, tokens_for, FingerspellingReader
from fingerspelling.ctc_targets import TargetEncoder
from fingerspelling.ctc_model import FingerspellingCTC, CTCModelConfig
from fingerspelling.ctc_training import CTCTrainer, CTCTrainingConfig, seed_everything
from fingerspelling.ctc_checkpoint import load_checkpoint
from fingerspelling.cer_metrics import edit_counts
from fingerspelling.ctc_decode import greedy_decode
from fingerspelling_video import probe_video, extract_timestamped


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()



def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))



def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)



def candidates(video_zip, label_zip):
    """Match by exact video ID; reject ambiguous labels without changing them."""
    encoder = TargetEncoder(vocabulary())
    rows, rejected = [], Counter()
    with zipfile.ZipFile(label_zip) as labels, zipfile.ZipFile(video_zip) as videos:
        mapping = {}
        for member in labels.infolist():
            if not member.filename.endswith('.json'):
                continue
            raw = json.loads(labels.read(member).decode('utf-8-sig'))
            vid = Path(raw['metaData']['name']).stem
            if vid in mapping:
                raise ValueError('duplicate_label_video_id')
            mapping[vid] = raw
        seen = set()
        for member in videos.infolist():
            if not member.filename.lower().endswith('.mp4'):
                continue
            original_stem = Path(member.filename).stem
            match = re.fullmatch(r'(NIA_SL_FS[0-9]{4}_CROWD[0-9]{2}_F)(_?)', original_stem)
            if not match:
                rejected['unsupported_video_filename'] += 1
                continue
            vid = match[1]
            if vid in seen:
                raise ValueError('duplicate_mp4_video_id')
            seen.add(vid)
            try:
                raw = mapping[vid]
                if len(raw['data']) != 1 or len(raw['data'][0]['attributes']) != 1:
                    raise ValueError('multiple_segments_or_attributes')
                segment = raw['data'][0]
                text = segment['attributes'][0]['name']
                target = encoder.encode(text)
                start, end = float(segment['start']), float(segment['end'])
                duration = float(raw['metaData']['duration'])
                if not np.isfinite([start, end, duration]).all() or not 0 <= start < end <= duration + .1:
                    raise ValueError('label_time_range')
                rows.append(dict(video_id=vid, member=member.filename, bytes=member.file_size,
                    filename_alias=original_stem != vid, association='unique_FS_CROWD_F_filename',
                    crc=member.CRC, text=text, target_ids=list(target.ids), start=start, end=end,
                    label_duration=duration, fs_id=vid.split('_')[2], crowd_group=vid.split('_')[3],
                    semantic_status='unverified', performer_id=None))
            except (ValueError, KeyError, TypeError) as exc:
                rejected['missing_label' if isinstance(exc, KeyError) else str(exc)] += 1
    return rows, dict(rejected), len(seen), len(mapping)



def feature_ok(path, row):
    with np.load(path, allow_pickle=False) as data:
        xy, mask, values, target, times = (data[k] for k in ('xy', 'mask', 'features', 'target', 't'))
        if (xy.shape != (len(values), 53, 2) or mask.shape != xy.shape[:2]
                or mask.dtype != np.bool_ or values.dtype != np.float32
                or values.shape != (len(times), 424) or len(times) < 2
                or not np.isfinite(values).all() or not np.isfinite(times).all()
                or not (np.diff(times) > 0).all() or not np.array_equal(target, row['target_ids'])
                or not np.array_equal(features(xy, mask), values)):
            raise ValueError('feature_roundtrip_mismatch')
        TargetEncoder(vocabulary()).encode(row['text'], input_length=len(values))
    return digest(path)



def source_config(args):
    import cv2
    import mediapipe
    return dict(schema=feature_schema(), runner_sha256=digest(__file__), seed=args.seed,
                  max_videos=args.max_videos, video_path=str(args.videos),
                  video_size=args.videos.stat().st_size, video_mtime=args.videos.stat().st_mtime_ns,
                  labels_sha256=digest(args.labels), landmarks_sha256=digest(ROOT/'src/landmarks.py'),
                  video_extractor_sha256=digest(ROOT/'src/fingerspelling_video.py'),
                  mediapipe=mediapipe.__version__, opencv=cv2.__version__,
                  extraction=dict(complexity=0, clock='decoded_PTS', target_fps=15, mirror=False))



def checked_temporary(path, row):
    if path.stat().st_size != row['bytes']:
        raise ValueError('temporary_size_mismatch')
    crc=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            crc=zlib.crc32(chunk,crc)
    if crc != row['crc']:
        raise ValueError('temporary_crc_mismatch')



def split_rows(rows, seed):
    """FS + identical-text connected groups; not verified performer independent."""
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in (('fs', row['fs_id']), ('text', unicodedata.normalize('NFC',row['text']))):
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    groups = list(groups.values())
    if len(groups) < 5:
        raise ValueError('insufficient_independent_label_groups')
    random.Random(seed).shuffle(groups)
    n = max(1, len(groups)//5)
    return {name: [row for group in subset for row in group] for name, subset in
            (('validation', groups[:n]), ('test', groups[n:2*n]), ('train', groups[2*n:]))}
