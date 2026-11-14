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



def prepare(args):
    import cv2
    import mediapipe
    import landmarks
    out = args.output
    config = source_config(args)
    state_path = out / 'progress.json'
    if state_path.exists():
        state = read(state_path)
        if state['config'] != config:
            raise ValueError('configuration_changed_use_new_output')
    else:
        rows, rejected, nvideos, nlabels = candidates(args.videos, args.labels)
        # Sample FS groups first, keeping paired CROWD groups together.
        rng = random.Random(args.seed)
        groups = sorted({r['fs_id'] for r in rows})
        rng.shuffle(groups)
        rank = {g: i for i, g in enumerate(groups)}
        rows.sort(key=lambda r: (rank[r['fs_id']], r['video_id']))
        chosen = rows[:args.max_videos]
        state = dict(config=config, zip_inventory_complete=True, whole_zip_crc_checked=False,
            candidate_videos=nvideos, label_jsons=nlabels, eligible_pairs=len(rows), exclusions=rejected,
            selected=chosen, completed={}, failed={}, stages=dict(zip_inventory=True, matching=True))
        write(state_path, state)
    started = time.perf_counter()
    temp_dir = out / 'temp'
    temp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.videos) as archive:
        for row in state['selected']:
            vid = row['video_id']
            path = out / 'features' / (vid + '.npz')
            if vid in state['completed']:
                if feature_ok(path, row) != state['completed'][vid]['sha256']:
                    raise ValueError('completed_feature_changed')
                old_temp=out/'temp'/(vid+'.mp4')
                owner=old_temp.with_suffix('.owner.json')
                if old_temp.exists() and owner.exists() and read(owner)==dict(member=row['member'],crc=row['crc'],bytes=row['bytes']):
                    checked_temporary(old_temp,row)
                    old_temp.unlink()
                    owner.unlink()
                continue
            if time.perf_counter() - started > args.extraction_seconds:
                state['budget_stop'] = True
                break
            tick = time.perf_counter()
            temporary = temp_dir / (vid + '.mp4')
            created = False
            try:
                if shutil.disk_usage(out).free < row['bytes'] + 256 * 1024**2:
                    raise OSError('insufficient_disk_space')
                owner=temporary.with_suffix('.owner.json')
                ownership=dict(member=row['member'],crc=row['crc'],bytes=row['bytes'])
                if temporary.exists():
                    if not owner.exists() or read(owner)!=ownership:
                        raise FileExistsError('unowned_temporary_video_preserved')
                    try:
                        checked_temporary(temporary,row)
                    except ValueError:
                        # Preserve the owned partial copy, then retry the member.
                        partial=temporary.with_name(temporary.name+'.partial-'+str(time.time_ns()))
                        temporary.rename(partial)
                    else:
                        created=True
                if not temporary.exists():
                    write(owner,ownership)
                    with archive.open(row['member']) as source, temporary.open('xb') as destination:
                        created = True
                        shutil.copyfileobj(source, destination, 1024**2)
                # ZipExtFile verifies this member CRC upon complete read.
                probe=probe_video(temporary)
                fps,count=probe['src_fps'],probe['source_frames']
                width,height=probe['resolution']
                if abs(count / fps - row['label_duration']) > max(.2, 2 / fps):
                    raise ValueError('video_label_duration_mismatch')
                reused=False
                previous=out.parent/'validated-run-01'
                prior_state_path=previous/'progress.json'
                if prior_state_path.exists() and probe['cfr_error_seconds']<1e-5:
                    prior=read(prior_state_path)
                    old=prior['completed'].get(vid)
                    old_path=previous/'features'/(vid+'.npz')
                    if (old and prior['config']['schema']==feature_schema()
                            and prior['config']['landmarks_sha256']==config['landmarks_sha256']
                            and prior['config']['video_size']==config['video_size']
                            and prior['config']['video_mtime']==config['video_mtime']
                            and prior['config']['mediapipe']==config['mediapipe']
                            and prior['config']['opencv']==config['opencv']
                            and feature_ok(old_path,row)==old['sha256']):
                        expected=probe['times'][probe['indices']]
                        expected=expected[(expected>=row['start'])&(expected<row['end'])]
                        with np.load(old_path,allow_pickle=False) as cached:
                            if np.allclose(cached['t'],expected,atol=1e-6) if len(cached['t'])==len(expected) else False:
                                clip=dict(xy=cached['xy'].copy(),mask=cached['mask'].copy(),t=expected,
                                          wh=old['fitted_wh'],src_fps=fps)
                                reused=True
                if not reused:
                    clip=extract_timestamped(temporary,probe)
                keep = (clip['t'] >= row['start']) & (clip['t'] < row['end'])
                xy, mask, times = clip['xy'][keep], clip['mask'][keep], clip['t'][keep]
                values = features(xy, mask)
                target = TargetEncoder(vocabulary()).encode(row['text'], input_length=len(values))
                if path.exists():
                    # Orphan file after interrupted state write: verify before adopting.
                    feature_ok(path, row)
                    with np.load(path, allow_pickle=False) as old:
                        if not all(np.array_equal(old[key],value) for key,value in
                            [('features',values),('xy',xy),('mask',mask),('t',times),('target',target.ids)]):
                            raise ValueError('orphan_feature_differs')
                else:
                    pending = path.with_suffix('.npz.tmp')
                    with pending.open('wb') as stream:
                        np.savez_compressed(stream, xy=xy, mask=mask, features=values,
                                            target=np.asarray(target.ids, np.int64), t=times)
                    os.replace(pending, path)
                sha = feature_ok(path, row)
                state['completed'][vid] = dict(sha256=sha, frames=len(values), source_fps=fps,
                    source_frames=count, resolution=[width, height], extracted_fps=15,
                    missing_rate=float(1-mask.mean()), hands_observed_rate=float(mask[:,11:].mean()),
                    seconds=time.perf_counter()-tick, member_crc_checked=True, path=str(path),
                    mediapipe=mediapipe.__version__, model_complexity=0, no_mirror=True,
                    reused_verified_cfr_features=reused,cfr_clock_error_seconds=probe['cfr_error_seconds'],
                    fitted_wh=clip['wh'], source_rotation_policy='OpenCV_auto_orientation_default',
                    interval_max_error_seconds=float(np.max(np.abs(np.diff(times)-1/15))),
                    start_time=float(times[0]), end_time=float(times[-1]))
                state['failed'].pop(vid, None)
                write(state_path, state)
                if created:
                    temporary.unlink()  # Only this invocation's verified, recorded MP4.
                    owner.unlink()
                print(json.dumps(dict(completed=len(state['completed']), video_id=vid,
                    frames=len(values), seconds=time.perf_counter()-tick)), flush=True)
            except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                state['failed'][vid] = dict(error=type(exc).__name__, reason=str(exc),
                    temporary_video_preserved=str(temporary) if created else None)
                write(state_path, state)
                print('failed', vid, repr(exc), flush=True)
    state['stages']['extraction'] = len(state['completed']) == len(state['selected'])
    state['stages']['features'] = bool(state['completed'])
    write(state_path, state)
    return state



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
