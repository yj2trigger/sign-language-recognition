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
    # Windows readers/indexers can briefly omit FILE_SHARE_DELETE.
    # Do not change permissions or delete the old journal to work around it.
    for attempt in range(8):
        try:
            os.replace(temporary,path)
            break
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.025*2**attempt,.25))


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
    result = dict(schema=feature_schema(), runner_sha256=digest(__file__), seed=args.seed,
                  max_videos=args.max_videos, video_path=str(args.videos),
                  video_size=args.videos.stat().st_size, video_mtime=args.videos.stat().st_mtime_ns,
                  labels_sha256=digest(args.labels), landmarks_sha256=digest(ROOT/'src/landmarks.py'),
                  video_extractor_sha256=digest(ROOT/'src/fingerspelling_video.py'),
                  mediapipe=mediapipe.__version__, opencv=cv2.__version__,
                  extraction=dict(complexity=0, clock='decoded_PTS', target_fps=15, mirror=False))
    for name in ('selection_file','reuse_features','temp_root'):
        value=getattr(args,name,None)
        if value is not None:result[name]=str(Path(value).resolve())
    if getattr(args,'selection_file',None):result['selection_sha256']=digest(args.selection_file)
    result['minimum_quality']={name:getattr(args,name,0.) for name in ('min_hand_coverage','min_shoulder_coverage')}
    return result


def quality_check(mask,args):
    if mask.ndim!=2 or mask.shape[1]!=53 or len(mask)<2 or mask.dtype!=np.bool_:
        raise ValueError('invalid_quality_mask')
    hands=float(mask[:,11:].mean())
    shoulders=float((mask[:,5]&mask[:,6]).mean())
    if hands<getattr(args,'min_hand_coverage',0.):raise ValueError('hand_coverage_below_minimum')
    if shoulders<getattr(args,'min_shoulder_coverage',0.):raise ValueError('shoulder_coverage_below_minimum')
    return hands,shoulders


def reuse_feature(args,row,destination,config):
    """Copy a verified PTS feature; never re-extract or modify its source."""
    previous=getattr(args,'reuse_features',None)
    if previous is None:return None
    previous=Path(previous);state=read(previous/'progress.json');vid=row['video_id']
    old=state['completed'].get(vid)
    if old is None:return None
    for key in ('schema','video_size','video_mtime','labels_sha256','landmarks_sha256',
                'video_extractor_sha256','mediapipe','opencv','extraction'):
        if state['config'].get(key)!=config.get(key):raise ValueError('reuse_source_contract_changed')
    source_row=next(r for r in state['selected'] if r['video_id']==vid)
    for key in ('target_ids','text','start','end','crc','member'):
        if source_row[key]!=row[key]:raise ValueError('reuse_label_or_member_changed')
    if not old.get('member_crc_checked'):raise ValueError('reuse_member_not_verified')
    source=previous/'features'/(vid+'.npz')
    if feature_ok(source,row)!=old['sha256']:raise ValueError('reuse_feature_changed')
    with np.load(source,allow_pickle=False) as data:quality_check(data['mask'],args)
    if not destination.exists():
        temporary=destination.with_suffix('.npz.copy.tmp')
        with source.open('rb') as stream,temporary.open('wb') as target:shutil.copyfileobj(stream,target)
        if feature_ok(temporary,row)!=old['sha256']:raise ValueError('reuse_copy_mismatch')
        os.replace(temporary,destination)
    elif feature_ok(destination,row)!=old['sha256']:raise ValueError('reuse_destination_changed')
    return dict(old,path=str(destination),seconds=0.,reused_verified_features=True,reused_from=str(source))


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
        if getattr(args,'selection_file',None):
            selection=read(args.selection_file)['video_ids'];lookup={r['video_id']:r for r in rows}
            if len(selection)!=len(set(selection)) or len(selection)>args.max_videos or any(v not in lookup for v in selection):
                raise ValueError('invalid_explicit_selection')
            chosen=[lookup[v] for v in selection]
        state = dict(config=config, zip_inventory_complete=True, whole_zip_crc_checked=False,
            candidate_videos=nvideos, label_jsons=nlabels, eligible_pairs=len(rows), exclusions=rejected,
            selected=chosen, completed={}, failed={}, stages=dict(zip_inventory=True, matching=True))
        write(state_path, state)
    started = time.perf_counter()
    temp_dir = Path(getattr(args,'temp_root',None) or out/'temp')
    temp_dir.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.videos) as archive:
        for row in state['selected']:
            vid = row['video_id']
            path = out / 'features' / (vid + '.npz')
            if vid in state['completed']:
                if feature_ok(path, row) != state['completed'][vid]['sha256']:
                    raise ValueError('completed_feature_changed')
                old_temp=temp_dir/(vid+'.mp4')
                owner=old_temp.with_suffix('.owner.json')
                if old_temp.exists() and owner.exists() and read(owner)==dict(member=row['member'],crc=row['crc'],bytes=row['bytes']):
                    checked_temporary(old_temp,row)
                    old_temp.unlink()
                    owner.unlink()
                continue
            reused=reuse_feature(args,row,path,config)
            if reused is not None:
                state['completed'][vid]=reused;state['failed'].pop(vid,None)
                write(state_path,state)
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
                quality_check(mask,args)
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


def batches(rows, output, batch_size, order=None):
    rows = rows if order is None else [rows[i] for i in order]
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start+batch_size]
        arrays = []
        for row in subset:
            path = output/'features'/(row['video_id']+'.npz')
            with np.load(path, allow_pickle=False) as data:
                arrays.append((data['features'].copy(), data['target'].copy()))
        lengths = np.asarray([len(x) for x,y in arrays], np.int64)
        values = np.zeros((len(arrays), int(lengths.max()), 424), np.float32)
        for i, (x,y) in enumerate(arrays):
            values[i,:len(x)] = x
        yield dict(features=values, input_lengths=lengths, targets=np.concatenate([y for x,y in arrays]),
                   target_lengths=np.asarray([len(y) for x,y in arrays], np.int64))


def evaluate(model, rows, output, device):
    videos = []
    model.eval()
    with torch.inference_mode():
        for row, batch in zip(rows, batches(rows, output, 1)):
            if device.type == 'cuda': torch.cuda.synchronize()
            tick = time.perf_counter()
            scores, lengths = model(torch.as_tensor(batch['features'], device=device), batch['input_lengths'])
            ids = greedy_decode(scores[0].cpu().tolist(), blank_id=0, num_classes=52, length=int(lengths[0]))
            predicted = ''.join(tokens_for(ids))
            target = ''.join(tokens_for(row['target_ids']))
            text = render(ids)
            elapsed = time.perf_counter()-tick
            counts = edit_counts(target, predicted)
            hangul = edit_counts(row['text'], text)
            videos.append(dict(video_id=row['video_id'], reference_jamo=target, predicted_jamo=predicted,
                reference=row['text'], prediction=text, D=counts.deletions, I=counts.insertions,
                S=counts.substitutions, N=len(target), hangul_errors=hangul.errors,
                hangul_N=len(row['text']), exact=text == row['text'], seconds=elapsed))
    return dict(samples=len(rows), jamo_cer=sum(r['D']+r['I']+r['S'] for r in videos)/sum(r['N'] for r in videos),
                hangul_cer=sum(r['hangul_errors'] for r in videos)/sum(r['hangul_N'] for r in videos),
                deletions=sum(r['D'] for r in videos), insertions=sum(r['I'] for r in videos),
                substitutions=sum(r['S'] for r in videos), exact_match=sum(r['exact'] for r in videos)/len(videos),
                inference_mean_ms=float(np.mean([r['seconds']*1000 for r in videos])), videos=videos)


def train(args, state):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('requested_CUDA_unavailable')
    seed_everything(args.seed)
    rows = [r for r in state['selected'] if r['video_id'] in state['completed']]
    splits = split_rows(rows, args.seed)
    vocab = vocabulary()
    config = dict(seed=args.seed, hidden_size=64, epochs=args.epochs, batch_size=8,
                  feature_schema=feature_schema(), vocabulary=dict(token_ids=dict(vocab.token_ids), blank_id=0),
                  splits=splits, feature_hashes={r['video_id']:state['completed'][r['video_id']]['sha256'] for r in rows},
                  performer_independent=False, label_semantics='unverified_exploratory')
    config_path = args.output/'training-config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('training_configuration_changed')
    write(config_path, config)
    model = FingerspellingCTC(CTCModelConfig(input_size=424, num_classes=52, blank_id=0, hidden_size=64))
    trainer = CTCTrainer(model, CTCTrainingConfig(seed=args.seed), device='cuda:0',
                         vocabulary=config['vocabulary'], feature_schema=config['feature_schema'])
    ckpt = args.output/'checkpoints'
    ckpt.mkdir(exist_ok=True)
    if (ckpt/'last.pt').exists(): trainer.resume(ckpt/'last.pt')
    # Explicit finite forward/backward smoke, without updating production weights.
    if trainer.epoch == 0:
        probe = next(batches(splits['train'][:2], args.output, 2))
        from fingerspelling.ctc_training import ctc_loss
        scores, lengths = trainer.model(torch.as_tensor(probe['features'], device=trainer.device), probe['input_lengths'])
        loss = ctc_loss(scores, lengths, probe['targets'], probe['target_lengths'], blank_id=0)
        loss.backward()
        trainer.optimizer.zero_grad(set_to_none=True)
        write(args.output/'forward-backward.json', dict(loss=float(loss.detach()), finite=True, device='cuda:0'))
    history_path = args.output/'history.json'
    history = read(history_path) if history_path.exists() else []
    if len(history) > trainer.epoch:
        raise ValueError('history_ahead_of_checkpoint')
    for row in trainer.history[len(history):]:
        history.append(dict(row, recovered=True, train_cer=None, validation_cer=None, seconds=None))
    started = time.perf_counter()
    while trainer.epoch < args.epochs:
        tick = time.perf_counter()
        order = np.random.default_rng(args.seed*1000+trainer.epoch).permutation(len(splits['train']))
        trainer.fit(lambda: batches(splits['train'], args.output, 8, order),
                    lambda: batches(splits['validation'], args.output, 8), epochs=1, checkpoint_dir=ckpt)
        training = evaluate(trainer.model, splits['train'], args.output, trainer.device)
        validation = evaluate(trainer.model, splits['validation'], args.output, trainer.device)
        history.append(dict(trainer.history[-1], train_cer=training['jamo_cer'],
            validation_cer=validation['jamo_cer'], seconds=time.perf_counter()-tick))
        write(history_path, history)
        print(json.dumps(history[-1]), flush=True)
    # Repair best artifact if interrupted between last.pt and best.pt write.
    from fingerspelling.ctc_checkpoint import write_checkpoint
    write_checkpoint(ckpt/'best.pt', trainer.best_checkpoint)
    reader = FingerspellingReader(ckpt/'best.pt', device='cuda:0')
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        selected_epoch=load_checkpoint(ckpt/'best.pt')['epoch'], epoch=trainer.epoch,
        parameters=sum(p.numel() for p in reader.recognizer.model.parameters()),
        checkpoint_bytes=(ckpt/'best.pt').stat().st_size, training_session_seconds=time.perf_counter()-started,
        training_seconds=sum(r['seconds'] for r in history) if all(r['seconds'] is not None for r in history) else None,
        final_train_loss=history[-1]['train_loss'], final_validation_loss=history[-1]['validation_loss'],
        performer_independent=False, semantic_validation='unverified', checkpoint=str(ckpt/'best.pt'),
        splits={name:evaluate(reader.recognizer.model, subset, args.output, reader.recognizer.device)
                for name,subset in splits.items()})
    probe_path=args.output/'features'/(splits['test'][0]['video_id']+'.npz')
    with np.load(probe_path, allow_pickle=False) as probe:
        result['interface_prediction']=reader(probe['xy'],probe['mask'])
        fresh=FingerspellingReader(ckpt/'best.pt', device='cuda:0')
        result['checkpoint_reload_identical']=result['interface_prediction']==fresh(probe['xy'],probe['mask'])
    write(args.output/'evaluation.json',result)
    state['stages'].update(training=True,evaluation=True,interface=True)
    write(args.output/'progress.json',state)
    print(json.dumps({k:v for k,v in result.items() if k!='splits'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','train','all'],default='all')
    parser.add_argument('--max-videos',type=int,default=64)
    parser.add_argument('--extraction-seconds',type=float,default=900)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--seed',type=int,default=20261114)
    parser.add_argument('--selection-file',type=Path)
    parser.add_argument('--reuse-features',type=Path)
    parser.add_argument('--temp-root',type=Path)
    parser.add_argument('--min-hand-coverage',type=float,default=0.)
    parser.add_argument('--min-shoulder-coverage',type=float,default=0.)
    args=parser.parse_args()
    if args.max_videos<10 or args.epochs<1 or not np.isfinite(args.extraction_seconds) or args.extraction_seconds<=0:
        parser.error('invalid execution budget')
    if any(not np.isfinite(v) or not 0<=v<=1 for v in (args.min_hand_coverage,args.min_shoulder_coverage)):
        parser.error('quality fractions must lie in [0,1]')
    if args.temp_root is not None and args.temp_root.resolve().drive.upper()!='D:':
        parser.error('temporary video path must be on D drive')
    args.output=args.output.resolve()
    if args.output.drive.upper()!='D:' or args.output==ROOT or ROOT in args.output.parents:
        parser.error('outputs must be outside repository on D drive')
    for p in ('features','logs','diagnostics'):
        (args.output/p).mkdir(parents=True,exist_ok=True)
    before={str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in (args.videos,args.labels)}
    state=prepare(args) if args.stage in ('prepare','all') else read(args.output/'progress.json')
    if args.stage in ('train','all'):
        # Validate all completed features and extraction configuration before training/resume.
        if state['config']!=source_config(args): raise ValueError('prepare_source_or_config_changed')
        for row in state['selected']:
            if row['video_id'] in state['completed']:
                expected=state['completed'][row['video_id']]['sha256']
                if feature_ok(args.output/'features'/(row['video_id']+'.npz'),row)!=expected:
                    raise ValueError('feature_changed')
        train(args,state)
    after={str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in (args.videos,args.labels)}
    if before!=after: raise RuntimeError('source_metadata_changed')
    write(args.output/('integrity-'+args.stage+'.json'),dict(source_stat_unchanged=True,
        protection='read-only ZIP access; size/mtime comparison, not whole ZIP hash',sources=after))


if __name__=='__main__': main()
