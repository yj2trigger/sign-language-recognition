"""Read-only final checks. Never extracts videos, trains, or changes source data."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fields(value, prefix=''):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + '.' + key if prefix else key
            result.append(path)
            result.extend(fields(child, path))
    elif isinstance(value, list):
        for child in value:
            result.extend(fields(child, prefix + '[]'))
    return result


def signer_audit(data, blocks):
    previous = data / 'outputs/mediapipe-ctc-performance/data-expansion-01'
    progress = read(previous / 'progress.json')
    labels = [p for p in data.rglob('*morpheme.zip') if 'outputs' not in p.parts]
    if len(labels) != 1:
        raise ValueError('ambiguous_original_label_zip')
    schema = Counter()
    urls = Counter()
    candidates = Counter()
    names = []
    url_candidates = {}
    with zipfile.ZipFile(labels[0]) as archive:
        for member in archive.infolist():
            if not member.filename.endswith('.json'):
                continue
            raw = json.loads(archive.read(member).decode('utf-8-sig'))
            paths = fields(raw)
            schema.update(set(paths))
            candidates.update(path for path in paths if re.search(
                r'signer|performer|person|speaker|subject|participant|session|actor', path, re.I))
            name = Path(raw['metaData']['name']).stem.rstrip('_')
            names.append(name)
            urls['/'.join(raw['metaData']['url'].split('/')[:-1])] += 1
            match = re.search(r'/\d{4}_([a-zA-Z]+)\d+(?:/|$)', raw['metaData']['url'])
            if match:
                url_candidates[name] = hashlib.sha256(match[1].encode()).hexdigest()[:12]
    with zipfile.ZipFile(progress['config']['video_path']) as archive:
        members = [m.filename for m in archive.infolist() if m.filename.lower().endswith('.mp4')]
    config = read(previous / 'training-config.json')
    # Find actual stored split objects, rather than inferring a performer from filenames.
    split = config.get('split', config.get('splits'))
    if split is None:
        raise ValueError('training_split_not_found')
    if not isinstance(split, dict) or not all(isinstance(v, list) for v in split.values()):
        raise ValueError('unrecognized_training_split')
    split_ids = {key: [item['video_id'] if isinstance(item, dict) else item for item in value]
                 for key, value in split.items()}
    split_sets = {key: set(value) for key, value in split_ids.items()}
    overlap = {a + ':' + b: sorted(split_sets[a] & split_sets[b])
               for a in split for b in split if a < b}
    groups = {key: dict(Counter(re.search(r'CROWD\d+', str(vid)).group() for vid in value))
              for key, value in split_ids.items()}
    states = [read(p) for p in (blocks / 'videos').glob('*/state.json')]
    proxies = {key: set(url_candidates[vid] for vid in value if vid in url_candidates)
               for key, value in split_ids.items()}
    return dict(label_json_count=len(names),metadata_field_counts=dict(schema),
        performer_or_session_fields=dict(candidates),url_parent_group_count=len(urls),
        unverified_url_prefix_groups=len(set(url_candidates.values())),
        url_pattern_matched=len(url_candidates),url_pattern_unmatched=len(names)-len(url_candidates),
        url_pattern_split_matched={k:sum(vid in url_candidates for vid in v) for k,v in split_ids.items()},
        unverified_url_prefix_split_counts={k:len(v) for k,v in proxies.items()},
        unverified_url_prefix_split_overlap={a+':'+b:len(proxies[a]&proxies[b]) for a in proxies for b in proxies if a<b},
        video_members=len(members),video_directories=dict(Counter(str(Path(m).parent) for m in members)),
        labels_unique_ids=len(set(names)),split_sizes={k: len(v) for k, v in split.items()},
        video_id_overlap=overlap,filename_group_counts=groups,
        technical_video_statuses=dict(Counter(s['technical_status'] for s in states)),
        original_label_zip_sha256=sha(labels[0]),reliable_performer_id_available=False,
        performer_independent_evaluation='NOT_ESTABLISHED',performer_count=None,
        independent_evaluation_video_count=0,jamo_CER=None,hangul_CER=None,exact_match=None,
        reason='No performer mapping in original JSON; CROWD directory/name is not an authenticated person ID. Official mapping required.')


def api_cases(xy, mask):
    """Contract probes, including real numeric integer coordinates (accepted)."""
    nan=xy.copy();nan[0,0,0]=np.nan
    inf=xy.copy();inf[0,0,0]=np.inf
    left=mask.copy();left[:,11:32]=False
    gaps=mask.copy();gaps[::3]=False
    return dict(normal=(xy,mask),empty=(xy[:0],mask[:0]),single=(xy[:1],mask[:1]),
        long=(np.tile(xy,(10,1,1)),np.tile(mask,(10,1))),
        no_hands=(xy,np.concatenate((mask[:,:11],np.zeros_like(mask[:,11:])),axis=1)),
        one_hand=(xy,left),missing_frames=(xy,gaps),nan=(nan,mask),inf=(inf,mask),
        wrong_shape=(xy[:,:52],mask[:,:52]),wrong_xy_dtype=(xy.astype(str),mask),
        wrong_mask_dtype=(xy,mask.astype(np.uint8)))


def api_audit(feature_directory, checkpoint):
    import torch
    import fingerspelling_runtime as runtime
    torch.set_num_threads(2)
    previous={key:os.environ.get(key) for key in ('FINGERSPELLING_CHECKPOINT','FINGERSPELLING_RENDER_MODE')}
    os.environ['FINGERSPELLING_CHECKPOINT']=str(checkpoint.resolve())
    os.environ['FINGERSPELLING_RENDER_MODE']='strict'
    samples=[];probes=[]
    try:
        paths=sorted(feature_directory.glob('*.npz'))[:3]
        if len(paths)!=3:raise ValueError('three_existing_features_required')
        gpu=runtime.FingerspellingReader(checkpoint,device='cuda') if torch.cuda.is_available() else None
        for path in paths:
            before=sha(path)
            with np.load(path,allow_pickle=False) as data:
                xy=data['xy'].copy();mask=data['mask'].copy()
            output=runtime.read_fingerspelling(xy,mask)
            cuda=gpu(xy,mask) if gpu is not None else None
            samples.append(dict(video_id=path.stem,input_shape=list(xy.shape),mask_shape=list(mask.shape),
                source_sha256=before,output_type=type(output).__name__,cpu_output=output,gpu_output=cuda,
                CPU_GPU_string_equal=output==cuda if gpu is not None else None))
            if sha(path)!=before:raise AssertionError('source_feature_changed')
        for name,(values,valid) in api_cases(xy,mask).items():
            try:
                result=runtime.read_fingerspelling(values,valid)
                probes.append(dict(case=name,result='RETURN',output_type=type(result).__name__,output=result))
            except (ValueError,TypeError,RuntimeError) as exc:
                probes.append(dict(case=name,result='ERROR',error_type=type(exc).__name__,reason=str(exc)))
        failures=api_failures(samples,probes)
        return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),contract_failures=failures,contract_passed=not failures,
            torch=torch.__version__,cuda_available=torch.cuda.is_available(),
            GPU=torch.cuda.get_device_name(0) if gpu is not None else None,
            public_API_device='CPU default; separate same-checkpoint CUDA reader compared',
            render_mode='strict',samples=samples,contract_probes=probes,
            interpretation='Predictions are software checks, never independent semantic ground truth.')
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def api_failures(samples,probes):
    failures=[]
    for sample in samples:
        if sample['output_type']!='str':failures.append(sample['video_id']+': output type')
        if sample.get('CPU_GPU_string_equal') is False:failures.append(sample['video_id']+': CPU/GPU strings differ')
    error_cases={'nan','inf','wrong_shape','wrong_xy_dtype','wrong_mask_dtype'}
    empty_cases={'empty','single','no_hands'}
    expected=set(api_cases(np.zeros((2,53,2),np.float32),np.ones((2,53),bool)))
    if {p['case'] for p in probes}!=expected:failures.append('missing contract probe')
    for probe in probes:
        if probe['case'] in error_cases:
            if probe['result']!='ERROR':failures.append(probe['case']+': error expected')
        elif probe['result']!='RETURN' or probe.get('output_type')!='str':
            failures.append(probe['case']+': str expected')
        elif probe['case'] in empty_cases and probe.get('output')!='':
            failures.append(probe['case']+': empty str expected')
    return failures


def protection_manifest(data,blocks,protected_repo):
    outputs=data/'outputs'
    previous=outputs/'mediapipe-ctc-performance/data-expansion-01'
    artifacts=[*sorted((previous/'features').glob('*.npz')),
        *sorted((outputs/'mediapipe-ctc-v1/pts-run-01/checkpoints').glob('*.pt')),
        *sorted((outputs/'mediapipe-ctc-performance/expanded-models-01').glob('*/*.pt'))]
    names=subprocess.run(['git','ls-files','--others','--exclude-standard','-z','--',':(exclude).tmp-issue2'],cwd=protected_repo,
        capture_output=True,check=True).stdout.decode('utf-8').split('\0')
    untracked=[protected_repo/name for name in names if name and not name.startswith('.tmp-issue2/')]
    originals=[p for p in data.rglob('*.zip') if 'outputs' not in p.parts]
    results=sorted(p for p in blocks.rglob('*') if p.is_file())
    return dict(original_zips={str(p):sha(p) for p in originals},
        previous_artifacts={str(p):sha(p) for p in artifacts},
        all_existing_checkpoints={str(p):sha(p) for p in outputs.rglob('*.pt')},
        expanded_training_metadata={str(p):sha(p) for p in (outputs/'mediapipe-ctc-performance/expanded-models-01').rglob('*.json')},
        original_training_config_sha256=sha(previous/'training-config.json'),
        previous_untracked={str(p):sha(p) for p in untracked},
        existing_block_results={str(p):sha(p) for p in results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['signers','api','guard'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--blocks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--protected-repo',type=Path,default=ROOT)
    args = parser.parse_args()
    if args.output.resolve().drive.upper() != 'D:':
        parser.error('D output required')
    if args.output.exists():
        parser.error('output already exists; never overwrite an existing result')
    if args.mode=='signers':
        result = signer_audit(args.data, args.blocks)
    elif args.mode=='api':
        if args.checkpoint is None:parser.error('api requires --checkpoint')
        result = api_audit(args.data/'outputs/mediapipe-ctc-performance/data-expansion-01/features',args.checkpoint)
    else:
        result=protection_manifest(args.data,args.blocks,args.protected_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:len(v) if isinstance(v,dict) else v for k,v in result.items()} if args.mode=='guard' else result, ensure_ascii=False))
    if args.mode=='api' and not result['contract_passed']:raise SystemExit(1)


if __name__ == '__main__':
    main()
