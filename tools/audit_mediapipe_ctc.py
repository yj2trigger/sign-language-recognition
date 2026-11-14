"""Read-only source/feature audit; writes only a new report in the run folder."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from run_mediapipe_ctc import ROOT, read, digest, feature_ok, split_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    state=read(args.run/'progress.json')
    report=dict(disk_free_bytes=shutil.disk_usage(args.run).free,
                torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                gpu_memory_free_total=list(torch.cuda.mem_get_info()),
                estimated_training_feature_bytes=sum(r['frames']*424*4 for r in state['completed'].values()),
                estimated_training_scope='batch8 hidden64; feature bytes are not total GPU allocation')
    if not args.preflight:
        outputs=args.run.parent.parent
        previous=read(outputs/'ctc-deletion-diagnosis-v1/integrity-analyze.json')
        unchanged={name:digest(outputs/'full-fit-v1'/name)==sha for name,sha in previous['protected_sha256'].items()}
        if not all(unchanged.values()):raise ValueError('old_training_artifact_changed')
        report['old_training_files_unchanged']=len(unchanged)
        backup=json.loads((args.backup/'backup-manifest.json').read_text(encoding='utf-8-sig'))
        if not all(digest(ROOT/r['path']).upper()==r['sha256'].upper() for r in backup):
            raise ValueError('preexisting_untracked_file_changed')
        report['preexisting_untracked_files_unchanged']=len(backup)
        for name,sha in previous['source_sha256'].items():
            pattern='keypoint' if name=='keypoints' else 'morpheme'
            # Synthetic test ZIPs also live under outputs; never select those.
            sources=[x for child in outputs.parent.iterdir() if child.is_dir() and child!=outputs
                     for x in child.rglob('*.zip') if pattern in x.name]
            if len(sources)!=1:raise ValueError('ambiguous_original_source_zip')
            source=sources[0]
            if digest(source)!=sha:raise ValueError('source_zip_changed')
        report['old_keypoint_and_morpheme_zip_sha256_unchanged']=True
        for model in ('A','B'):
            provenance=read(outputs/'ctc-deletion-diagnosis-v1/probability-provenance.json')
            cache=outputs/'ctc-deletion-diagnosis-v1'/f'log-probs-{model}.npz'
            if digest(cache)!=provenance['probabilities'][model]['sha256']:
                raise ValueError('old_probability_cache_changed')
        report['old_probability_caches_unchanged']=True
        intervals=[]
        selected=[]
        for row in state['selected']:
            if row['video_id'] not in state['completed']:continue
            selected.append(row)
            path=args.run/'features'/(row['video_id']+'.npz')
            if feature_ok(path,row)!=state['completed'][row['video_id']]['sha256']:
                raise ValueError('new_feature_changed')
            with np.load(path,allow_pickle=False) as data:intervals.extend(np.diff(data['t']).tolist())
        splits=split_rows(selected,state['config']['seed'])
        for field in ('fs_id','text','video_id'):
            sets=[{r[field] for r in subset} for subset in splits.values()]
            if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise ValueError('split_leakage')
        report.update(feature_files_verified=len(selected),split_sizes={k:len(v) for k,v in splits.items()},
            actual_selected_dt_min=float(min(intervals)),actual_selected_dt_max=float(max(intervals)),
            nominal_target_fps=15,uniform_15fps_claim=False,
            actual_selected_dt_median=float(np.median(intervals)),
            reused_verified_features=sum(r['reused_verified_cfr_features'] for r in state['completed'].values()),
            failures=state['failed'],performer_independent=False)
        provenance=(ROOT/'src/fingerspelling/provenance.md').read_text().splitlines()
        copied=0
        for line in provenance:
            if not line.startswith('- `'):continue
            revision=line.split('`')[1]
            for name in line.split(': ',1)[1].split(', '):
                raw=subprocess.check_output(['git','show',f'{revision}:src/fingerspelling/{name}.py'],cwd=ROOT)
                if raw!=(ROOT/'src/fingerspelling'/(name+'.py')).read_bytes():raise ValueError('reused_module_changed')
                copied+=1
        report['byte_identical_reused_modules']=copied
    destination=args.run/('resource-preflight.json' if args.preflight else 'validation-audit.json')
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='failures'}),flush=True)


if __name__=='__main__':main()
