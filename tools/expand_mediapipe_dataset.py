"""Private inventory and train-only rare-token selection; reuse the PTS builder."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import sys
import unicodedata
import zipfile

import run_mediapipe_ctc as base


def select_additions(rows,baseline,failed,limit,seed):
    """Held-out labels are used only to forbid group leakage, never for scoring."""
    existing={r['video_id'] for group in baseline['splits'].values() for r in group}
    heldout=baseline['splits']['validation']+baseline['splits']['test']
    blocked_fs={r['fs_id'] for r in heldout}
    blocked_text={unicodedata.normalize('NFC',r['text']) for r in heldout}
    pool=[r for r in rows if r['video_id'] not in existing and r['video_id'] not in failed
          and r['fs_id'] not in blocked_fs and unicodedata.normalize('NFC',r['text']) not in blocked_text]
    random.Random(seed).shuffle(pool)
    rank={r['video_id']:i for i,r in enumerate(pool)}
    counts=Counter(i for r in baseline['splits']['train'] for i in r['target_ids'])
    chosen=[]
    for _ in range(min(limit,len(pool))):
        def priority(r):
            tokens=set(r['target_ids'])
            return (sum(t not in counts for t in tokens),sum(1/(1+counts[t]) for t in tokens)/max(len(tokens),1),
                    -r['bytes'],-rank[r['video_id']])
        best=max(pool,key=priority)
        chosen.append(best);pool.remove(best);counts.update(best['target_ids'])
    return chosen


def write_inventory(args,eligible,baseline):
    path=args.output/'inventory.jsonl'
    identity=dict(label_sha256=base.digest(args.labels),video_size=args.videos.stat().st_size,
                  video_mtime=args.videos.stat().st_mtime_ns)
    metadata=args.output/'inventory-source.json'
    if metadata.exists() and base.read(metadata)!=identity:raise ValueError('inventory_source_changed')
    if path.exists() and not metadata.exists():raise ValueError('inventory_without_source_contract')
    if path.exists():return
    base.write(metadata,identity)
    lookup={r['video_id']:r for r in eligible}
    source_state=base.read(args.source/'progress.json')
    split={r['video_id']:name for name,rows in baseline['splits'].items() for r in rows}
    labels={}
    with zipfile.ZipFile(args.labels) as archive:
        for member in archive.infolist():
            if member.filename.endswith('.json'):
                raw=json.loads(archive.read(member).decode('utf-8-sig'))
                vid=Path(raw['metaData']['name']).stem
                if vid in labels:raise ValueError('duplicate_label_ID')
                labels[vid]=(member.filename,raw)
    temporary=path.with_suffix('.jsonl.tmp')
    with zipfile.ZipFile(args.videos) as archive,temporary.open('w',encoding='utf-8') as output:
        seen=set()
        for member in archive.infolist():
            if not member.filename.lower().endswith('.mp4'):continue
            match=re.fullmatch(r'(NIA_SL_FS[0-9]{4}_CROWD[0-9]{2}_F)_?',Path(member.filename).stem)
            vid=match[1] if match else Path(member.filename).stem
            if vid in seen:raise ValueError('duplicate_video_ID')
            seen.add(vid)
            label_id,raw=labels.get(vid,(None,{}))
            record=dict(zip=str(args.videos),mp4_member=member.filename,video_id=vid,bytes=member.file_size,
                crc=member.CRC,label_id=label_id,label_document=raw,target_supported=vid in lookup,
                automatic_status='candidate_only' if vid in lookup else 'unsupported_target_or_mapping',
                human_review_status='uncertain',fingerspelling_verified=False,performer_id=None,
                extraction_status='verified_feature' if vid in source_state['completed'] else 'unprocessed',
                previous_failure=source_state['failed'].get(vid),split=split.get(vid,'unused'))
            output.write(json.dumps(record,ensure_ascii=False)+'\n')
    temporary.replace(path)


def finalize(args,state,baseline,plan):
    original={r['video_id']:name for name,rows in baseline['splits'].items() for r in rows}
    splits={name:[] for name in ('train','validation','test')}
    by_id={r['video_id']:r for r in state['selected']}
    for name in splits:
        for old in baseline['splits'][name]:
            vid=old['video_id']
            if vid not in state['completed']:raise ValueError('baseline_sample_not_reused')
            splits[name].append(by_id[vid])
    for row in state['selected']:
        if row['video_id'] not in original and row['video_id'] in state['completed']:splits['train'].append(row)
    for field in ('video_id','fs_id','text'):
        groups=[{unicodedata.normalize('NFC',r[field]) for r in rows} for rows in splits.values()]
        if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):raise ValueError('expanded_group_leakage')
    counts={name:Counter(i for r in rows for i in r['target_ids']) for name,rows in splits.items()}
    config=dict(baseline,splits=splits,feature_hashes={vid:info['sha256'] for vid,info in state['completed'].items()},
        expanded=True,semantic_validation='unverified_research_diagnosis',performer_independent=False,
        baseline_config_sha256=base.digest(args.source/'training-config.json'))
    result=dict(original_videos=state['candidate_videos'],label_jsons=state['label_jsons'],
        eligible_targets=state['eligible_pairs'],exclusions=state['exclusions'],planned_additional=len(plan['additional_ids']),
        reused_features=len(original),new_features=sum(v not in original for v in state['completed']),
        total_features=len(state['completed']),split_sizes={k:len(v) for k,v in splits.items()},
        unique_video_ids=len(state['completed']),failed=state['failed'],
        unprocessed=len(state['selected'])-len(state['completed'])-len(state['failed']),
        token_counts={k:dict(v) for k,v in counts.items()},token_types={k:len(v) for k,v in counts.items()},
        old_train_types=len({i for r in baseline['splits']['train'] for i in r['target_ids']}),
        hand_coverage_min=.25,shoulder_coverage_min=.5,performer_independent=False,
        semantic_verified_samples=0,dataset_frozen=True)
    for path,value in ((args.output/'training-config.json',config),(args.output/'expansion-summary.json',result)):
        if path.exists():
            if base.read(path)!=value:raise ValueError('frozen_dataset_metadata_changed')
        else:base.write(path,value)
    inventory=args.output/'inventory-final.jsonl'
    if not inventory.exists():
        temporary=inventory.with_suffix('.jsonl.tmp')
        assignment={r['video_id']:name for name,rows in splits.items() for r in rows}
        with (args.output/'inventory.jsonl').open(encoding='utf-8') as source,temporary.open('w',encoding='utf-8') as out:
            for line in source:
                row=json.loads(line);vid=row['video_id'];row['split']=assignment.get(vid,'unused')
                row['extraction_status']='verified_feature' if vid in state['completed'] else ('failed' if vid in state['failed'] else 'unprocessed')
                row['failure']=state['failed'].get(vid);row['selected']=vid in by_id
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
        temporary.replace(inventory)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--additional',type=int,default=96);p.add_argument('--seconds',type=float,default=1200)
    p.add_argument('--seed',type=int,default=20261114);args=p.parse_args()
    args.output=args.output.resolve();args.source=args.source.resolve()
    if not 1<=args.additional<=256 or not math.isfinite(args.seconds) or args.seconds<=0 or args.output.drive.upper()!='D:' or base.ROOT in args.output.parents:
        p.error('invalid bounded expansion configuration')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'features').mkdir(exist_ok=True)
    baseline=base.read(args.source/'training-config.json')
    if baseline['feature_schema']!=base.feature_schema():raise ValueError('baseline_feature_schema_changed')
    contract=dict(seed=args.seed,additional=args.additional,baseline_sha256=base.digest(args.source/'training-config.json'),
        label_sha256=base.digest(args.labels),video_bytes=args.videos.stat().st_size,video_mtime=args.videos.stat().st_mtime_ns,
        selection_code_sha256=base.digest(__file__))
    if (args.output/'training-config.json').exists():
        if base.read(args.output/'selection.json')['contract']!=contract:raise ValueError('frozen_source_contract_changed')
        frozen=base.read(args.output/'training-config.json')
        for rows in frozen['splits'].values():
            for row in rows:
                if base.feature_ok(args.output/'features'/(row['video_id']+'.npz'),row)!=frozen['feature_hashes'][row['video_id']]:
                    raise ValueError('frozen_feature_changed')
        finalize(args,base.read(args.output/'progress.json'),baseline,base.read(args.output/'selection.json'))
        print('frozen_dataset_verified_no_reextraction',flush=True);return
    rows,rejected,nvideos,nlabels=base.candidates(args.videos,args.labels)
    write_inventory(args,rows,baseline)
    source_state=base.read(args.source/'progress.json')
    plan_path=args.output/'selection.json'
    if plan_path.exists():
        plan=base.read(plan_path)
        if plan['contract']!=contract:raise ValueError('selection_contract_changed')
    else:
        additions=select_additions(rows,baseline,set(source_state['failed']),args.additional,args.seed)
        original=[r['video_id'] for group in baseline['splits'].values() for r in group]
        plan=dict(contract=contract,video_ids=original+[r['video_id'] for r in additions],
            additional_ids=[r['video_id'] for r in additions],selection='Train rarity only; held-out FS/text excluded')
        base.write(plan_path,plan)
    prepare_args=argparse.Namespace(output=args.output,videos=args.videos,labels=args.labels,seed=args.seed,
        max_videos=len(plan['video_ids']),selection_file=plan_path,reuse_features=args.source,
        min_hand_coverage=.25,min_shoulder_coverage=.5,extraction_seconds=args.seconds,
        temp_root=args.output.parents[1].parent/'original-video'/'temp'/args.output.name)
    # Both source files and all already extracted baseline features stay immutable.
    before={path:base.digest(args.source/path) for path in ('training-config.json','progress.json')}
    state=base.prepare(prepare_args)
    result=finalize(args,state,baseline,plan)
    if any(base.digest(args.source/path)!=sha for path,sha in before.items()):raise RuntimeError('baseline_changed')
    print(json.dumps({k:v for k,v in result.items() if k not in ('failed','token_counts')}),flush=True)


if __name__=='__main__':main()
