"""Bind Stage3 provenance to the unchanged PR23 independent-evaluation gate."""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fingerspelling.ctc_checkpoint import load_checkpoint

PR23='ece5a90dcaea65ecb0dc297f43f55e9b78651a83'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def prior_evaluator():
    """Reuse pinned Git objects, without merge/cherry-pick or copying pipelines."""
    hashes={}
    for name in ('process_crowd_blocks','build_independent_review_handoff','evaluate_independent_cer'):
        path=f'tools/{name}.py'
        try:raw=subprocess.check_output(['git','show',f'{PR23}:{path}'],cwd=ROOT,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:raise RuntimeError('required_PR23_commit_unavailable') from exc
        module=types.ModuleType(name)
        module.__file__=str(ROOT/path)
        sys.modules[name]=module
        exec(compile(raw,f'{PR23}:{path}','exec'),module.__dict__)
        hashes[path]=hashlib.sha256(raw).hexdigest()
    return module,hashes


def bind_provenance(training_config,stage3,checkpoint):
    """Bridge recorded hashes, not fabricate signer/transcription evidence."""
    training_config=Path(training_config).resolve();stage3=Path(stage3).resolve();checkpoint=Path(checkpoint).resolve()
    manifest=read(stage3/'manifest.json');selection=read(stage3/'selection.json')['selected']
    if Path(manifest['source']).resolve()!=training_config.parent:
        raise ValueError('stage3_training_source_mismatch')
    if manifest['protected'].get(str(training_config))!=digest(training_config):
        raise ValueError('stage3_training_config_hash_mismatch')
    if Path(selection['checkpoint']).resolve()!=checkpoint or selection['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('stage3_selected_checkpoint_mismatch')
    result=read(checkpoint.parent/'result.json')
    if result!=selection:raise ValueError('stage3_selected_result_mismatch')
    payload=load_checkpoint(checkpoint)
    if payload['epoch']!=result['selected_epoch']:
        raise ValueError('stage3_checkpoint_epoch_mismatch')
    for field in ('seed','weight_decay'):
        if payload['training_config'][field]!=result['config'][field]:
            raise ValueError('stage3_checkpoint_training_config_mismatch')
    if payload['training_config']['learning_rate']!=result['config']['lr']:
        raise ValueError('stage3_checkpoint_learning_rate_mismatch')
    return dict(config=dict(source_config_sha256=digest(training_config)),inference_sha256=digest(checkpoint),
        stage3_inputs={str(p):digest(p) for p in (stage3/'manifest.json',stage3/'selection.json',checkpoint.parent/'result.json')},
        interpretation='Compatibility binding only; no signer or transcript approval is created.')


def exposure_filter(config_paths,reviews,canonical):
    """Keep evaluation new to recorded training/selection/observed-test histories."""
    identities=set();hashes=set()
    for path in config_paths:
        history=read(path)
        groups=history['splits'].values() if 'splits' in history else [history['train']]
        identities.update(canonical(row['video_id'].removesuffix('.mp4')) for rows in groups for row in rows)
        for field in ('feature_hashes','feature_sha256'):
            values=history.get(field,{})
            # Legacy scalar is a manifest-wide digest, not one feature identity.
            if isinstance(values,str):continue
            if not isinstance(values,dict):raise ValueError('unsupported_history_hash_schema')
            hashes.update(v for v in values.values() if isinstance(v,str) and len(v)==64)
    accepted=[];rejected=[]
    for row in reviews:
        reasons=[]
        if canonical(row['video_id']) in identities:reasons.append('PREVIOUS_EXPERIMENT_VIDEO')
        if row.get('source_video_id') and canonical(row['source_video_id']) in identities:
            reasons.append('PREVIOUS_EXPERIMENT_SOURCE')
        if row.get('feature_sha256') in hashes:reasons.append('PREVIOUS_EXPERIMENT_FEATURE')
        if reasons:rejected.append(dict(video_id=row['video_id'],reasons=reasons))
        else:accepted.append(row)
    return accepted,rejected,len(identities)


def run(training_config,stage3,checkpoint,mapping_csv,review_csv,feature_root,evidence_root,output,device='cpu',history_configs=()):
    paths=[Path(p).resolve() for p in (training_config,stage3,checkpoint,mapping_csv,review_csv,feature_root,evidence_root,output)]
    training_config,stage3,checkpoint,mapping_csv,review_csv,feature_root,evidence_root,output=paths
    if output.drive.upper()!='D:' or any(output==p or output in p.parents or p in output.parents
                                         for p in (stage3,feature_root,checkpoint.parent)):
        raise ValueError('separate_D_output_required')
    if output.exists():raise FileExistsError('output_already_exists')
    if not history_configs:raise ValueError('experiment_history_required')
    histories=[Path(p).resolve() for p in history_configs]
    inputs=(training_config,stage3/'manifest.json',stage3/'selection.json',checkpoint,
            checkpoint.parent/'result.json',mapping_csv,review_csv,*histories)
    initial={str(p):digest(p) for p in inputs}
    binding=bind_provenance(training_config,stage3,checkpoint)
    evaluator,reused=prior_evaluator()
    reviews=evaluator.load_csv(review_csv)
    accepted,rejected,known=exposure_filter([training_config,*histories],reviews,evaluator.canonical)
    output.mkdir(parents=True,exist_ok=False)
    provenance=output/'bound-model-provenance.json'
    provenance.write_text(json.dumps(binding,ensure_ascii=False,indent=2),encoding='utf-8')
    filtered=output/'new-to-experiments-reviews.csv'
    with filtered.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(reviews[0]) if reviews else ['video_id','review_status'])
        writer.writeheader();writer.writerows(accepted)
    result=evaluator.run(training_config,provenance,checkpoint,mapping_csv,filtered,feature_root,evidence_root,device)
    if initial!={str(p):digest(p) for p in inputs}:raise ValueError('stage4_inputs_changed')
    result.update(stage4_inputs=initial,reused_commit=PR23,reused_source_sha256=reused,
                  exposure_preflight=dict(known_experiment_videos=known,input_reviews=len(reviews),
                                          retained_reviews=len(accepted),rejected=rejected),
                  model_weights_modified=False,test_used_for_selection=False)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training-config','stage3','checkpoint','mapping-csv','review-csv','feature-root','evidence-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--history-config',type=Path,action='append',required=True)
    args=parser.parse_args()
    result=run(args.training_config,args.stage3,args.checkpoint,args.mapping_csv,args.review_csv,
               args.feature_root,args.evidence_root,args.output,args.device,history_configs=args.history_config)
    print(json.dumps({k:result[k] for k in ('status','independent_evaluation_videos','independent_performer_count','jamo_CER','hangul_CER')}))
