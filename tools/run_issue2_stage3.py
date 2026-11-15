"""Bounded stage3: paired augmentation seeds, local diagnostics, three ablations."""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import time

import numpy as np
import torch

import run_issue2_stage2 as stage2
from fingerspelling.ctc_alignment import align_tokens
import ctc_postprocessing as decoder

exp = stage2.exp


def stats(results):
    values = [r['validation']['jamo_cer'] for r in results]
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values,ddof=1)), n=len(values))


def train(args, data, name, seed, strength=(.9,1.1), wd=.01):
    settings = SimpleNamespace(output=args.output, checkpoint=args.baseline, seed=seed, device='cuda:0')
    transform = stage2.TemporalAugmentation(args.source,data['train'],factor_range=strength)
    tick = time.perf_counter()
    result = stage2.previous.fine_tune(settings,name,.001,data,37/120,
                                      train_transform=transform,weight_decay=wd)
    result['stage3_spec'] = dict(seed=seed,strength=list(strength),wd=wd)
    result['wall_seconds'] = time.perf_counter()-tick
    exp.base.write(args.output/name/'result.json',result)
    return result


def logits_for(path, rows, device='cuda:0'):
    reader = exp.FingerspellingReader(path,device=device)
    model=reader.recognizer.model
    with torch.inference_mode():
        for row,x,y in rows:
            logits,lengths=model(torch.as_tensor(x[None],device=device),[len(x)])
            yield row,x,y,logits[0,:int(lengths[0])].cpu().numpy()


def analysis(args,data):
    errors=Counter(); substitutions=Counter(); cases=[]; margins=[]; repeats=[]
    for row,x,y,logits in logits_for(args.reference,data['validation']):
        path=logits.argmax(-1)
        ids=exp.base.greedy_decode(logits.tolist(),blank_id=0,num_classes=52)
        reference=exp.base.tokens_for(y.tolist());prediction=exp.base.tokens_for(ids)
        aligned=align_tokens(reference,prediction)
        counts=exp.base.edit_counts(''.join(reference),''.join(prediction))
        assert sum(a['op']!='M' for a in aligned)==counts.errors
        for a in aligned:
            if a['op']=='D':errors[a['expected']]+=1
            if a['op']=='S':substitutions[a['expected']+'>'+a['predicted']]+=1
        values=x.reshape(len(x),53,8)[:,11:]
        valid=values[:,:,7]>0
        speed=np.linalg.norm(values[:,:,5:7],axis=-1)
        speed_proxy=float(np.median(speed[valid])) if valid.any() else 0.
        blank=path==0
        nonblank=logits[:,1:].max(-1)
        # Label-free acoustic ambiguity: blank is only slightly ahead.
        marginal=(logits[:,0]-nonblank>=0)&(logits[:,0]-nonblank<.3)
        runs=[];previous=None
        for i,token in enumerate(path):
            if token==0:continue
            if token==previous and i and path[i-1]==0:runs.append(i)
            previous=token
        repeated_target=int(np.sum(y[1:]==y[:-1]))
        repeats.append(dict(adjacent_repeated_target=repeated_target,separated_repeated_runs=len(runs),
                            D=counts.deletions,I=counts.insertions,S=counts.substitutions))
        margins.append(float(marginal.mean()))
        cases.append(dict(video_id=row['video_id'],reference=reference,prediction=prediction,
            alignment=aligned,frames=len(x),seconds=len(x)/15,motion_proxy=speed_proxy,
            blank_fraction=float(blank.mean()),near_blank_fraction=float(marginal.mean()),
            frames_per_target=len(x)/len(y),D=counts.deletions,I=counts.insertions,S=counts.substitutions,N=len(y)))
    durations=np.array([c['frames'] for c in cases]);speeds=np.array([c['motion_proxy'] for c in cases])
    summary=dict(deleted_tokens=dict(errors),substitution_pairs=dict(substitutions),
        repeats=repeats,mean_near_blank_fraction=float(np.mean(margins)),
        caveats=['motion is a feature-derived proxy, not verified hand speed',
                 'clip duration is not an independently verified per-jamo interval',
                 'CTC alignment is an edit alignment, not a ground-truth time annotation'])
    for label,selected in [('short_half',[c for c in cases if c['frames']<=np.median(durations)]),
                           ('long_half',[c for c in cases if c['frames']>np.median(durations)]),
                           ('high_motion_half',[c for c in cases if c['motion_proxy']>=np.median(speeds)]),
                           ('low_motion_half',[c for c in cases if c['motion_proxy']<np.median(speeds)])]:
        summary[label]=dict(samples=len(selected),CER=sum(c['D']+c['I']+c['S'] for c in selected)/sum(c['N'] for c in selected))
    reference=exp.base.read(args.reference.parent/'result.json')
    summary.update(train_cer=reference['train']['jamo_cer'],validation_cer=reference['validation']['jamo_cer'],
                   official_DIS={k:reference['validation'][k] for k in ('D','I','S')})
    exp.base.write(args.output/'analysis-local.json',dict(summary=summary,cases=cases))
    return summary


def beam_score(path, rows, device='cuda:0'):
    cases=[]
    for row,x,y,logits in logits_for(path,rows,device):
        ranked=decoder.exact_ranked_beam(decoder.log_probabilities(logits),beam_width=10)
        ids=ranked[0]['ids']
        ref=''.join(exp.base.tokens_for(y.tolist()));pred=''.join(exp.base.tokens_for(ids))
        c=exp.base.edit_counts(ref,pred)
        cases.append(dict(video_id=row['video_id'],prediction_jamo=pred,D=c.deletions,I=c.insertions,S=c.substitutions,N=len(y)))
    total={k:sum(c[k] for c in cases) for k in ('D','I','S','N')}
    return dict(samples=len(rows),jamo_cer=(total['D']+total['I']+total['S'])/total['N'],**total,videos=cases)


def implementation_hashes():
    root=Path(__file__).resolve().parents[1]
    files=['tools/run_issue2_stage3.py','tools/run_issue2_stage2.py','tools/run_issue2_performance_batch.py',
           'src/ctc_postprocessing.py','src/fingerspelling/ctc_alignment.py']
    return {f:exp.base.digest(root/f) for f in files}


def check_manifest(args,manifest):
    for key in ('source','baseline','reference','stage2'):
        if str(getattr(args,key))!=manifest[key]:raise ValueError('phase_path_changed')
    if manifest['implementation']!=implementation_hashes():raise ValueError('phase_implementation_changed')
    if any(exp.base.digest(Path(p))!=h for p,h in manifest['protected'].items()):
        raise RuntimeError('protected_artifact_changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','baseline','reference','stage2','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--phase',choices=['seeds','analyze','candidates','stability','verify'],required=True)
    args=p.parse_args()
    for key in ('source','baseline','reference','stage2','output'):setattr(args,key,getattr(args,key).resolve())
    if args.output.drive.upper()!='D:' or any(args.output==v or args.output in v.parents or v in args.output.parents
          for v in (args.source,args.baseline.parent,args.stage2,args.reference.parent)):
        p.error('new separate output on D required')
    torch.set_num_threads(2);exp.seed_everything(20261114)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<512*1024**2:
        raise RuntimeError('insufficient_GPU')
    if shutil.disk_usage(args.output.parent).free<1024**3:raise RuntimeError('insufficient_disk')
    if args.phase=='seeds':
        args.output.mkdir(exist_ok=False)
        protected={str(v):exp.base.digest(v) for v in list(args.source.rglob('*'))+list(args.stage2.rglob('*'))+[args.baseline] if v.is_file()}
        manifest={key:str(getattr(args,key)) for key in ('source','baseline','reference','stage2')}
        manifest.update(protected=protected,implementation=implementation_hashes(),epochs=20,lr=.001,
                        seeds=[42,123,2026],test_evaluated=False,
                        resources=dict(disk_free=shutil.disk_usage(args.output).free,gpu_free_total=list(torch.cuda.mem_get_info())))
        exp.base.write(args.output/'manifest.json',manifest)
    manifest=exp.base.read(args.output/'manifest.json');check_manifest(args,manifest)
    if any(exp.load_checkpoint(v)['feature_schema']!=stage2.previous.hand_local_schema() for v in (args.baseline,args.reference)):
        raise ValueError('schema_mismatch')
    data=stage2.previous.training_data(args.source,exp.base.read(args.source/'training-config.json'))
    reference=exp.base.read(args.reference.parent/'result.json')
    if stage2.evaluate(args.reference,data,'cuda:0')['videos']!=reference['validation']['videos']:
        raise RuntimeError('reference_not_reproduced')
    if args.phase=='seeds':
        results=[train(args,data,f'temporal-seed-{seed}',seed) for seed in (42,123,2026)]
        unaug=exp.base.read(args.stage2/'seeds.json')
        exp.base.write(args.output/'seeds.json',dict(reference=reference,candidates=results,
            temporal_stats=stats([reference]+results),unaugmented_stats=stats([unaug['reference']]+unaug['candidates']),
            paired_CER_deltas=[a['validation']['jamo_cer']-b['validation']['jamo_cer'] for a,b in
                              zip([reference]+results,[unaug['reference']]+unaug['candidates'])]))
    elif args.phase=='analyze':
        print(json.dumps(analysis(args,data),ensure_ascii=False),flush=True)
    elif args.phase=='candidates':
        if not (args.output/'candidate-plan.json').exists():raise ValueError('analysis_based_plan_required')
        plan=exp.base.read(args.output/'candidate-plan.json')
        if plan!=dict(A=dict(strength=[.95,1.05]),B=dict(weight_decay=.03),C=dict(beam_width=10)):
            raise ValueError('bounded_candidate_plan_changed')
        results=[train(args,data,'weaker-temporal',20261114,strength=(.95,1.05)),
                 train(args,data,'decay03',20261114,wd=.03)]
        tick=time.perf_counter()
        beam=dict(name='beam10',checkpoint=str(args.reference),selected_epoch=reference['selected_epoch'],
                  validation=beam_score(args.reference,data['validation']),decoder='beam10',
                  checkpoint_sha256=exp.base.digest(args.reference),test_evaluated=False,
                  wall_seconds=time.perf_counter()-tick,status='success')
        results.append(beam)
        # On equal CER preserve the existing model; no gratuitous replacement.
        pool=[dict(reference,checkpoint=str(args.reference))]+results
        chosen=min(pool,key=lambda r:r['validation']['jamo_cer'])
        exp.base.write(args.output/'selection.json',dict(candidates=results,selected=chosen,
            improved=chosen['validation']['jamo_cer']<27/120,test_evaluated=False))
    elif args.phase=='stability':
        chosen=exp.base.read(args.output/'selection.json')['selected']
        if chosen['validation']['jamo_cer']>=27/120:
            exp.base.write(args.output/'stability.json',dict(status='no_new_improvement',existing_model_retained=True))
        elif chosen.get('decoder')=='beam10':
            seeds=exp.base.read(args.output/'seeds.json')
            metrics=[beam_score(r['checkpoint'],data['validation']) for r in seeds['candidates']]
            exp.base.write(args.output/'stability.json',dict(decoder='beam10',candidates=metrics,
                stats=stats([chosen]+[dict(validation=m) for m in metrics])))
        else:
            spec=chosen['stage3_spec'];results=[train(args,data,f'selected-seed-{seed}',seed,
                                        strength=spec['strength'],wd=spec['wd']) for seed in (42,123,2026)]
            exp.base.write(args.output/'stability.json',dict(candidates=results,stats=stats([chosen]+results)))
    else:
        chosen=exp.base.read(args.output/'selection.json')['selected']
        if chosen.get('decoder')=='beam10':
            metric=beam_score(chosen['checkpoint'],data['validation'],'cpu')
            if metric['videos']!=chosen['validation']['videos']:raise RuntimeError('beam_CPU_GPU_mismatch')
            verification=dict(CPU_GPU_predictions_equal=True,decoder_repeated=True)
        elif chosen['validation']['jamo_cer']<27/120:
            spec=chosen['stage3_spec'];repeated=train(args,data,'repeat-selected',spec['seed'],strength=spec['strength'],wd=spec['wd'])
            a,b=(exp.load_checkpoint(r['checkpoint']) for r in (chosen,repeated))
            if not all(torch.equal(v,b['model_state'][k]) for k,v in a['model_state'].items()):raise RuntimeError('weights_not_reproduced')
            if chosen['validation']['videos']!=repeated['validation']['videos']:raise RuntimeError('metrics_not_reproduced')
            verification=dict(weights_bitwise_equal=True,predictions_equal=True)
        else:
            verification=dict(existing_model_retained=True,reference_reevaluated=True)
        reader=exp.FingerspellingReader(chosen['checkpoint'],device='cpu')
        for row,x,y in data['validation']:
            with np.load(args.source/'features'/(row['video_id']+'.npz'),allow_pickle=False) as f:text=reader(f['xy'],f['mask'])
            with torch.inference_mode():scores,_=reader.recognizer.model(torch.from_numpy(x[None]),[len(x)])
            ids=exp.base.greedy_decode(scores[0].tolist(),blank_id=0,num_classes=52)
            if text!=exp.base.render(ids):raise RuntimeError('raw_API_mismatch')
        verification.update(raw_greedy_API_verified_samples=12,test_evaluated=False)
        exp.base.write(args.output/'verification.json',verification)
    check_manifest(args,manifest)
    print(json.dumps(dict(phase=args.phase,status='success')),flush=True)


if __name__=='__main__':main()
