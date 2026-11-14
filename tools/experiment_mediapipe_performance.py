"""Bounded validation-only tuning using immutable, already extracted features."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unicodedata

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_mediapipe_ctc as base
from fingerspelling_runtime import feature_schema, vocabulary, FingerspellingReader
from fingerspelling.ctc_checkpoint import load_checkpoint, write_checkpoint
from fingerspelling.ctc_model import FingerspellingCTC, CTCModelConfig
from fingerspelling.ctc_training import CTCTrainer, CTCTrainingConfig, seed_everything


def affine_statistics(arrays, floor=.05):
    """Fit real Train frames only; constant channels become zero, no clipping."""
    if not np.isfinite(floor) or floor<=0 or not arrays or arrays[0].ndim!=2:
        raise ValueError('invalid_standardization_input')
    width=arrays[0].shape[1]
    if any(x.ndim!=2 or x.shape[1]!=width or not len(x) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('invalid_training_features')
    count=sum(len(x) for x in arrays)
    mean=sum(x.astype(np.float64).sum(0) for x in arrays)/count
    variance=sum(np.square(x.astype(np.float64)-mean).sum(0) for x in arrays)/count
    scale=np.maximum(np.sqrt(variance),floor)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or max(np.abs(mean).max(),scale.max())>np.finfo(np.float32).max:
        raise ValueError('standardization_overflow')
    return mean.astype(np.float32),scale.astype(np.float32)


def fold_affine(model,mean,scale):
    """Fold input standardization into the existing first Linear; no new model."""
    if (mean.shape!=(model.config.input_size,) or scale.shape!=mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any()):
        raise ValueError('invalid_affine_parameters')
    result=copy.deepcopy(model).eval()
    layer=result.projection[0]
    with torch.no_grad():
        center=torch.as_tensor(mean,device=layer.weight.device)
        divisor=torch.as_tensor(scale,device=layer.weight.device)
        weight=layer.weight.clone()/divisor[None]
        layer.bias.sub_(weight@center)
        layer.weight.copy_(weight)
    return result


def load_data(source,config):
    data={}
    for split,rows in config['splits'].items():
        data[split]=[]
        for row in rows:
            path=source/'features'/(row['video_id']+'.npz')
            if base.feature_ok(path,row)!=config['feature_hashes'][row['video_id']]:
                raise ValueError('source_feature_hash_mismatch')
            with np.load(path,allow_pickle=False) as f:
                data[split].append((row,f['features'].copy(),f['target'].copy()))
    ids=[{r['video_id'] for r,x,y in data[name]} for name in data]
    if any(a&b for i,a in enumerate(ids) for b in ids[i+1:]):raise ValueError('video_leakage')
    for field in ('fs_id','text'):
        groups=[{unicodedata.normalize('NFC',r[field]) for r,x,y in data[name]} for name in data]
        if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):raise ValueError('group_leakage')
    return data


def batches(rows,mean,scale,order=None,batch_size=8):
    selected=rows if order is None else [rows[i] for i in order]
    for start in range(0,len(selected),batch_size):
        group=selected[start:start+batch_size]
        lengths=np.asarray([len(x) for r,x,y in group],np.int64)
        values=np.zeros((len(group),int(lengths.max()),424),np.float32)
        for i,(r,x,y) in enumerate(group):values[i,:len(x)]=(x-mean)/scale
        yield dict(features=values,input_lengths=lengths,targets=np.concatenate([y for r,x,y in group]),
                   target_lengths=np.asarray([len(y) for r,x,y in group],np.int64))


def score(model,rows,device,mean,scale):
    model.eval()
    videos=[]; blank=0; frames=0; sum_blank=0.; predicted_lengths=[]
    with torch.inference_mode():
        for (row,x,y),batch in zip(rows,batches(rows,mean,scale,batch_size=1)):
            logits,lengths=model(torch.as_tensor(batch['features'],device=device),batch['input_lengths'])
            values=logits[0,:int(lengths[0])].cpu()
            path=values.argmax(-1)
            blank+=int((path==0).sum()); frames+=len(path)
            sum_blank+=float(values.softmax(-1)[:,0].sum())
            ids=base.greedy_decode(values.tolist(),blank_id=0,num_classes=52)
            reference=''.join(base.tokens_for(y.tolist()));prediction=''.join(base.tokens_for(ids))
            counts=base.edit_counts(reference,prediction)
            text=base.render(ids);hangul=base.edit_counts(row['text'],text)
            predicted_lengths.append(len(ids))
            videos.append(dict(video_id=row['video_id'],D=counts.deletions,I=counts.insertions,
                S=counts.substitutions,N=len(y),prediction_jamo=prediction,
                hangul_errors=hangul.errors,hangul_N=len(row['text']),exact=text==row['text']))
    target=sum(v['N'] for v in videos)
    return dict(samples=len(rows),jamo_cer=sum(v['D']+v['I']+v['S'] for v in videos)/target,
        D=sum(v['D'] for v in videos),I=sum(v['I'] for v in videos),S=sum(v['S'] for v in videos),N=target,
        hangul_cer=sum(v['hangul_errors'] for v in videos)/sum(v['hangul_N'] for v in videos),
        exact_match=sum(v['exact'] for v in videos)/len(rows),blank_argmax_fraction=blank/frames,
        mean_p_blank=sum_blank/frames,mean_prediction_length=float(np.mean(predicted_lengths)),
        unique_predictions=len({v['prediction_jamo'] for v in videos}),videos=videos)


def repair_selection(trainer,directory,validation_rows,mean,scale):
    """The completed epoch in last.pt is authoritative, including after a crash."""
    from fingerspelling.ctc_checkpoint import checkpoint_payload
    selected_path=directory/'selected.pt'
    best=(float('inf'),)*3
    if selected_path.exists():
        saved=load_checkpoint(selected_path)
        if (not 1<=saved['epoch']<=trainer.epoch or saved['model_config']!=trainer.model.configuration()
                or saved['feature_schema']!=trainer.feature_schema or saved['vocabulary']!=trainer.vocabulary):
            raise ValueError('selected_checkpoint_contract_mismatch')
        # Constructing a comparison model must not consume resumed training RNG.
        with torch.random.fork_rng(devices=[]):
            model=FingerspellingCTC(CTCModelConfig(**saved['model_config'])).to(trainer.device)
        model.load_state_dict(saved['model_state'])
        evaluation=score(model,validation_rows,trainer.device,mean,scale)
        best=(evaluation['jamo_cer'],saved['history'][-1]['validation_loss'],saved['epoch'])
    if trainer.epoch:
        evaluation=score(trainer.model,validation_rows,trainer.device,mean,scale)
        current=(evaluation['jamo_cer'],trainer.history[-1]['validation_loss'],trainer.epoch)
        if current<best:
            write_checkpoint(selected_path,checkpoint_payload(trainer,include_best=False))
            best=current
    if selected_path.exists():
        # Rebuild metadata from validated weights, never trust an interrupted JSON write.
        base.write(directory/'selected.json',dict(key=list(best),epoch=best[2]))
    elif (directory/'selected.json').exists():
        raise ValueError('selection_metadata_without_weights')
    return best


def run_candidate(args,name,dropout,lr,standardized,data,mean,scale):
    directory=args.output/name;directory.mkdir(exist_ok=True)
    config=dict(name=name,dropout=dropout,lr=lr,standardized=standardized,epochs=args.epochs,seed=args.seed,
        source_config_sha256=base.digest(args.source/'training-config.json'),
        implementation_sha256=base.digest(__file__),batch_size=8,
        selection='minimum_validation_CER_then_validation_loss_then_earlier_epoch')
    input_schema=getattr(args,'input_schema',None) or feature_schema()
    config['input_schema']=input_schema
    if (directory/'config.json').exists() and base.read(directory/'config.json')!=config:
        raise ValueError('candidate_configuration_changed')
    if (directory/'result.json').exists():
        result=base.read(directory/'result.json')
        for artifact,key in [('inference.pt','inference_sha256'),('selected.pt','selected_sha256')]:
            if base.digest(directory/artifact)!=result.get(key):raise ValueError('completed_candidate_changed')
        FingerspellingReader(directory/'inference.pt',device='cpu')
        return result
    base.write(directory/'config.json',config)
    mean=mean if standardized else np.zeros(424,np.float32)
    scale=scale if standardized else np.ones(424,np.float32)
    base.write(directory/'affine.json',dict(mean=mean.tolist(),scale=scale.tolist(),fit_split='train'))
    seed_everything(args.seed)
    model=FingerspellingCTC(CTCModelConfig(424,52,0,hidden_size=64,dropout=dropout))
    schema=dict(input_schema,experiment_input_standardized=standardized)
    trainer=CTCTrainer(model,CTCTrainingConfig(seed=args.seed,learning_rate=lr),device='cuda:0',
        vocabulary=dict(token_ids=dict(vocabulary().token_ids),blank_id=0),feature_schema=schema)
    checkpoint=directory/'training';checkpoint.mkdir(exist_ok=True)
    if (checkpoint/'last.pt').exists():trainer.resume(checkpoint/'last.pt')
    history=base.read(directory/'history.json') if (directory/'history.json').exists() else []
    if len(history)>trainer.epoch:raise ValueError('history_checkpoint_epoch_disagree')
    for saved,existing in zip(trainer.history,history):
        if any(saved[k]!=existing[k] for k in ('epoch','train_loss','validation_loss')):
            raise ValueError('history_checkpoint_metrics_disagree')
    for saved in trainer.history[len(history):]:
        history.append(dict(saved,train_cer=None,validation_cer=None,train_blank=None,
                            validation_blank=None,seconds=None,recovered=True))
    base.write(directory/'history.json',history)
    best_key=repair_selection(trainer,directory,data['validation'],mean,scale)
    started=time.perf_counter()
    while trainer.epoch<args.epochs:
        tick=time.perf_counter()
        order=np.random.default_rng(args.seed*1000+trainer.epoch).permutation(len(data['train']))
        trainer.fit(lambda:batches(data['train'],mean,scale,order),
                    lambda:batches(data['validation'],mean,scale),epochs=1,checkpoint_dir=checkpoint)
        training=score(trainer.model,data['train'],trainer.device,mean,scale)
        validation=score(trainer.model,data['validation'],trainer.device,mean,scale)
        key=(validation['jamo_cer'],trainer.history[-1]['validation_loss'],trainer.epoch)
        # Test is never evaluated here or used for checkpoint/model selection.
        if key<best_key:
            from fingerspelling.ctc_checkpoint import checkpoint_payload
            write_checkpoint(directory/'selected.pt',checkpoint_payload(trainer,include_best=False))
            base.write(directory/'selected.json',dict(key=list(key),epoch=trainer.epoch))
            best_key=key
        history.append(dict(trainer.history[-1],train_cer=training['jamo_cer'],validation_cer=validation['jamo_cer'],
            train_blank=training['blank_argmax_fraction'],validation_blank=validation['blank_argmax_fraction'],
            seconds=time.perf_counter()-tick))
        base.write(directory/'history.json',history)
        print(json.dumps(dict(candidate=name,**history[-1])),flush=True)
    payload=load_checkpoint(directory/'selected.pt')
    selected=FingerspellingCTC(CTCModelConfig(**payload['model_config'])).to('cuda:0')
    selected.load_state_dict(payload['model_state']);selected.eval()
    result=dict(config=config,selected_epoch=payload['epoch'],
        training_seconds=sum(r['seconds'] for r in history) if all(r['seconds'] is not None for r in history) else None,
        known_training_seconds=sum(r['seconds'] for r in history if r['seconds'] is not None),
        session_seconds=time.perf_counter()-started,
        train=score(selected,data['train'],torch.device('cuda:0'),mean,scale),
        validation=score(selected,data['validation'],torch.device('cuda:0'),mean,scale),
        gpu=torch.cuda.get_device_name(0),performer_independent=False)
    folded=fold_affine(selected,mean,scale)
    comparisons=[]
    with torch.inference_mode():
        for row,x,y in data['validation']:
            a,_=selected(torch.as_tensor(((x-mean)/scale)[None],device='cuda:0'),[len(x)])
            b,_=folded(torch.as_tensor(x[None],device='cuda:0'),[len(x)])
            comparisons.append(float((a-b).abs().max()))
            if not torch.allclose(a,b,atol=2e-4,rtol=2e-4) or not torch.equal(a.argmax(-1),b.argmax(-1)):
                raise RuntimeError('affine_export_changed_predictions')
    deployment=copy.deepcopy(payload)
    deployment.update(model_state={k:v.cpu() for k,v in folded.state_dict().items()},feature_schema=input_schema,
        optimizer_state={},deployment_only=True,affine_folded=True)
    deployment.pop('best_checkpoint',None)
    write_checkpoint(directory/'inference.pt',deployment)
    reader=FingerspellingReader(directory/'inference.pt',device='cpu')
    row,x,y=data['validation'][0]
    with np.load(args.source/'features'/(row['video_id']+'.npz'),allow_pickle=False) as f:
        text=reader(f['xy'],f['mask'])
    with torch.inference_mode():
        logits,_=selected(torch.as_tensor(((x-mean)/scale)[None],device='cuda:0'),[len(x)])
        ids=base.greedy_decode(logits[0].cpu().tolist(),blank_id=0,num_classes=52)
    if text!=base.render(ids):raise RuntimeError('reader_export_mismatch')
    result.update(export_max_abs=max(comparisons),cpu_reader_verified=True,
        inference_checkpoint=str(directory/'inference.pt'),test_evaluated=False,
        inference_sha256=base.digest(directory/'inference.pt'),selected_sha256=base.digest(directory/'selected.pt'))
    base.write(directory/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if (not 1<=args.epochs<=100 or args.output.drive.upper()!='D:' or args.output==ROOT or ROOT in args.output.parents
            or args.output==args.source or args.source in args.output.parents):p.error('invalid output or epoch budget')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'temp').mkdir(exist_ok=True)
    os.environ['TEMP']=str(args.output/'temp');os.environ['TMP']=str(args.output/'temp')
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA_required')
    config=base.read(args.source/'training-config.json')
    if config['feature_schema']!=feature_schema():raise ValueError('source_schema_changed')
    protected={str(path.relative_to(args.source)):base.digest(path) for path in args.source.rglob('*') if path.is_file()}
    data=load_data(args.source,config)
    mean,scale=affine_statistics([x for row,x,y in data['train']])
    token_counts=Counter(int(i) for row,x,y in data['train'] for i in y)
    diagnostic=dict(train_token_counts=dict(token_counts),train_token_types=len(token_counts),
        validation_unseen_tokens=sum(i not in token_counts for row,x,y in data['validation'] for i in y),
        std_floor=.05,max_amplification=20,split_sizes={k:len(v) for k,v in data.items()},
        test_reserved_from_candidate_selection=True,source_sha256=protected,
        resources=dict(disk_free=shutil.disk_usage(args.output).free,gpu_free_total=list(torch.cuda.mem_get_info())))
    if diagnostic['resources']['disk_free']<128*1024**2 or diagnostic['resources']['gpu_free_total'][0]<256*1024**2:
        raise RuntimeError('insufficient_experiment_resources')
    baseline=FingerspellingReader(args.source/'checkpoints/best.pt',device='cuda:0').recognizer
    zero=np.zeros(424,np.float32);one=np.ones(424,np.float32)
    diagnostic['baseline']={split:score(baseline.model,data[split],baseline.device,zero,one) for split in ('train','validation')}
    base.write(args.output/'diagnosis.json',diagnostic)
    results=[]
    for name,dropout,lr,standardized in [('dropout-zero',0.,.001,False),('lr-three',0.,.003,False),('train-affine',0.,.003,True)]:
        results.append(run_candidate(args,name,dropout,lr,standardized,data,mean,scale))
    selected=min(results,key=lambda r:(r['validation']['jamo_cer'],r['validation']['hangul_cer'],r['config']['name']))
    baseline_cer=diagnostic['baseline']['validation']['jamo_cer']
    summary=dict(selected=selected['config']['name'],validation_improved=selected['validation']['jamo_cer']<baseline_cer,
        baseline_validation_cer=baseline_cer,candidates=[{k:v for k,v in r.items() if k not in ('train','validation')}|
            {split:{k:v for k,v in r[split].items() if k!='videos'} for split in ('train','validation')} for r in results],
        test_evaluated=False,performer_independent=False,label_semantics_verified=False)
    after={str(path.relative_to(args.source)):base.digest(path) for path in args.source.rglob('*') if path.is_file()}
    if after!=protected:raise RuntimeError('protected_source_changed')
    summary['source_unchanged']=True
    base.write(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
