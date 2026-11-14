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
