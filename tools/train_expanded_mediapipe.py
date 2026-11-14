"""Compare two representations on the same expanded Train and frozen hold-outs."""
import argparse
import copy
from pathlib import Path
import json
import sys
import numpy as np

import experiment_mediapipe_performance as exp
from fingerspelling_hand_features import hand_local_features,hand_local_schema


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if not 1<=args.epochs<=80 or args.output.drive.upper()!='D:' or exp.ROOT in args.output.parents or args.output==args.source:
        p.error('invalid bounded training output')
    args.output.mkdir(parents=True,exist_ok=True)
    exp.torch.set_num_threads(2)
    if not exp.torch.cuda.is_available():raise RuntimeError('CUDA_required')
    summary=exp.base.read(args.source/'expansion-summary.json')
    if not summary['dataset_frozen'] or summary['new_features']<1:raise ValueError('expanded_dataset_not_ready')
    config=exp.base.read(args.source/'training-config.json');data=exp.load_data(args.source,config)
    protected={str(p.relative_to(args.source)):exp.base.digest(p) for p in args.source.rglob('*') if p.is_file()}
    results=[]
    for variant in ('expanded-base','expanded-hand-local'):
        prepared=data if variant=='expanded-base' else {name:[(r,hand_local_features(x),y) for r,x,y in rows] for name,rows in data.items()}
        args.input_schema=exp.feature_schema() if variant=='expanded-base' else hand_local_schema()
        mean,scale=exp.affine_statistics([x for r,x,y in prepared['train']])
        results.append(exp.run_candidate(args,variant,0.,.003,True,prepared,mean,scale))
    chosen=min(results,key=lambda r:(r['validation']['jamo_cer'],r['validation']['hangul_cer'],r['config']['name']))
    decision=dict(selected=chosen['config']['name'],selection='Validation jamo CER then Hangul CER only',
        test_not_used_for_selection=True,performer_independent=False,label_semantics_verified=False,
        split_sizes={k:len(v) for k,v in data.items()},source_config_sha256=exp.base.digest(args.source/'training-config.json'),
        candidates=[{k:v for k,v in r.items() if k not in ('train','validation')}|
            {name:{k:v for k,v in r[name].items() if k!='videos'} for name in ('train','validation')} for r in results])
    exp.base.write(args.output/'selection.json',decision)  # Lock before the diagnostic Test pass.
    if (args.output/'heldout-diagnostic.json').exists():
        heldout=exp.base.read(args.output/'heldout-diagnostic.json')
        if (heldout['selected']!=decision['selected']
                or heldout['checkpoint_sha256']!=exp.base.digest(chosen['inference_checkpoint'])):
            raise ValueError('locked_selection_or_checkpoint_changed')
    else:
        reader=exp.FingerspellingReader(chosen['inference_checkpoint'],device='cpu')
        test_data=data['test']
        if chosen['config']['name']=='expanded-hand-local':
            test_data=[(r,hand_local_features(x),y) for r,x,y in test_data]
        heldout=exp.score(reader.recognizer.model,test_data,reader.recognizer.device,np.zeros(424,np.float32),np.ones(424,np.float32))
        heldout.update(selected=decision['selected'],selection_locked_before_test=True,previously_observed_test=True,
            performer_independent=False,checkpoint_sha256=exp.base.digest(chosen['inference_checkpoint']))
        exp.base.write(args.output/'heldout-diagnostic.json',heldout)
    after={str(p.relative_to(args.source)):exp.base.digest(p) for p in args.source.rglob('*') if p.is_file()}
    if after!=protected:raise RuntimeError('frozen_dataset_changed')
    decision.update(source_unchanged=True,test_diagnostic={k:v for k,v in heldout.items() if k!='videos'})
    exp.base.write(args.output/'summary.json',decision)
    print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
