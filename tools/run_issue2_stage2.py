"""Predeclared stage2: three seeds, two training ablations, one blank bias."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import torch

import run_issue2_performance_batch as previous
from fingerspelling_runtime import features as raw_features
from fingerspelling.ctc_checkpoint import checkpoint_payload

exp = previous.exp


def resample_indices(length, target, factor):
    """Monotonic uniform speed change; retain endpoints and CTC feasibility."""
    if not .9 <= factor <= 1.1 or length < 2:
        raise ValueError('invalid_temporal_resampling')
    required = len(target) + int(np.sum(target[1:] == target[:-1]))
    size = max(2, required, int(round(length*factor)))
    return np.rint(np.linspace(0, length-1, size)).astype(np.int64)


class TemporalAugmentation:
    description = dict(type='uniform speed resampling raw xy/mask', factor_range=[.9, 1.1],
                       preserves='order, endpoints, target, CTC minimum length',
                       recompute='interpolation, shoulder normalization, hand-local, delta')

    def __init__(self, source, rows, factor_range=(.9, 1.1)):
        self.factor_range = factor_range
        self.description = dict(self.description, factor_range=list(factor_range))
        self.raw = {}
        for row, x, y in rows:
            with np.load(source/'features'/(row['video_id']+'.npz'), allow_pickle=False) as f:
                self.raw[row['video_id']] = (f['xy'].copy(), f['mask'].copy())

    def __call__(self, rows, epoch, seed):
        rng = np.random.default_rng(seed*1000+epoch+100000000)
        result = []
        for row, x, y in rows:
            xy, mask = self.raw[row['video_id']]
            indices = resample_indices(len(xy), y, float(rng.uniform(*self.factor_range)))
            values = previous.hand_local_features(raw_features(xy[indices], mask[indices]))
            result.append((row, values, y))
        return result


def evaluate(path, data, device):
    reader = exp.FingerspellingReader(path, device=device)
    return exp.score(reader.recognizer.model, data['validation'], reader.recognizer.device,
                     np.zeros(424, np.float32), np.ones(424, np.float32))


def candidate(args, data, name, seed, dropout=None, augment=False):
    settings = SimpleNamespace(output=args.output, checkpoint=args.baseline, seed=seed, device='cuda:0')
    transform = TemporalAugmentation(args.source, data['train']) if augment else None
    result = previous.fine_tune(settings, name, .001, data, 37/120,
                               dropout=dropout, train_transform=transform)
    result['stage2_spec'] = dict(seed=seed, dropout=dropout, augment=augment,
                                implementation_sha256=exp.base.digest(__file__))
    exp.base.write(args.output/name/'result.json', result)
    return result


def blank_candidate(args, data, parent):
    directory = args.output/'blank-minus03'
    directory.mkdir(exist_ok=False)
    payload = exp.load_checkpoint(parent['checkpoint'])
    model = exp.FingerspellingCTC(exp.CTCModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model_state'])
    with torch.no_grad():
        model.head[-1].bias[0].sub_(.3)
    payload.update(model_state=model.state_dict(), optimizer_state={}, deployment_only=True)
    payload.pop('best_checkpoint', None)
    exp.write_checkpoint(directory/'candidate.pt', payload)
    gpu = evaluate(directory/'candidate.pt', data, 'cuda:0')
    cpu = evaluate(directory/'candidate.pt', data, 'cpu')
    if gpu['videos'] != cpu['videos']:
        raise RuntimeError('blank_CPU_GPU_mismatch')
    result = dict(name='blank-minus03', parent=parent['checkpoint'], blank_bias=-.3,
                  validation=gpu, checkpoint=str(directory/'candidate.pt'),
                  checkpoint_sha256=exp.base.digest(directory/'candidate.pt'),
                  selected_epoch=parent['selected_epoch'], parent_spec=parent.get('stage2_spec'),
                  test_evaluated=False, status='success')
    exp.base.write(directory/'result.json', result)
    return result


def verify_repeat(args, data, selected):
    if selected.get('blank_bias') is not None:
        spec = selected['parent_spec']
    else:
        spec = selected.get('stage2_spec')
    if spec is None:
        spec = dict(seed=20261114, dropout=None, augment=False)
    repeated = candidate(args, data, 'repeat-selected', **{k:spec[k] for k in ('seed','dropout','augment')})
    if selected.get('blank_bias') is not None:
        payload = exp.load_checkpoint(repeated['checkpoint'])
        payload['model_state']['head.1.bias'][0] += selected['blank_bias']
        exp.write_checkpoint(args.output/'repeat-selected/with-bias.pt', payload)
        path = args.output/'repeat-selected/with-bias.pt'
    else:
        path = Path(repeated['checkpoint'])
    a, b = exp.load_checkpoint(selected['checkpoint']), exp.load_checkpoint(path)
    equal = all(torch.equal(v, b['model_state'][k]) for k,v in a['model_state'].items())
    metric = evaluate(path, data, 'cuda:0')
    if not equal or metric['videos'] != selected['validation']['videos']:
        raise RuntimeError('selected_reproduction_failed')
    reader = exp.FingerspellingReader(selected['checkpoint'], device='cpu')
    for row,x,y in data['validation']:
        with np.load(args.source/'features'/(row['video_id']+'.npz'), allow_pickle=False) as f:
            text = reader(f['xy'], f['mask'])
        with torch.inference_mode():
            scores,_ = reader.recognizer.model(torch.from_numpy(x[None]), [len(x)])
        ids = exp.base.greedy_decode(scores[0].tolist(), blank_id=0, num_classes=52)
        if text != exp.base.render(ids):
            raise RuntimeError('raw_reader_mismatch')
    return dict(weights_bitwise_equal=equal, validation_predictions_equal=True,
                raw_reader_verified_samples=len(data['validation']), repeat_checkpoint=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'baseline', 'reference', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--phase', choices=['seeds','candidates','verify'], required=True)
    args = parser.parse_args()
    for name in ('source','baseline','reference','output'):
        setattr(args, name, getattr(args,name).resolve())
    if args.output.drive.upper() != 'D:' or any(
            args.output == p or args.output in p.parents or p in args.output.parents
            for p in (args.source, args.baseline.parent, args.reference.parent)):
        parser.error('separate D drive output required')
    torch.set_num_threads(2)
    exp.seed_everything(20261114)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_required')
    if (shutil.disk_usage(args.output.parent).free < 1024**3
            or torch.cuda.mem_get_info()[0] < 512*1024**2):
        raise RuntimeError('insufficient_resources')
    if args.phase == 'seeds':
        args.output.mkdir(exist_ok=False)
        paths = list(args.source.rglob('*')) + list(args.reference.parent.parent.rglob('*')) + [args.baseline]
        protected = {str(p): exp.base.digest(p) for p in paths if p.is_file()}
        exp.base.write(args.output/'manifest.json', dict(protected=protected,
            source=str(args.source), baseline=str(args.baseline), reference=str(args.reference),
            seed_plan=[42,123,2026], training_candidates=['dropout-.1','temporal-.9-1.1'],
            calibration=[-.3], control_seed=20261114, epochs=20, lr=.001,
            implementation_sha256=exp.base.digest(__file__),
            resources=dict(disk_free=shutil.disk_usage(args.output).free,
                           gpu_free_total=list(torch.cuda.mem_get_info())),
            torch_version=str(torch.__version__), test_evaluated=False))
    manifest = exp.base.read(args.output/'manifest.json')
    if manifest['implementation_sha256'] != exp.base.digest(__file__):
        raise RuntimeError('implementation_changed_between_phases')
    if any(exp.base.digest(Path(p)) != h for p,h in manifest['protected'].items()):
        raise RuntimeError('protected_artifact_changed')
    if any(exp.load_checkpoint(p)['feature_schema'] != previous.hand_local_schema()
           for p in (args.baseline, args.reference)):
        raise ValueError('hand_local_schema_mismatch')
    data = previous.training_data(args.source, exp.base.read(args.source/'training-config.json'))
    reference = exp.base.read(args.reference.parent/'result.json')
    if evaluate(args.reference, data, 'cuda:0')['videos'] != reference['validation']['videos']:
        raise RuntimeError('reference_not_reproduced')
    if args.phase == 'seeds':
        results = [candidate(args,data,f'seed-{seed}',seed) for seed in (42,123,2026)]
        values = [reference['validation']['jamo_cer']] + [r['validation']['jamo_cer'] for r in results]
        exp.base.write(args.output/'seeds.json', dict(reference=reference, candidates=results,
            mean=float(np.mean(values)), population_std=float(np.std(values)),
            sample_std=float(np.std(values,ddof=1)), seeds=[20261114,42,123,2026]))
    elif args.phase == 'candidates':
        seeds = exp.base.read(args.output/'seeds.json')
        results = [candidate(args,data,'dropout01',20261114,dropout=.1),
                   candidate(args,data,'temporal',20261114,augment=True)]
        pool = [reference] + seeds['candidates'] + results
        parent = min(pool, key=lambda r:(r['validation']['jamo_cer'],r['selected_epoch'],r['checkpoint']))
        if parent is reference:
            parent = dict(reference,checkpoint=str(args.reference))
        results.append(blank_candidate(args,data,parent))
        pool.append(results[-1])
        selected = min(pool,key=lambda r:(r['validation']['jamo_cer'],r['selected_epoch'],r['checkpoint']))
        if selected is reference:
            selected = dict(reference,checkpoint=str(args.reference))
        exp.base.write(args.output/'selection.json',dict(candidates=results,selected=selected,
            improved=selected['validation']['jamo_cer'] < 28/120, test_evaluated=False))
    else:
        selected = exp.base.read(args.output/'selection.json')['selected']
        verification = verify_repeat(args,data,selected)
        exp.base.write(args.output/'verification.json',verification)
    if any(exp.base.digest(Path(p)) != h for p,h in manifest['protected'].items()):
        raise RuntimeError('protected_artifact_changed')
    print(json.dumps(dict(phase=args.phase,status='success',test_evaluated=False)),flush=True)


if __name__ == '__main__':
    main()
