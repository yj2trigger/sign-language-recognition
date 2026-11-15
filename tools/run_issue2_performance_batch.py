"""One bounded validation-only batch; reuse frozen data and existing CTC APIs."""
import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path
import shutil
import time
import unicodedata

import numpy as np
import torch

import experiment_mediapipe_performance as exp
from fingerspelling_hand_features import hand_local_features, hand_local_schema
from fingerspelling.ctc_checkpoint import checkpoint_payload


def training_data(source, config):
    # Deliberately do not load Test arrays or targets.
    for field in ('video_id', 'fs_id', 'text'):
        groups = [{unicodedata.normalize('NFC', r[field]) for r in rows}
                  for rows in config['splits'].values()]
        if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
            raise ValueError('split_metadata_leakage')
    limited = dict(config, splits={k: config['splits'][k] for k in ('train', 'validation')})
    data = exp.load_data(source, limited)
    hashes = [{exp.base.digest(source/'features'/(r['video_id']+'.npz')) for r, x, y in rows}
              for rows in data.values()]
    if hashes[0] & hashes[1]:
        raise ValueError('duplicate_npz_across_splits')
    feature_hashes = [{hashlib.sha256(x.tobytes()).hexdigest() for r, x, y in rows}
                      for rows in data.values()]
    if feature_hashes[0] & feature_hashes[1]:
        raise ValueError('duplicate_features_across_splits')
    return {k: [(r, hand_local_features(x), y) for r, x, y in rows] for k, rows in data.items()}


def fine_tune(args, name, lr, data, baseline_cer):
    directory = args.output/name
    directory.mkdir(exist_ok=False)
    exp.seed_everything(args.seed)
    saved = exp.load_checkpoint(args.checkpoint)
    model = exp.FingerspellingCTC(exp.CTCModelConfig(**saved['model_config']))
    model.load_state_dict(saved['model_state'])
    trainer = exp.CTCTrainer(model, exp.CTCTrainingConfig(seed=args.seed, learning_rate=lr),
                             device=args.device, vocabulary=saved['vocabulary'],
                             feature_schema=saved['feature_schema'])
    zero, one = np.zeros(424, np.float32), np.ones(424, np.float32)
    config = dict(name=name, seed=args.seed, epochs=20, lr=lr, weight_decay=.01,
                  initialization_sha256=exp.base.digest(args.checkpoint), optimizer='fresh AdamW',
                  implementation_sha256=exp.base.digest(__file__),
                  features='unchanged hand-local incl. delta, folded baseline affine', batch_size=8,
                  selection='validation CER then loss then earlier epoch', test_evaluated=False)
    exp.base.write(directory/'config.json', config)
    best = (baseline_cer, float('inf'), 0)
    history = []
    for epoch in range(20):
        tick = time.perf_counter()
        order = np.random.default_rng(args.seed*1000+epoch).permutation(len(data['train']))
        trainer.fit(lambda: exp.batches(data['train'], zero, one, order),
                    lambda: exp.batches(data['validation'], zero, one), epochs=1,
                    checkpoint_dir=directory/'training')
        val = exp.score(model, data['validation'], trainer.device, zero, one)
        key = (val['jamo_cer'], trainer.history[-1]['validation_loss'], epoch+1)
        # Preserve weights only if they strictly improve the original CER.
        if val['jamo_cer'] < baseline_cer and key < best:
            exp.write_checkpoint(directory/'improved.pt', checkpoint_payload(trainer, include_best=False))
            best = key
        history.append(dict(trainer.history[-1], validation=val, seconds=time.perf_counter()-tick))
        exp.base.write(directory/'history.json', history)
        print(name, epoch+1, val['jamo_cer'], flush=True)
    selected = min(history, key=lambda h: (h['validation']['jamo_cer'], h['validation_loss'], h['epoch']))
    result = dict(config=config, selected_epoch=selected['epoch'], validation=selected['validation'],
                  improved=best[2]>0, status='success', performer_independent=False,
                  checkpoint=str(directory/'improved.pt') if best[2]>0 else None)
    if best[2]>0:
        payload = exp.load_checkpoint(directory/'improved.pt')
        cpu = exp.FingerspellingReader(directory/'improved.pt', device='cpu').recognizer
        result['cpu_validation'] = exp.score(cpu.model, data['validation'], cpu.device, zero, one)
        if result['cpu_validation']['videos'] != result['validation']['videos']:
            raise RuntimeError('CPU_GPU_CER_mismatch')
        result['train'] = exp.score(cpu.model, data['train'], cpu.device, zero, one)
        result['checkpoint_sha256'] = exp.base.digest(directory/'improved.pt')
    exp.base.write(directory/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if (args.output.drive.upper() != 'D:' or args.output == args.source
            or args.source in args.output.parents or args.output in args.source.parents
            or args.output in args.checkpoint.parents):
        parser.error('separate new output on D drive required')
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    exp.seed_everything(args.seed)
    resources = dict(disk_free=shutil.disk_usage(args.output).free,
                     gpu_memory=list(torch.cuda.mem_get_info()) if args.device.startswith('cuda') else None)
    if resources['disk_free'] < 512*1024**2:
        raise RuntimeError('insufficient_disk')
    protected = {str(p): exp.base.digest(p) for p in args.source.rglob('*') if p.is_file()}
    protected[str(args.checkpoint)] = exp.base.digest(args.checkpoint)
    config = exp.base.read(args.source/'training-config.json')
    if exp.load_checkpoint(args.checkpoint)['feature_schema'] != hand_local_schema():
        raise ValueError('hand_local_checkpoint_required')
    data = training_data(args.source, config)
    reader = exp.FingerspellingReader(args.checkpoint, device=args.device).recognizer
    zero, one = np.zeros(424, np.float32), np.ones(424, np.float32)
    baseline = exp.score(reader.model, data['validation'], reader.device, zero, one)
    counts = Counter(int(i) for r, x, y in data['train'] for i in y)
    exp.base.write(args.output/'baseline.json', dict(validation=baseline, resources=resources,
        checkpoint_sha256=exp.base.digest(args.checkpoint), protected=protected,
        split_sizes={k:len(v) for k,v in data.items()}, train_token_counts=dict(counts),
        test_evaluated=False, performer_independent=False, label_semantics_verified=False))
    if baseline['D']+baseline['I']+baseline['S'] != 37 or baseline['N'] != 120:
        raise RuntimeError('baseline_not_reproduced')
    print('Baseline reproduced:', baseline['jamo_cer'], flush=True)
    # A cheap acoustic decoding calibration; no dictionary or reference-based correction.
    decoding = []
    for bias in (-.3, .3):
        calibrated = copy.deepcopy(reader.model)
        # Modify only blank logit bias in a temporary model; baseline stays intact.
        with torch.no_grad():
            calibrated.head[-1].bias[0].add_(bias)
        decoding.append(dict(blank_bias=bias, validation=exp.score(calibrated, data['validation'],
                         reader.device, zero, one)))
    exp.base.write(args.output/'decoding.json', decoding)
    results = [fine_tune(args, name, lr, data, baseline['jamo_cer'])
               for name, lr in [('finetune-lr0003', .0003), ('finetune-lr001', .001)]]
    if any(exp.base.digest(Path(p)) != sha for p, sha in protected.items()):
        raise RuntimeError('protected_artifact_changed')
    exp.base.write(args.output/'summary.json', dict(baseline=baseline, candidates=results,
        decoding=decoding, source_unchanged=True, test_evaluated=False,
        performer_independent=False, label_semantics_verified=False))


if __name__ == '__main__':
    main()
