"""Read-only review prioritization and analysis of saved model results."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_independent_review_handoff import canonical, index_unique, write_csv
from process_crowd_blocks import digest


def quality(xy, mask, times, indices):
    if xy.shape != (len(times), 53, 2) or mask.shape != (len(times), 53):
        raise ValueError('invalid saved feature shape')
    if mask.dtype != np.bool_ or not np.isfinite(xy).all() or not np.isfinite(times).all():
        raise ValueError('invalid saved observations')
    if not len(times) or not (np.diff(times) > 0).all():
        raise ValueError('invalid saved PTS')
    if not np.array_equal(indices, np.arange(len(times))):
        raise ValueError('missing or duplicated selected frame')
    hands = mask[:, 11:].reshape(-1, 2, 21)
    complete = hands.all(axis=2)
    return dict(frames=len(times), duration=float(times[-1] - times[0]),
                any_complete_hand=float(complete.any(axis=1).mean()),
                both_complete_hands=float(complete.all(axis=1).mean()),
                observed_joint_fraction=float(hands.mean()),
                maximum_pts_gap=float(np.diff(times).max()) if len(times) > 1 else 0.0)


def prioritize(blocks, pack, output):
    with (pack / 'blind-transcription-template.csv').open(encoding='utf-8-sig', newline='') as stream:
        reviews = list(csv.DictReader(stream))
    index_unique(reviews)
    rows = []
    failures = []
    for review in reviews:
        vid = canonical(review['video_id'])
        try:
            state = json.loads((blocks / 'videos' / vid / 'state.json').read_text(encoding='utf-8'))
            if state['technical_status'] != 'SUCCESS':
                raise ValueError('incomplete video')
            coordinates, masks, times, indices = [], [], [], []
            records = sorted(state['blocks'].values(), key=lambda row: row['block'])
            for record in records:
                if record['status'] != 'VERIFIED_TECHNICAL':
                    raise ValueError('incomplete block')
                if digest(record['path']) != record['sha256']:
                    raise ValueError('saved feature hash mismatch')
                if digest(record['review_image']) != record['review_image_sha256']:
                    raise ValueError('saved RGB hash mismatch')
                with np.load(record['path'], allow_pickle=False) as data:
                    coordinates.append(data['xy']); masks.append(data['mask'])
                    times.append(data['t']); indices.append(data['selected_index'])
            metrics = quality(np.concatenate(coordinates), np.concatenate(masks),
                              np.concatenate(times), np.concatenate(indices))
            page = pack / (review['task_id'] + '.html')
            if not page.is_file():
                raise ValueError('missing existing blind page')
            metrics.update(video_id=vid, task_id=review['task_id'], page=page.resolve().as_uri(),
                           start=review['start'], end=review['end'], review_status=review['review_status'],
                           block_count=len(records), first_pts=float(times[0][0]), last_pts=float(times[-1][-1]),
                           reason='Complete observed hands and intact PTS; human RGB visibility/motion review required')
            rows.append(metrics)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(dict(video_id=vid, error=str(exc)))
    rows.sort(key=lambda row: (-row['any_complete_hand'], -row['observed_joint_fraction'],
                               -row['both_complete_hands'], row['maximum_pts_gap'], row['video_id']))
    for rank, row in enumerate(rows, 1):
        row['priority_rank'] = rank
    write_csv(output / 'review-priority.csv', rows, list(rows[0]) if rows else ['video_id'])
    return dict(candidates=len(reviews), ranked=len(rows), failures=failures, top_20=rows[:20],
                interpretation='Quality proxies only. RGB clarity, blur, meaningful handshape and independent spelling remain human review tasks.')


def saved_errors(models, features):
    config = json.loads((features.parent / 'training-config.json').read_text(encoding='utf-8'))
    expected_hashes = config['feature_hashes']
    heldout_path = models / 'heldout-diagnostic.json'
    heldout = json.loads(heldout_path.read_text(encoding='utf-8')) if heldout_path.exists() else None
    result = {}
    for name in ('expanded-base', 'expanded-hand-local'):
        path = models / name / 'result.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        splits = {}
        for split in ('train', 'validation', 'test'):
            if split == 'test' and heldout and heldout.get('selected') == name:
                if heldout.get('checkpoint_sha256') != digest(models / name / 'inference.pt'):
                    raise ValueError('heldout checkpoint hash mismatch')
                data = heldout
            elif split in saved:
                data = saved[split]
            else:
                continue
            predictions = Counter()
            missing = []
            lengths = []
            groups = {'complete_hand': [], 'incomplete_hand': []}
            for row in data['videos']:
                predictions.update(row['prediction_jamo'])
                feature = features / (row['video_id'] + '.npz')
                if not feature.is_file():
                    missing.append(row['video_id']); continue
                if digest(feature) != expected_hashes.get(row['video_id']):
                    raise ValueError('original training feature hash mismatch')
                with np.load(feature, allow_pickle=False) as arrays:
                    coverage = arrays['mask'][:, 11:].reshape(-1, 2, 21).all(axis=2).any(axis=1).mean()
                    lengths.append(dict(video_id=row['video_id'], input_frames=len(arrays['mask']),
                        prediction_tokens=len(row['prediction_jamo']), reference_tokens=row['N'],
                        D=row['D'], I=row['I'], S=row['S'], complete_hand_fraction=float(coverage)))
                group = 'complete_hand' if coverage >= .95 else 'incomplete_hand'
                groups[group].append(row)
            splits[split] = dict(samples=len(data['videos']),
                D=sum(row['D'] for row in data['videos']), I=sum(row['I'] for row in data['videos']),
                S=sum(row['S'] for row in data['videos']), N=sum(row['N'] for row in data['videos']),
                empty_predictions=sum(not row['prediction_jamo'] for row in data['videos']),
                prediction_tokens=dict(predictions), blank_argmax_fraction=data.get('blank_argmax_fraction'),
                mean_p_blank=data.get('mean_p_blank'), mean_prediction_length=data.get('mean_prediction_length'),
                quality_groups={key: dict(samples=len(rows), N=sum(row['N'] for row in rows),
                    errors=sum(row['D'] + row['I'] + row['S'] for row in rows)) for key, rows in groups.items()},
                missing_features=missing, per_video=lengths)
            entry = splits[split]
            expected_ids = {row['video_id'] for row in config['splits'][split]}
            actual_ids = [row['video_id'] for row in data['videos']]
            if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
                raise ValueError('saved prediction split differs from original configuration')
            entry['provided_label_jamo_CER'] = (entry['D'] + entry['I'] + entry['S']) / entry['N'] if entry['N'] else None
        result[name] = dict(source_sha256=digest(path), splits=splits)
    result['interpretation'] = 'Saved provided-label reference metrics; observed test is diagnostic. No performer-independent or independently transcribed evaluation.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('blocks', 'pack', 'models', 'features', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().drive.upper() != 'D:':
        parser.error('D output required')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(review=prioritize(args.blocks, args.pack, args.output), errors=saved_errors(args.models, args.features))
    with (args.output / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(ranked=report['review']['ranked'], failures=len(report['review']['failures']))))


if __name__ == '__main__':
    main()
