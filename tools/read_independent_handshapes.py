"""Limited chart-based hypotheses from saved observations, independent of CTC."""
import argparse
import json
from pathlib import Path

import numpy as np

from build_independent_review_handoff import canonical, index_unique, review_valid
from evaluate_independent_cer import evidence_file, load_csv
from fingerspelling.jamo import decompose_hangul
from inspect_issue2_semantics import hand_measures
from process_crowd_blocks import digest

REFERENCE = 'https://www.korean.go.kr/common/download.do?file_path=etcData&c_file_name=c7f3fcaa-016a-4b72-8803-b597ab5ca52b_0.pdf'


def hypothesis(points, mask):
    """Interpret only vertical single-index forms in the official chart.

    Thresholds are engineering hypotheses, never certified alphabet boundaries.
    Unimplemented letters, depth, occlusion and RGB orientation remain unresolved.
    """
    measures = hand_measures(points, mask)
    if not mask.all():
        return dict(candidates=[], reason='HAND_JOINTS_MISSING', measures=measures)
    xy = np.asarray(points, dtype=np.float64) * [16 / 9, 1]
    palm = np.linalg.norm(xy[9] - xy[0])
    if palm < 1e-6:
        return dict(candidates=[], reason='DEGENERATE_PALM', measures=measures)
    states = []
    for finger, angles in enumerate(measures['joint_angles_degrees']):
        if any(angle is None for angle in angles):
            states.append('unknown')
            continue
        pip, tip = 2 + 4 * finger, 4 + 4 * finger
        outward = np.linalg.norm(xy[tip] - xy[0]) > np.linalg.norm(xy[pip] - xy[0])
        if min(angles) >= 155 and outward:
            states.append('extended')
        elif min(angles) <= 125 and not outward:
            states.append('bent')
        else:
            states.append('unknown')
    direction = xy[8] - xy[6]
    norm = np.linalg.norm(direction)
    vertical = float(direction[1] / norm) if norm > 1e-9 else 0.0
    candidates = []
    if states[1] == 'extended' and states[2:] == ['bent'] * 3:
        if vertical > .8:
            candidates = ['ㄱ', 'ㅜ']
        elif vertical < -.8 and states[0] == 'bent':
            candidates = ['ㅣ']
    return dict(candidates=candidates, finger_states=states, index_direction_y=vertical,
                reason='PARTIAL_CHART_HYPOTHESIS' if candidates else 'OUTSIDE_SUPPORTED_RULES',
                measures=measures, outside_rule_set_not_excluded=True)


def read_sequence(xy, mask, times, blocks, contexts=None):
    times = np.asarray(times)
    contexts = np.zeros(len(times), dtype=int) if contexts is None else np.asarray(contexts)
    if xy.shape != (len(times), 53, 2) or mask.shape != (len(times), 53):
        raise ValueError('invalid observation shapes')
    if not len(times) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('invalid actual PTS')
    if contexts.shape != (len(times),) or len(blocks) != len(times):
        raise ValueError('invalid observation provenance')
    observations = []
    holds = []
    for side, offset in (('left', 11), ('right', 32)):
        active = []
        def close_hold():
            if len(active) >= 3 and active[-1]['PTS'] - active[0]['PTS'] >= .12:
                holds.append(dict(hand=side, start=active[0]['PTS'], end=active[-1]['PTS'],
                    candidates=active[0]['candidates'], observed_frames=len(active),
                    blocks=sorted({row['block'] for row in active}), status='UNCERTAIN'))
        for index, timestamp in enumerate(times):
            reading = hypothesis(xy[index, offset:offset + 21], mask[index, offset:offset + 21])
            row = dict(PTS=float(timestamp), block=int(blocks[index]), hand=side,
                       context=int(contexts[index]), **reading)
            observations.append(row)
            continuous = active and timestamp - active[-1]['PTS'] <= .25
            same = continuous and active[-1]['candidates'] == row['candidates'] and active[-1]['context'] == row['context']
            if not same:
                close_hold()
                active = []
            if row['candidates']:
                active.append(row)
        close_hold()
    return dict(holds=holds, observations=observations, verified_transcript=None,
                status='UNCERTAIN', verified_jamo_count=0,
                reason='Limited geometric hypotheses require RGB and independent complete transcription.')


def compare_labels(labels, review, evidence_root):
    """Compare only a complete independently approved transcript for this interval."""
    result = dict(status='UNCERTAIN', verified_jamo_count=0,
                  reason='Complete independent transcript and evidence are required.')
    if not review or not review_valid(review):
        return result
    if len(labels) != 1:
        return dict(result, reason='Exactly one provided label segment is required.')
    start = min(row['start'] for row in labels)
    end = max(row['end'] for row in labels)
    if abs(float(review['start']) - start) > 1e-6 or abs(float(review['end']) - end) > 1e-6:
        return dict(result, reason='Reviewed interval does not exactly match one provided label segment.')
    try:
        proof = evidence_file(evidence_root, review, 'evidence_source')
        independent = decompose_hangul(review['transcript_text'])
        provided = decompose_hangul(labels[0]['text'])
    except (ValueError, TypeError, OSError):
        return dict(result, reason='Independent evidence or Hangul transcript rejected.')
    if not independent:
        return result
    return dict(status='VERIFIED' if independent == provided else 'MISMATCH',
                verified_jamo_count=len(independent), independent_transcript=review['transcript_text'],
                reviewer=review['reviewer'], basis=review['transcript_basis'], evidence=proof,
                reason='Approved independent transcript compared with provided label in the same interval.')


def analyze(block_root, video_ids, output, reviews=(), evidence_root=None):
    video_ids = [canonical(vid) for vid in video_ids]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError('duplicate video ID')
    review_index = index_unique(reviews)
    output.mkdir(parents=True, exist_ok=False)
    result = []
    for vid in video_ids:
        folder = block_root / 'videos' / vid
        state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
        parts = []; provenance = []
        for record in sorted(state['blocks'].values(), key=lambda row: row['block']):
            if record['status'] != 'VERIFIED_TECHNICAL' or digest(record['path']) != record['sha256']:
                raise ValueError('unverified saved block')
            with np.load(record['path'], allow_pickle=False) as arrays:
                parts.append({key: arrays[key].copy() for key in ('xy', 'mask', 't', 'selected_index', 'context')})
            provenance.append(dict(block=record['block'], path=record['path'], sha256=record['sha256'],
                                   RGB=record['review_image'], RGB_sha256=record['review_image_sha256']))
            if digest(record['review_image']) != record['review_image_sha256']:
                raise ValueError('RGB evidence changed')
        combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        if not np.array_equal(combined['selected_index'], np.arange(len(combined['t']))):
            raise ValueError('saved observation gap')
        block_numbers = np.concatenate([np.full(len(part['t']), row['block'])
            for part, row in zip(parts, provenance)])
        reading = read_sequence(combined['xy'], combined['mask'], combined['t'], block_numbers, combined['context'])
        # The label is opened only after the independent hypothesis function returns.
        labels = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))['original_labels']
        comparison = compare_labels(labels, review_index.get(vid), evidence_root)
        result.append(dict(video_id=vid, frames=len(combined['t']), reading=reading,
            provided_labels=labels, label_match_verified=comparison['status']=='VERIFIED',
            label_mismatch_confirmed=comparison['status']=='MISMATCH', comparison=comparison,
            evidence=provenance))
    report = dict(reference=REFERENCE, reference_pages='printed 186-187, tables 8-9', videos=result,
                  analyzed_videos=len(result),
                  verified_videos=sum(row['label_match_verified'] for row in result),
                  mismatch_videos=sum(row['label_mismatch_confirmed'] for row in result),
                  uncertain_videos=sum(row['comparison']['status']=='UNCERTAIN' for row in result),
                  verified_jamo_count=sum(row['comparison']['verified_jamo_count'] for row in result))
    with (output / 'readings.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=Path, required=True)
    parser.add_argument('--video-id', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    args = parser.parse_args()
    if args.output.resolve().drive.upper() != 'D:':
        parser.error('D output required')
    if args.reviews and args.evidence_root is None:
        parser.error('--reviews requires --evidence-root')
    report = analyze(args.blocks, args.video_id, args.output,
                     load_csv(args.reviews) if args.reviews else (), args.evidence_root)
    print(json.dumps({key: value for key, value in report.items() if key != 'videos'}))
