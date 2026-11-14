import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from evaluate_independent_cer import evaluate, evidence_file, run
from read_independent_handshapes import hypothesis, read_sequence, compare_labels
from process_crowd_blocks import digest
from test_independent_review_handoff import TRAIN, EVAL, mapping, review


@pytest.fixture
def case(tmp_path):
    features = tmp_path / 'features'; features.mkdir()
    evidence = tmp_path / 'proof.txt'; evidence.write_text('synthetic only', encoding='utf-8')
    xy = np.zeros((10, 53, 2), np.float32)
    xy[:, 5] = [.4, .3]; xy[:, 6] = [.6, .3]
    mask = np.ones((10, 53), bool)
    path = features / (EVAL + '.npz')
    np.savez(path, xy=xy, mask=mask, t=np.arange(10) / 15)
    mappings = [mapping(TRAIN, 'synthetic-train'), mapping(EVAL, 'synthetic-eval')]
    for row in mappings:
        row.update(mapping_source=evidence.name, evidence_sha256=digest(evidence))
    row = review()
    row.update(start=0., end=.6, transcript_text='가', evidence_source=evidence.name,
               evidence_sha256=digest(evidence), feature_sha256=digest(path))
    config = {'splits': {'train': [{'video_id': TRAIN}]}, 'feature_hashes': {TRAIN: 'a' * 64}}
    return dict(config=config, mappings=mappings, reviews=[row], feature_root=features, evidence_root=tmp_path)


def forbidden_factory():
    raise AssertionError('inference must not run for ineligible data')


def test_synthetic_disjoint_performers_run_existing_cer(case):
    result = evaluate(**case, predictor_factory=lambda: lambda x, m, r: ('ㄱㅓ', '거', 10))
    assert result['independent_evaluation_videos'] == 1
    assert result['independent_performer_count'] == 1
    assert result['jamo_CER'] == .5 and result['hangul_CER'] == 1.
    assert result['jamo_counts']['substitutions'] == 1
    assert result['exact_match'] == 0


def test_partial_missing_frame_keeps_observation_mask(case):
    path = case['feature_root'] / (EVAL + '.npz')
    with np.load(path) as arrays:
        xy, mask, t = arrays['xy'].copy(), arrays['mask'].copy(), arrays['t'].copy()
    mask[1] = False
    np.savez(path, xy=xy, mask=mask, t=t)
    case['reviews'][0]['feature_sha256'] = digest(path)
    def predict(xy, mask, reference):
        assert not mask[1].any()
        return 'ㄱㅏ', '가', 10
    assert evaluate(**case, predictor_factory=lambda: predict)['jamo_CER'] == 0


@pytest.mark.parametrize('fault', ['performer', 'unknown', 'unverified', 'model_target', 'source',
                                  'train_video', 'feature_hash', 'evidence_hash', 'interval', 'no_hands',
                                  'nan', 'mask', 'vfr', 'lowfps', 'ctc_short', 'shoulders_missing', 'shoulders_degenerate'])
def test_ineligible_real_evidence_gates_do_not_initialize_model(case, fault):
    row = case['reviews'][0]
    if fault == 'performer':case['mappings'][1]['performer_id'] = 'synthetic-train'
    if fault == 'unknown':case['mappings'][0]['performer_id'] = 'unknown'
    if fault == 'unverified':row['review_status'] = 'UNCERTAIN'
    if fault == 'model_target':row['transcript_basis'] = 'ctc_prediction'
    if fault == 'source':row['source_video_id'] = TRAIN
    if fault == 'train_video':row['video_id'] = TRAIN
    if fault == 'feature_hash':row['feature_sha256'] = 'b' * 64
    if fault == 'evidence_hash':row['evidence_sha256'] = 'b' * 64
    if fault == 'interval':row['end'] = 8.
    if fault in ('no_hands', 'nan', 'mask', 'vfr', 'lowfps', 'ctc_short', 'shoulders_missing', 'shoulders_degenerate'):
        path = case['feature_root'] / (EVAL + '.npz')
        with np.load(path) as arrays:
            xy, mask, t = arrays['xy'].copy(), arrays['mask'].copy(), arrays['t'].copy()
        if fault == 'no_hands':mask[:, 11:] = False
        if fault == 'nan':xy[0, 0, 0] = np.nan
        if fault == 'mask':mask = mask.astype(np.uint8)
        if fault == 'vfr':t[3] += .01
        if fault == 'lowfps':t = np.arange(10) / 10
        if fault == 'ctc_short':row['transcript_text'] = '가' * 20
        if fault == 'shoulders_missing':mask[:, 5:7] = False
        if fault == 'shoulders_degenerate':xy[:, 6] = xy[:, 5]
        np.savez(path, xy=xy, mask=mask, t=t)
        row['feature_sha256'] = digest(path)
    result = evaluate(**case, predictor_factory=forbidden_factory)
    assert result['status'] == 'BLOCKED' and result['jamo_CER'] is None


def test_evaluation_intervals_and_duplicate_feature_are_blocked(case):
    second = copy.deepcopy(case['reviews'][0]); second['video_id'] = 'NIA_SL_FS0003_CROWD18_F'
    case['reviews'].append(second)
    case['mappings'].append(dict(case['mappings'][1], video_id=second['video_id']))
    assert evaluate(**case, predictor_factory=forbidden_factory)['status'] == 'BLOCKED'
    second['source_video_id'] = second['video_id']
    source = case['feature_root'] / (EVAL + '.npz')
    (case['feature_root'] / (second['video_id'] + '.npz')).write_bytes(source.read_bytes())
    result = evaluate(**case, predictor_factory=forbidden_factory)
    assert all('EVALUATION_FEATURE_DUPLICATE' in d['reasons'] for d in result['gate']['decisions'])


def test_local_evidence_cannot_escape_root(case):
    row = dict(case['reviews'][0], evidence_source='../elsewhere')
    with pytest.raises(ValueError):evidence_file(case['evidence_root'], row, 'evidence_source')


def test_checkpoint_provenance_required_before_any_inference(tmp_path):
    config = tmp_path / 'config.json'; config.write_text('{}')
    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'config': {'source_config_sha256': 'a' * 64}}))
    with pytest.raises(ValueError, match='configuration'):
        run(config, result, tmp_path/'model.pt', None, None, None, None, 'cpu')


def test_label_comparison_requires_complete_independent_evidence(case):
    labels = [{'text': '가', 'start': 0., 'end': .6}]
    row = case['reviews'][0]
    assert compare_labels(labels, row, case['evidence_root'])['status'] == 'VERIFIED'
    row['transcript_text'] = '거'
    assert compare_labels(labels, row, case['evidence_root'])['status'] == 'MISMATCH'
    row['transcript_basis'] = 'ctc_prediction'
    assert compare_labels(labels, row, case['evidence_root'])['status'] == 'UNCERTAIN'


def test_label_comparison_rejects_wrong_interval_and_evidence(case):
    labels = [{'text': '가', 'start': 0., 'end': .6}]
    row = case['reviews'][0]; row['end'] = .7
    assert compare_labels(labels, row, case['evidence_root'])['status'] == 'UNCERTAIN'
    row['end'] = .6; row['evidence_sha256'] = 'c' * 64
    assert compare_labels(labels, row, case['evidence_root'])['status'] == 'UNCERTAIN'


def test_missing_label_segment_abstains(case):
    assert compare_labels([], case['reviews'][0], case['evidence_root'])['status'] == 'UNCERTAIN'


def test_missing_hand_and_degenerate_hand_abstain():
    assert hypothesis(np.zeros((21, 2)), np.zeros(21, bool))['candidates'] == []
    assert hypothesis(np.zeros((21, 2)), np.ones(21, bool))['reason'] == 'DEGENERATE_PALM'


def test_block_boundary_and_actual_vfr_times_are_preserved(monkeypatch):
    monkeypatch.setattr('read_independent_handshapes.hypothesis', lambda p, m: {'candidates': ['ㄱ'], 'reason': 'synthetic'})
    t = np.array([0., .1, .19, .3])
    result = read_sequence(np.zeros((4, 53, 2)), np.ones((4, 53), bool), t, [1, 1, 2, 2])
    assert result['holds'][0]['blocks'] == [1, 2]
    assert result['holds'][0]['end'] == .3
    assert result['verified_transcript'] is None and result['verified_jamo_count'] == 0


def test_missing_frame_gap_splits_holds(monkeypatch):
    monkeypatch.setattr('read_independent_handshapes.hypothesis', lambda p, m: {'candidates': ['ㄱ']})
    t = np.array([0., .1, 1., 1.1])
    result = read_sequence(np.zeros((4, 53, 2)), np.ones((4, 53), bool), t, [1, 1, 2, 2])
    assert result['holds'] == []


def test_tracking_reset_does_not_merge_shape_holds(monkeypatch):
    monkeypatch.setattr('read_independent_handshapes.hypothesis', lambda p, m: {'candidates': ['ㄱ']})
    result = read_sequence(np.zeros((4, 53, 2)), np.ones((4, 53), bool), np.arange(4)/10,
                           [1, 1, 2, 2], [0, 0, 1, 1])
    assert result['holds'] == []


@pytest.mark.parametrize('times', [[0., 0.], [0., float('nan')], [1., 0.]])
def test_invalid_reading_pts_rejected(times):
    with pytest.raises(ValueError):
        read_sequence(np.zeros((2, 53, 2)), np.ones((2, 53), bool), times, [1, 1])
