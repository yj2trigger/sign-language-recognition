from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fingerspelling_realtime import read_segment
from audit_fingerspelling_followup import quality, saved_errors, prioritize
from test_independent_review_handoff import TRAIN, EVAL, mapping, review
from build_independent_review_handoff import eligibility


def buffer():
    return [(i / 15, np.zeros((53, 2), np.float32), np.ones(53, bool)) for i in range(4)]


def test_segment_uses_existing_buffer_contract_and_preserves_mask():
    rows = buffer()
    rows[1][2][12] = False
    def reader(xy, mask):
        assert xy.shape == (2, 53, 2)
        assert not mask[0, 12]
        return 'fixture'
    assert read_segment(rows, 1 / 15, 2 / 15, reader=reader) == 'fixture'
    assert read_segment([], 0, 1) == ''


@pytest.mark.parametrize('kind', ['nan', 'inf', 'shape', 'mask', 'clock', 'duplicate'])
def test_invalid_segment(kind):
    rows = buffer()
    if kind in ('nan', 'inf'):rows[1][1][0, 0] = float(kind)
    if kind == 'shape':rows[1] = (1 / 15, np.zeros((52, 2)), rows[1][2])
    if kind == 'mask':rows[1] = (1 / 15, rows[1][1], np.ones(53, np.uint8))
    if kind == 'clock':rows[1] = (.1, rows[1][1], rows[1][2])
    if kind == 'duplicate':rows[1] = rows[0]
    with pytest.raises(ValueError):read_segment(rows, 0, 1, reader=lambda x, m: '')


@pytest.mark.parametrize('times', [np.arange(4)/15, np.arange(4)/10, np.array([0., .04, .13, .2])])
def test_quality_preserves_vfr_and_low_fps(times):
    value = quality(np.zeros((4, 53, 2)), np.ones((4, 53), bool), times, np.arange(4))
    assert value['frames'] == 4 and value['any_complete_hand'] == 1


def test_quality_rejects_missing_frames_and_counts_no_hands():
    xy = np.zeros((2, 53, 2)); mask = np.zeros((2, 53), bool)
    assert quality(xy, mask, np.array([0., .1]), np.arange(2))['any_complete_hand'] == 0
    with pytest.raises(ValueError):quality(xy, mask, np.array([0., .1]), np.array([0, 2]))


def test_evaluation_interval_overlap_is_blocked_but_touching_is_allowed():
    first = review(); second = dict(first, video_id='NIA_SL_FS0003_CROWD18_F', start=.5, end=2.)
    mappings = [mapping(TRAIN, 'p1'), mapping(EVAL, 'p2'), mapping(second['video_id'], 'p2')]
    result = eligibility([TRAIN], mappings, [first, second])
    assert result['declared_ready_count'] == 0
    assert all('EVALUATION_SOURCE_INTERVAL_OVERLAP' in row['reasons'] for row in result['decisions'])
    second['start'] = 1.
    assert eligibility([TRAIN], mappings, [first, second])['declared_ready_count'] == 2


def test_missing_source_is_not_ready():
    row = review(); row.pop('source_video_id')
    assert eligibility([TRAIN], [mapping(TRAIN, 'p1'), mapping(EVAL, 'p2')], [row])['declared_ready_count'] == 0


def test_error_analysis_rejects_changed_original_features(tmp_path):
    models = tmp_path / 'models'; features = tmp_path / 'features'
    features.mkdir(); (models / 'expanded-base').mkdir(parents=True)
    (features.parent / 'training-config.json').write_text(json.dumps({'feature_hashes': {EVAL: 'a' * 64}}))
    (features / (EVAL + '.npz')).write_bytes(b'changed')
    saved = {'train': {'videos': [{'video_id': EVAL, 'prediction_jamo': ''}]}}
    (models / 'expanded-base' / 'result.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='hash mismatch'):saved_errors(models, features)


def test_error_analysis_rejects_wrong_heldout_checkpoint(tmp_path):
    models = tmp_path / 'models'; features = tmp_path / 'features'
    features.mkdir()
    (tmp_path / 'training-config.json').write_text(json.dumps({'feature_hashes': {}, 'splits': {'train': []}}))
    for name in ('expanded-base', 'expanded-hand-local'):
        folder = models / name; folder.mkdir(parents=True)
        (folder / 'result.json').write_text(json.dumps({'train': {'videos': []}}))
        (folder / 'inference.pt').write_bytes(b'fixture')
    (models / 'heldout-diagnostic.json').write_text(json.dumps({'selected': 'expanded-hand-local', 'checkpoint_sha256': 'a'*64}))
    with pytest.raises(ValueError, match='heldout checkpoint'):saved_errors(models, features)


def test_review_priority_rejects_duplicate_aliases(tmp_path):
    (tmp_path / 'blind-transcription-template.csv').write_text('video_id\n' + EVAL + '\n' + EVAL + '_\n')
    with pytest.raises(ValueError, match='duplicate'):prioritize(tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize('bounds', [(float('nan'), 1), (2, 1), (-1, 1)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):read_segment(buffer(), *bounds, reader=lambda x, m: '')
