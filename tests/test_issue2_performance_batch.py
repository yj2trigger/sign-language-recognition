"""Guard evaluation boundaries even when heldout arrays are unavailable."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_issue2_performance_batch as batch


def config():
    return dict(splits={k:[dict(video_id=k, fs_id=k, text=k)]
                       for k in ('train', 'validation', 'test')})


@pytest.mark.parametrize('field', ['video_id', 'fs_id', 'text'])
def test_test_metadata_overlap_rejected_before_loading(monkeypatch, tmp_path, field):
    value = config()
    value['splits']['test'][0][field] = value['splits']['train'][0][field]
    monkeypatch.setattr(batch.exp, 'load_data', lambda *a: pytest.fail('must reject before arrays'))
    with pytest.raises(ValueError, match='split_metadata_leakage'):
        batch.training_data(tmp_path, value)


def test_test_arrays_never_requested(monkeypatch, tmp_path):
    value = config()
    def load(source, limited):
        assert set(limited['splits']) == {'train', 'validation'}
        return {k:[(limited['splits'][k][0], np.full((3,424), i, np.float32), np.array([1],np.int64))]
                for i,k in enumerate(limited['splits'])}
    monkeypatch.setattr(batch.exp, 'load_data', load)
    monkeypatch.setattr(batch.exp.base, 'digest', lambda p: str(p))
    monkeypatch.setattr(batch, 'hand_local_features', lambda x:x.copy())
    assert set(batch.training_data(tmp_path, value)) == {'train', 'validation'}


def test_identical_feature_bytes_rejected(monkeypatch, tmp_path):
    value=config()
    monkeypatch.setattr(batch.exp, 'load_data', lambda *a: {
        k:[(value['splits'][k][0], np.zeros((3,424),np.float32), np.array([1],np.int64))]
        for k in ('train','validation')})
    monkeypatch.setattr(batch.exp.base, 'digest', lambda p:str(p))
    with pytest.raises(ValueError, match='duplicate_features_across_splits'):
        batch.training_data(tmp_path,value)
