"""Temporal augmentation must preserve labels, ordering, endpoints and CTC length."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_issue2_stage2 as stage


@pytest.mark.parametrize('factor',[.9,1.,1.1])
def test_temporal_order_endpoints_and_repeated_ctc_target(factor):
    target=np.array([1,1,2,2,3],np.int64)
    ids=stage.resample_indices(7,target,factor)
    assert len(ids)>=7 and ids[0]==0 and ids[-1]==6
    assert np.all(np.diff(ids)>=0) and ids.dtype==np.int64


@pytest.mark.parametrize('factor',[.8,1.2,float('nan')])
def test_invalid_speed_rejected(factor):
    with pytest.raises(ValueError):stage.resample_indices(10,np.array([1]),factor)


def test_augmentation_recomputes_features_is_repeatable_and_does_not_mutate(monkeypatch):
    augment=object.__new__(stage.TemporalAugmentation)
    xy=np.arange(20*53*2,dtype=np.float32).reshape(20,53,2)
    mask=np.ones((20,53),bool)
    augment.raw={'train':(xy,mask)}
    rows=[({'video_id':'train'},np.zeros((20,424),np.float32),np.array([1,1,2],np.int64))]
    seen=[]
    def features(x,m):
        seen.append((x.copy(),m.copy()))
        return np.repeat(x[:,0,:1],424,axis=1)
    monkeypatch.setattr(stage,'raw_features',features)
    monkeypatch.setattr(stage.previous,'hand_local_features',lambda x:x)
    a=augment(rows,0,42);b=augment(rows,0,42)
    np.testing.assert_array_equal(a[0][1],b[0][1])
    np.testing.assert_array_equal(a[0][2],rows[0][2])
    np.testing.assert_array_equal(xy,np.arange(20*53*2,dtype=np.float32).reshape(20,53,2))
    assert len(seen)==2 and not rows[0][1].any()
    assert a[0][1].dtype==np.float32 and a[0][1].shape[1]==424
