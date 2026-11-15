"""Evaluation boundaries, pinned phase inputs, and existing alignment consistency."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_issue2_stage3 as stage


@pytest.mark.parametrize('reference,prediction',[
    ('ㄱㅏㄱㄱ','ㄱㅏㄱ'),('ㅅㅓㄴ','ㅈㅓㄴ'),('ㄱㅏ','ㄱㅏㅏ'),('','ㄱ'),('ㄱ','')])
def test_reused_alignment_counts_agree_with_existing_metric(reference,prediction):
    path=stage.align_tokens(reference,prediction)
    counts=stage.exp.base.edit_counts(reference,prediction)
    assert sum(row['op']!='M' for row in path)==counts.errors
    assert sum(row['op']=='D' for row in path)==counts.deletions
    assert sum(row['op']=='I' for row in path)==counts.insertions
    assert sum(row['op']=='S' for row in path)==counts.substitutions


def manifest_args(tmp_path):
    args=SimpleNamespace(**{k:tmp_path/k for k in ('source','baseline','reference','stage2')})
    manifest={k:str(getattr(args,k)) for k in ('source','baseline','reference','stage2')}
    manifest.update(implementation={},protected={})
    return args,manifest


@pytest.mark.parametrize('field',['source','baseline','reference','stage2'])
def test_changed_phase_path_rejected_before_reading_protected_data(tmp_path,field):
    args,manifest=manifest_args(tmp_path)
    setattr(args,field,tmp_path/'changed')
    with pytest.raises(ValueError,match='phase_path_changed'):stage.check_manifest(args,manifest)


def test_changed_implementation_rejected(tmp_path,monkeypatch):
    args,manifest=manifest_args(tmp_path)
    monkeypatch.setattr(stage,'implementation_hashes',lambda:{'trainer':'changed'})
    with pytest.raises(ValueError,match='phase_implementation_changed'):stage.check_manifest(args,manifest)


def test_changed_protected_file_rejected(tmp_path,monkeypatch):
    args,manifest=manifest_args(tmp_path)
    protected=tmp_path/'data.npz';protected.write_bytes(b'original')
    manifest['protected']={str(protected):stage.exp.base.digest(protected)}
    protected.write_bytes(b'changed')
    monkeypatch.setattr(stage,'implementation_hashes',lambda:{})
    with pytest.raises(RuntimeError,match='protected_artifact_changed'):stage.check_manifest(args,manifest)


def test_weak_augmentation_stays_within_ctc_safe_supported_range():
    y=np.array([1,1,2,2,3],np.int64)
    for factor in (.95,1.05):
        ids=stage.stage2.resample_indices(8,y,factor)
        assert len(ids)>=7 and ids[0]==0 and ids[-1]==7 and (np.diff(ids)>=0).all()
