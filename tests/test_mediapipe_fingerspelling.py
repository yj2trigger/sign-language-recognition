"""Synthetic contract tests; no AI-Hub data or GPU learning required."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from fingerspelling_runtime import features, vocabulary, feature_schema, render, FingerspellingReader, load_reader
from fingerspelling.ctc_targets import TargetEncoder
from fingerspelling.ctc_model import FingerspellingCTC, CTCModelConfig
from fingerspelling.ctc_training import CTCTrainer, CTCTrainingConfig
from fingerspelling.ctc_checkpoint import checkpoint_payload, write_checkpoint
from fingerspelling_video import timestamp_indices
import json
import zipfile
import zlib


def sequence(n=16):
    rng=np.random.default_rng(11)
    xy=rng.uniform(.1,.9,(n,53,2)).astype(np.float32)
    xy[:,5]=(.3,.4)
    xy[:,6]=(.7,.4)
    return xy,np.ones((n,53),bool)



@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    model=FingerspellingCTC(CTCModelConfig(input_size=424,num_classes=52,blank_id=0,hidden_size=8))
    v=vocabulary()
    trainer=CTCTrainer(model,CTCTrainingConfig(),vocabulary=dict(token_ids=dict(v.token_ids),blank_id=0),
                       feature_schema=feature_schema())
    path=tmp_path/'fixture.pt'
    write_checkpoint(path,checkpoint_payload(trainer))
    return path



def test_features_shape_and_time_preservation():
    x,m=sequence(101)
    original=x.copy()
    y=features(x,m)
    assert y.shape==(101,424) and y.dtype==np.float32 and np.isfinite(y).all()
    assert np.array_equal(x,original)
    assert np.array_equal(y.reshape(101,53,8)[...,2].astype(bool),m)



def test_translation_scale_invariance():
    x,m=sequence()
    np.testing.assert_allclose(features(x,m),features(x*2+.25,m),atol=2e-6)



def test_wrist_motion_and_missing_mask():
    x,m=sequence()
    x[:,11,0]=np.linspace(.1,.8,len(x))
    m[4:6,12]=False
    y=features(x,m).reshape(len(x),53,8)
    assert np.any(y[1:,11,5]!=0)
    assert not y[4:6,12,2].any() and y[4:6,12,3].all()



@pytest.mark.parametrize('side',[11,32])
def test_one_hand(side):
    x,m=sequence()
    m[:,side:side+21]=False
    assert np.isfinite(features(x,m)).all()



@pytest.mark.parametrize('bad',[float('nan'),float('inf'),1e300])
def test_nonfinite_extreme_rejected(bad):
    x,m=sequence()
    x=x.astype(np.float64)
    x[0,0,0]=bad
    m[0,0]=False
    with pytest.raises(ValueError):features(x,m)



def test_no_shoulders_and_no_hands():
    x,m=sequence()
    m[:,[5,6]]=False
    with pytest.raises(ValueError):features(x,m)
    m[:]=True
    m[:,11:]=False
    with pytest.raises(ValueError):features(x,m)



@pytest.mark.parametrize('text',['한글','가나','까','괜','뛰','강강'])
def test_existing_jamo_roundtrip(text):
    encoded=TargetEncoder(vocabulary()).encode(text,input_length=100)
    assert render(encoded.ids)==text



@pytest.mark.parametrize('text',['','A','1','가!'])
def test_target_rejects_unsupported(text):
    with pytest.raises(ValueError):TargetEncoder(vocabulary()).encode(text)



def test_ctc_repeat_minimum():
    target=TargetEncoder(vocabulary()).encode('각가')
    with pytest.raises(ValueError):
        TargetEncoder(vocabulary()).encode('각가',input_length=target.minimum_input_length-1)



def test_reader_cache_reload_and_empty(checkpoint):
    reader=load_reader(str(checkpoint))
    assert reader is load_reader(str(checkpoint))
    x,m=sequence()
    assert isinstance(reader(x,m),str)
    assert reader(x,m)==FingerspellingReader(checkpoint)(x,m)
    assert reader(np.zeros((0,53,2)),np.zeros((0,53),bool))==''
    assert reader(x[:1],m[:1])==''
    m[:,11:]=False
    assert reader(x,m)==''



@pytest.mark.parametrize('kind',['shape','mask','nan'])
def test_reader_bad_input(checkpoint,kind):
    x,m=sequence()
    if kind=='shape':x=x[:,:52]
    if kind=='mask':m=m.astype(np.float32)
    if kind=='nan':x[0,1,0]=np.nan
    with pytest.raises(ValueError):FingerspellingReader(checkpoint)(x,m)



@pytest.mark.parametrize('field',['feature_schema','vocabulary','model_config'])
def test_incompatible_checkpoint(checkpoint,field):
    payload=torch.load(checkpoint,weights_only=True)
    if field=='feature_schema':payload[field]['fps']=30
    if field=='vocabulary':payload[field]['blank_id']=1
    if field=='model_config':payload[field]['input_size']=138
    wrong=checkpoint.with_name('wrong.pt')
    write_checkpoint(wrong,payload)
    with pytest.raises((ValueError,RuntimeError)):FingerspellingReader(wrong)



def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):FingerspellingReader(tmp_path/'absent.pt')



def test_public_function_uses_configured_checkpoint(checkpoint,monkeypatch):
    from fingerspelling_runtime import read_fingerspelling
    monkeypatch.setenv('FINGERSPELLING_CHECKPOINT',str(checkpoint))
    x,m=sequence()
    assert read_fingerspelling(x,m)==FingerspellingReader(checkpoint)(x,m)



def test_public_function_requires_configuration(monkeypatch):
    from fingerspelling_runtime import read_fingerspelling
    monkeypatch.delenv('FINGERSPELLING_CHECKPOINT',raising=False)
    with pytest.raises(RuntimeError,match='FINGERSPELLING_CHECKPOINT'):
        read_fingerspelling(*sequence())
