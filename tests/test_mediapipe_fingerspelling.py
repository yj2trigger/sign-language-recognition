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
from run_mediapipe_ctc import split_rows, feature_ok, candidates, checked_temporary
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



def test_grouped_split_no_fs_or_text_leak():
    rows=[dict(video_id=f'{i}-{g}',fs_id=str(i),text=f'label{i}') for i in range(20) for g in (18,19)]
    rows[4]['text']=rows[0]['text']
    splits=split_rows(rows,42)
    assert splits==split_rows(rows,42)
    assert sum(map(len,splits.values()))==len(rows)
    for field in ('fs_id','text','video_id'):
        sets=[{r[field] for r in subset} for subset in splits.values()]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])



def test_insufficient_split_groups():
    with pytest.raises(ValueError):split_rows([dict(fs_id='a',text='b')]*10,2)



def test_feature_roundtrip(tmp_path):
    x,m=sequence()
    ids=list(TargetEncoder(vocabulary()).encode('가').ids)
    path=tmp_path/'sample.npz'
    np.savez_compressed(path,xy=x,mask=m,features=features(x,m),target=ids,t=np.arange(len(x))/15)
    assert len(feature_ok(path,dict(text='가',target_ids=ids)))==64



def test_zip_alias_and_unsupported_label(tmp_path):
    labels=tmp_path/'labels.zip'
    videos=tmp_path/'videos.zip'
    with zipfile.ZipFile(labels,'w') as archive:
        for i,text in [(1,'가'),(2,'12')]:
            vid=f'NIA_SL_FS{i:04d}_CROWD18_F'
            archive.writestr(f'{vid}_morpheme.json',json.dumps(dict(metaData=dict(name=vid+'.mp4',duration=2),
                data=[dict(start=.1,end=1.9,attributes=[dict(name=text)])])))
    with zipfile.ZipFile(videos,'w') as archive:
        for i in (1,2): archive.writestr(f'CROWD/18/NIA_SL_FS{i:04d}_CROWD18_F_.mp4',b'fixture')
    rows,reasons,total,label_count=candidates(videos,labels)
    assert total==label_count==2 and len(rows)==1
    assert rows[0]['filename_alias'] and rows[0]['video_id']=='NIA_SL_FS0001_CROWD18_F'
    assert reasons==dict(digit_policy_required=1)
    assert rows[0]['semantic_status']=='unverified' and rows[0]['performer_id'] is None



def test_duplicate_alias_rejected(tmp_path):
    labels=tmp_path/'labels.zip'; videos=tmp_path/'videos.zip'
    vid='NIA_SL_FS0001_CROWD18_F'
    with zipfile.ZipFile(labels,'w') as archive:
        archive.writestr('a.json',json.dumps(dict(metaData=dict(name=vid+'.mp4',duration=2),
            data=[dict(start=0,end=1,attributes=[dict(name='가')])])))
    with zipfile.ZipFile(videos,'w') as archive:
        archive.writestr(vid+'.mp4',b'x')
        archive.writestr(vid+'_.mp4',b'x')
    with pytest.raises(ValueError,match='duplicate_mp4'):candidates(videos,labels)



@pytest.mark.parametrize('content',[b'abc',b'xy',b'xyz'])
def test_temporary_crc_size(content,tmp_path):
    path=tmp_path/'owned.mp4'
    path.write_bytes(content)
    row=dict(bytes=3,crc=zlib.crc32(b'abc'))
    if content==b'abc':checked_temporary(path,row)
    else:
        with pytest.raises(ValueError):checked_temporary(path,row)



def test_nfc_nfd_split_leakage():
    import unicodedata
    rows=[dict(video_id=str(i),fs_id=str(i),text=f'label{i}') for i in range(12)]
    rows[0]['text']='가'; rows[1]['text']=unicodedata.normalize('NFD','가')
    split=split_rows(rows,42)
    assert next(name for name,items in split.items() if rows[0] in items)==next(name for name,items in split.items() if rows[1] in items)



def test_constant_60fps_selects_every_fourth_frame():
    assert np.array_equal(timestamp_indices(np.arange(120)/60),np.arange(0,120,4))



def test_variable_rate_uses_timestamps_without_duplicate_frames():
    times=np.array([0,.033,.067,.100,.150,.183,.217,.250])
    indices=timestamp_indices(times)
    assert np.array_equal(indices,[0,2,4,6])
    assert len(set(indices))==len(indices)
    assert np.all(times[indices]>=np.arange(len(indices))/15-1e-7)



@pytest.mark.parametrize('times',[[0],[0,0],[0,.1,.05],[0,float('nan')],[0,float('inf')],[-.1,0]])
def test_invalid_video_clock_rejected(times):
    with pytest.raises(ValueError):timestamp_indices(times)
