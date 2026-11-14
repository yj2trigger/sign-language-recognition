"""Synthetic tests for selection, leakage boundaries and immutable reuse."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import expand_mediapipe_dataset as expand
base=expand.base


def row(vid,fs,text,tokens,size=100):
    return dict(video_id=vid,fs_id=fs,text=text,target_ids=tokens,bytes=size,
                start=0.,end=1.,crc=1,member=vid+'.mp4')


def baseline():
    return dict(splits=dict(train=[row('t','t','가',[1,1,1])],
        validation=[row('v','v','나',[999])],test=[row('x','x','다',[998])]))


def test_rare_selection_uses_train_only():
    source=baseline()
    rows=[row('a','a','마',[2],10),row('b','b','바',[1],1),row('c','c','사',[3],20)]
    chosen=expand.select_additions(rows,source,set(),1,17)
    assert chosen[0]['video_id']=='a'
    changed=copy.deepcopy(source);changed['splits']['validation'][0]['target_ids']=[3]*1000
    assert expand.select_additions(rows,changed,set(),1,17)==chosen


def test_existing_failed_fs_and_text_are_excluded():
    rows=[row('t','other','기',[2]),row('bad','bad','미',[2]),row('fs','v','비',[2]),
          row('label','other','나',[2]),row('ok','ok','라',[2])]
    assert [r['video_id'] for r in expand.select_additions(rows,baseline(),{'bad'},9,1)]==['ok']


def test_nfc_equivalent_heldout_text_is_excluded():
    import unicodedata
    source=baseline();source['splits']['validation'][0]['text']=unicodedata.normalize('NFD','나')
    assert not expand.select_additions([row('new','new','나',[1])],source,set(),1,1)


def test_selection_seed_reproducible_and_no_input_mutation():
    rows=[row(str(i),str(i),'가'+str(i),[2,3]) for i in range(20)]
    original=copy.deepcopy(rows)
    assert expand.select_additions(rows,baseline(),set(),8,17)==expand.select_additions(rows,baseline(),set(),8,17)
    assert rows==original


def test_quality_one_hand_passes_and_missing_shoulders_rejects():
    args=SimpleNamespace(min_hand_coverage=.25,min_shoulder_coverage=.5)
    mask=np.ones((4,53),bool);mask[:,32:]=False
    assert base.quality_check(mask,args)==(.5,1.)
    mask[:,[5,6]]=False
    with pytest.raises(ValueError,match='shoulder_coverage'):base.quality_check(mask,args)


@pytest.mark.parametrize('mask',[np.zeros((4,53),bool),np.zeros((0,53),bool),np.ones((4,53)),np.ones((4,52),bool)])
def test_quality_invalid_or_absent_hands(mask):
    with pytest.raises(ValueError):base.quality_check(mask,SimpleNamespace(min_hand_coverage=.25))


def reuse_fixture(tmp_path):
    previous=tmp_path/'old';(previous/'features').mkdir(parents=True)
    destination=tmp_path/'new';destination.mkdir()
    x=np.random.default_rng(2).uniform(.2,.8,(15,53,2)).astype(np.float32)
    x[:,5]=(.3,.4);x[:,6]=(.7,.4);mask=np.ones((15,53),bool)
    target=base.TargetEncoder(base.vocabulary()).encode('가')
    record=row('synthetic','synthetic','가',list(target.ids))
    path=previous/'features/synthetic.npz'
    np.savez_compressed(path,xy=x,mask=mask,features=base.features(x,mask),target=target.ids,t=np.arange(15)/15)
    config={key:1 for key in ('video_size','video_mtime','labels_sha256','landmarks_sha256',
                              'video_extractor_sha256','mediapipe','opencv','extraction')}
    config['schema']=base.feature_schema()
    state=dict(config=config,selected=[record],completed={'synthetic':dict(sha256=base.digest(path),member_crc_checked=True,frames=15)})
    base.write(previous/'progress.json',state)
    args=SimpleNamespace(reuse_features=previous,min_hand_coverage=.25,min_shoulder_coverage=.5)
    return args,record,destination/'synthetic.npz',config,path


def test_reuse_checks_hash_and_does_not_modify_source(tmp_path):
    args,record,dest,config,source=reuse_fixture(tmp_path);before=base.digest(source)
    result=base.reuse_feature(args,record,dest,config)
    assert result['reused_verified_features'] and base.digest(dest)==base.digest(source)==before
    assert base.reuse_feature(args,record,dest,config)['sha256']==before


def test_reuse_contract_and_target_mismatch_fail_closed(tmp_path):
    args,record,dest,config,source=reuse_fixture(tmp_path)
    changed=dict(config,video_mtime=99)
    with pytest.raises(ValueError,match='reuse_source_contract'):base.reuse_feature(args,record,dest,changed)
    record=dict(record,text='나')
    with pytest.raises(ValueError,match='reuse_label'):base.reuse_feature(args,record,dest,config)
    assert not dest.exists()


def test_corrupt_reuse_destination_is_preserved(tmp_path):
    args,record,dest,config,source=reuse_fixture(tmp_path);dest.write_bytes(b'corrupt')
    with pytest.raises((ValueError,OSError)):base.reuse_feature(args,record,dest,config)
    assert dest.read_bytes()==b'corrupt'


def test_atomic_write_retries_transient_reader_lock(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';base.write(path,dict(epoch=1));replace=base.os.replace;calls=[]
    def locked(source,destination):
        calls.append(1)
        if len(calls)<3:raise PermissionError('synthetic transient sharing violation')
        return replace(source,destination)
    monkeypatch.setattr(base.os,'replace',locked);monkeypatch.setattr(base.time,'sleep',lambda seconds:None)
    base.write(path,dict(epoch=2))
    assert len(calls)==3 and base.read(path)==dict(epoch=2)


def test_permanent_lock_preserves_old_state_and_pending_file(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';base.write(path,dict(epoch=1))
    def locked(*args):raise PermissionError('synthetic permanent denial')
    monkeypatch.setattr(base.os,'replace',locked);monkeypatch.setattr(base.time,'sleep',lambda seconds:None)
    with pytest.raises(PermissionError):base.write(path,dict(epoch=2))
    assert base.read(path)==dict(epoch=1) and base.read(path.with_suffix('.json.tmp'))==dict(epoch=2)
