"""Observed PTS, bounded recovery and honest block interpretation."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import process_crowd_blocks as blocks


def test_low_fps_preserves_every_real_frame():
    times=np.arange(31)/10
    np.testing.assert_array_equal(blocks.sampling_indices(times),np.arange(31))


def test_vfr_has_no_duplicate_source_and_keeps_endpoint():
    times=np.array([0.,.01,.07,.1,.5,.51,.7])
    selected=blocks.sampling_indices(times)
    assert len(selected)==len(set(selected)) and selected[0]==0 and selected[-1]==len(times)-1
    assert np.all(np.diff(times[selected])>0)


def test_partial_last_block_and_single_frame():
    assert blocks.block_ranges(31)==[(0,15),(15,30),(30,31)]
    np.testing.assert_array_equal(blocks.sampling_indices([0.]),[0])


@pytest.mark.parametrize('times',[[],[0.,0.],[.1,0.],[float('nan')],[-1.,0.]])
def test_invalid_source_pts_is_structural(times):
    with pytest.raises(blocks.Unrecoverable):blocks.sampling_indices(times)


def test_exactly_ten_extra_retries(monkeypatch):
    monkeypatch.setattr(blocks.time,'sleep',lambda value:None)
    calls=[]
    def transient():
        calls.append(1)
        if len(calls)<11:raise PermissionError('transient')
        return 'verified'
    events=[]
    assert blocks.retry(transient,events)==('verified',11)
    assert len(events)==10


def test_exhaustion_and_structural_no_pointless_repetition(monkeypatch):
    monkeypatch.setattr(blocks.time,'sleep',lambda value:None)
    events=[]
    def failing():raise RuntimeError('decoder')
    with pytest.raises(RuntimeError):blocks.retry(failing,events)
    assert len(events)==11
    calls=[]
    def structural():
        calls.append(1);raise blocks.Unrecoverable('missing required input')
    with pytest.raises(blocks.Unrecoverable):blocks.retry(structural)
    assert len(calls)==1


def test_atomic_journal_preserves_old_file_during_lock(tmp_path,monkeypatch):
    path=tmp_path/'state.json';blocks.atomic_json(path,{'old':True})
    original=blocks.os.replace;calls=[]
    def locked(source,target):
        calls.append(1)
        if len(calls)<3:
            assert blocks.load(path)=={'old':True}
            raise PermissionError('locked')
        return original(source,target)
    monkeypatch.setattr(blocks.os,'replace',locked);monkeypatch.setattr(blocks.time,'sleep',lambda value:None)
    blocks.atomic_json(path,{'new':True})
    assert len(calls)==3 and blocks.load(path)=={'new':True}


def test_block_readback_identity_and_mask(tmp_path):
    path=tmp_path/'block.npz';indices=np.array([1,3]);positions=np.array([0,1]);times=np.array([.1,.2])
    np.savez(path,xy=np.zeros((2,53,2),np.float32),mask=np.zeros((2,53),bool),t=times,
        source_index=indices,selected_index=positions,origin=np.zeros(2,np.uint8),context=np.ones(2,np.int64),schema=np.asarray('schema'))
    assert len(blocks.validate_block(path,indices,positions,times,'schema'))==64
    with pytest.raises(ValueError):blocks.validate_block(path,indices+1,positions,times,'schema')
    with pytest.raises(blocks.Unrecoverable):blocks.validate_block(path,indices,positions,times,'changed')


def test_geometry_keeps_cross_block_spans_and_unknown_letters():
    xy=np.zeros((31,53,2),np.float32);mask=np.ones((31,53),bool)
    for start in (11,32):
        for finger in range(5):
            for joint in range(4):xy[:,start+1+finger*4+joint]=(.3+joint*.03,.3+finger*.01)
        xy[:,start]=(.2,.2)
    result=blocks.geometry(xy,mask,np.arange(31)/15,np.arange(31),np.ones(31),16/9)
    assert result['semantic_status']=='UNCERTAIN' and not result['handshape_to_jamo_candidates']
    assert any(s['start_selected_frame']<15 and s['end_selected_frame']>16 for s in result['stable_observation_spans'])


def test_stage_attempt_budget_is_durable(tmp_path,monkeypatch):
    state={};path=tmp_path/'state.json';monkeypatch.setattr(blocks.time,'sleep',lambda value:None)
    def failing():raise RuntimeError('decoder')
    with pytest.raises(RuntimeError):blocks.stage_retry(failing,state,path,'probe')
    restored=blocks.load(path)
    assert restored['stages']['probe']['attempts']==11
    with pytest.raises(blocks.Unrecoverable):blocks.stage_retry(lambda:None,restored,path,'probe')
    assert restored['stages']['probe']['attempts']==11


@pytest.mark.parametrize('text',['123','ABC','?','\uac00 1'])
def test_unsupported_label_is_review_context_not_technical_failure(text):
    label=dict(text=text,start=0.,end=1.)
    result=blocks.label_review_context(label)
    assert result['text']==text and result['jamo'] is None
    assert result['jamo_status']=='UNSUPPORTED' and result['jamo_error']
    assert label==dict(text=text,start=0.,end=1.)


def test_supported_label_context():
    result=blocks.label_review_context(dict(text='\uac00',start=0.,end=1.))
    assert result['jamo']==['\u3131','\u314f']
    assert result['jamo_status']=='SUPPORTED'


def test_verified_source_reuse_does_not_consume_failure_budget(tmp_path,monkeypatch):
    import zlib
    source=tmp_path/'owned.mp4';source.write_bytes(b'verified fixture')
    row=dict(video_id='video',bytes=source.stat().st_size,crc=zlib.crc32(source.read_bytes()))
    state=dict(temporary_video=str(source),stages=dict(source=dict(status='SUCCESS',attempts=11)))
    def forbidden(*args):raise AssertionError('must not extract source again')
    monkeypatch.setattr(blocks,'source_file',forbidden)
    result,attempts=blocks.acquire_source(row,dict(old_failed={}),state,tmp_path/'state.json')
    assert result==(source,True) and attempts==11
    assert state['stages']['source']['attempts']==11


def test_complete_resume_reads_artifacts_without_extraction(tmp_path,monkeypatch):
    directory=tmp_path/'videos'/'video';directory.mkdir(parents=True)
    artifact=directory/'block.npz';image=directory/'review.jpg';image.write_bytes(b'previously verified image')
    np.savez(artifact,xy=np.zeros((1,53,2),np.float32),mask=np.zeros((1,53),bool),t=np.array([0.]),
        source_index=np.array([0]),selected_index=np.array([0]),origin=np.zeros(1,np.uint8),
        context=np.ones(1,np.int64),schema=np.asarray('schema'))
    before=blocks.digest(artifact)
    record=dict(status='VERIFIED_TECHNICAL',path=str(artifact),sha256=before,
        review_image=str(image),review_image_sha256=blocks.digest(image))
    blocks.atomic_json(directory/'plan.json',dict(selected=[0],times=[0.]))
    blocks.atomic_json(directory/'state.json',dict(schema_hash='schema',source_crc=1,blocks={'1':record},
        technical_status='FAILED_AFTER_RETRIES',preparation_errors=[]))
    monkeypatch.setattr(blocks,'video_summary',lambda *args:None)
    def forbidden(*args):raise AssertionError('must not acquire source or extract completed frames')
    monkeypatch.setattr(blocks,'acquire_source',forbidden)
    result=blocks.process_video(dict(video_id='video',crc=1),dict(output=str(tmp_path),schema_hash='schema'))
    assert result['technical_status']=='SUCCESS' and result['resumed_without_extraction']
    assert blocks.digest(artifact)==before


def test_final_audit_continues_after_artifact_failure(tmp_path,monkeypatch):
    rows=[dict(video_id=value,labels=[]) for value in ('bad','good','not_started')]
    for row in rows[:2]:
        directory=tmp_path/'videos'/row['video_id'];directory.mkdir(parents=True)
        image=directory/'review.jpg';image.write_bytes(b'fixture')
        blocks.atomic_json(directory/'plan.json',dict(selected=[0],times=[0.]))
        record=dict(status='VERIFIED_TECHNICAL',path=row['video_id'],sha256='hash',attempts=1,
            review_image=str(image),review_image_sha256=blocks.digest(image))
        blocks.atomic_json(directory/'state.json',dict(technical_status='SUCCESS',blocks={'1':record}))
    calls=[]
    def verify(path,*args):
        calls.append(path)
        if path=='bad':raise PermissionError('temporary lock persists')
        return 'hash'
    monkeypatch.setattr(blocks,'validate_block',verify)
    monkeypatch.setattr(blocks.time,'sleep',lambda value:None)
    totals=blocks.audit(dict(output=str(tmp_path),schema_hash='schema'),rows,verify=True)
    assert calls.count('bad')==11 and calls.count('good')==1
    assert totals['total_blocks']==2 and totals['successful_blocks']==1 and totals['failed_blocks']==1
    assert totals['technical_success_videos']==1 and totals['failed_videos']==1
    assert totals['unplanned_videos']==1 and totals['unprocessed_videos']==1
    assert totals['partition_verified'] and not totals['block_total_complete']


def test_review_catalog_preserves_human_edits(tmp_path):
    rows=[dict(video_id='video',labels=[dict(text='123',start=0.,end=1.)])]
    blocks.review_catalog(dict(output=str(tmp_path)),rows)
    review=tmp_path/'review-template.csv'
    assert 'uncertain' in review.read_text(encoding='utf-8-sig')
    review.write_text('human review must remain untouched',encoding='utf-8')
    blocks.review_catalog(dict(output=str(tmp_path)),rows)
    assert review.read_text(encoding='utf-8')=='human review must remain untouched'


def test_worker_submit_failure_is_durable_and_retried(tmp_path,monkeypatch):
    from types import SimpleNamespace
    data=tmp_path/'data';data.mkdir()
    (data/'labels_morpheme.zip').write_bytes(b'unchanged label fixture')
    source=data/'video.zip';source.write_bytes(b'unchanged source fixture')
    extractor=data/'extractor.py';extractor.write_text('# fixture',encoding='utf-8')
    fake=SimpleNamespace(__file__=str(extractor),mp=SimpleNamespace(__version__='fixture'))
    monkeypatch.setitem(sys.modules,'landmarks',fake)
    previous=tmp_path/'previous';previous.mkdir()
    blocks.atomic_json(previous/'progress.json',dict(config=dict(video_path=str(source),
        landmarks_sha256=blocks.digest(extractor),mediapipe='fixture',opencv=blocks.cv2.__version__),
        completed={},selected=[],failed={}))
    monkeypatch.setattr(blocks,'inventory',lambda settings:[dict(video_id='video')])
    monkeypatch.setattr(blocks,'audit',lambda *args,**kwargs:{'fixture':True})
    calls=[];joins=[]
    class Future:
        def result(self):return dict(video_id='video',technical_status='SUCCESS')
    class Executor:
        def __init__(self,**kwargs):pass
        def submit(self,*args):
            calls.append(1)
            if len(calls)<3:raise OSError('worker creation temporarily unavailable')
            return Future()
        def shutdown(self,**kwargs):joins.append(kwargs)
    monkeypatch.setattr(blocks,'ProcessPoolExecutor',Executor)
    monkeypatch.setattr(blocks,'wait',lambda active,**kwargs:(set(active),set()))
    output=tmp_path/'output'
    blocks.run(SimpleNamespace(previous=previous,data=data,output=output,audit_only=False,ids=None,limit=None,workers=1))
    assert len(calls)==3 and all(j['wait'] for j in joins)
    assert blocks.load(output/'worker-dispatch-state.json')=={'video':2}
    assert blocks.load(output/'source-integrity.json')['original_video_zip_unchanged']
