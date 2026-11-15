"""Stage3 binding is not independent evidence; the unchanged gate stays fail-closed."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import evaluate_stage4 as adapter


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def bound(tmp_path,monkeypatch):
    config=tmp_path/'source/training-config.json';write(config,{})
    model=tmp_path/'model/improved.pt';model.parent.mkdir();model.write_bytes(b'synthetic checkpoint')
    stage3=tmp_path/'stage3'
    result=dict(config=dict(seed=1,weight_decay=.03,lr=.001),selected_epoch=13,
                checkpoint=str(model),checkpoint_sha256=adapter.digest(model))
    write(model.parent/'result.json',result)
    write(stage3/'manifest.json',dict(source=str(config.parent),protected={str(config):adapter.digest(config)}))
    write(stage3/'selection.json',dict(selected=result))
    payload=dict(epoch=13,training_config=dict(seed=1,weight_decay=.03,learning_rate=.001))
    monkeypatch.setattr(adapter,'load_checkpoint',lambda p:copy.deepcopy(payload))
    return config,stage3,model,payload


def test_binding_contains_no_signer_or_transcript_approval(bound):
    config,stage3,model,_=bound
    binding=adapter.bind_provenance(config,stage3,model)
    assert binding['inference_sha256']==adapter.digest(model)
    assert binding['config']['source_config_sha256']==adapter.digest(config)
    assert not {'performer_id','transcript_text','approved_by','review_status'} & set(binding)


@pytest.mark.parametrize('fault',['source','config','checkpoint','selected_result','epoch','seed','weight_decay','lr'])
def test_changed_binding_inputs_rejected(bound,fault):
    config,stage3,model,payload=bound
    if fault=='source':
        manifest=adapter.read(stage3/'manifest.json');manifest['source']=str(stage3);write(stage3/'manifest.json',manifest)
    elif fault=='config':config.write_text('changed',encoding='utf-8')
    elif fault=='checkpoint':model.write_bytes(b'changed')
    elif fault=='selected_result':write(model.parent/'result.json',{})
    elif fault=='epoch':payload['epoch']=12
    elif fault=='lr':payload['training_config']['learning_rate']=.002
    else:payload['training_config'][fault]=0
    with pytest.raises(ValueError):adapter.bind_provenance(config,stage3,model)


def test_pinned_gate_does_not_infer_without_verified_training_signers():
    evaluator,sources=adapter.prior_evaluator()
    train='NIA_SL_FS0001_CROWD18_F';candidate='NIA_SL_FS0002_CROWD18_F'
    config=dict(splits=dict(train=[dict(video_id=train)]),feature_hashes={train:'a'*64})
    reviews=[dict(video_id=candidate,review_status='UNCERTAIN')]
    result=evaluator.evaluate(config,[],reviews,Path('.'),Path('.'),lambda:pytest.fail('must not infer'))
    assert result['status']=='BLOCKED' and result['jamo_CER'] is None
    assert 'TRAIN_PERFORMER_MAPPING_INCOMPLETE' in result['gate']['decisions'][0]['reasons']
    assert len(sources)==3 and all(len(h)==64 for h in sources.values())


def test_empty_evidence_does_not_initialize_model():
    evaluator,_=adapter.prior_evaluator()
    result=evaluator.evaluate(dict(splits=dict(train=[]),feature_hashes={}),[],[],Path('.'),Path('.'),
                              lambda:pytest.fail('must not infer'))
    assert result['status']=='BLOCKED' and result['independent_evaluation_videos']==0


@pytest.mark.parametrize('split',['train','validation','test'])
def test_old_training_selection_and_test_videos_cannot_be_new_evaluation(tmp_path,split):
    evaluator,_=adapter.prior_evaluator()
    seen='NIA_SL_FS0001_CROWD18_F';fresh='NIA_SL_FS0002_CROWD18_F'
    history=tmp_path/'history.json';write(history,dict(splits={split:[dict(video_id=seen)]}))
    accepted,rejected,count=adapter.exposure_filter([history],[dict(video_id=seen+'_'),dict(video_id=fresh)],evaluator.canonical)
    assert count==1 and accepted==[dict(video_id=fresh)]
    assert 'PREVIOUS_EXPERIMENT_VIDEO' in rejected[0]['reasons']


def test_old_source_and_feature_aliases_rejected(tmp_path):
    evaluator,_=adapter.prior_evaluator()
    seen='NIA_SL_FS0001_CROWD18_F';fresh='NIA_SL_FS0002_CROWD18_F'
    history=tmp_path/'legacy.json'
    write(history,dict(train=[dict(video_id=seen)],feature_sha256={seen:'a'*64}))
    accepted,rejected,_=adapter.exposure_filter([history],[dict(video_id=fresh,source_video_id=seen,feature_sha256='a'*64)],evaluator.canonical)
    assert not accepted and set(rejected[0]['reasons'])=={'PREVIOUS_EXPERIMENT_SOURCE','PREVIOUS_EXPERIMENT_FEATURE'}


def test_legacy_mp4_ids_share_canonical_identity(tmp_path):
    evaluator,_=adapter.prior_evaluator()
    seen='NIA_SL_FS0001_CROWD18_F';history=tmp_path/'legacy.json'
    write(history,dict(train=[dict(video_id=seen+'.mp4')],feature_sha256='b'*64))
    accepted,rejected,count=adapter.exposure_filter([history],[dict(video_id=seen)],evaluator.canonical)
    assert count==1 and not accepted and len(rejected)==1


def test_corrupt_feature_file_rejected_before_inference(tmp_path):
    evaluator,_=adapter.prior_evaluator()
    vid='NIA_SL_FS0002_CROWD18_F';path=tmp_path/(vid+'.npz');path.write_bytes(b'not an NPZ')
    with pytest.raises(ValueError):
        evaluator.candidate_features(dict(video_id=vid,feature_sha256=adapter.digest(path)),tmp_path)
