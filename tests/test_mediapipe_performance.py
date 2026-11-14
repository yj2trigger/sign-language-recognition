"""Synthetic checks for train-only conditioning and compatible checkpoint export."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import experiment_mediapipe_performance as exp


def test_statistics_use_only_supplied_train_frames():
    train=[np.zeros((2,4),np.float32),np.full((2,4),2,np.float32)]
    before=[x.copy() for x in train]
    mean,scale=exp.affine_statistics(train)
    np.testing.assert_array_equal(mean,np.ones(4))
    np.testing.assert_array_equal(scale,np.ones(4))
    assert all(np.array_equal(a,b) for a,b in zip(train,before))



def test_constant_channels_safe_floor():
    mean,scale=exp.affine_statistics([np.ones((3,424),np.float32)])
    assert np.isfinite(scale).all() and (scale==np.float32(.05)).all()
    assert not ((np.ones((3,424))-mean)/scale).any()



@pytest.mark.parametrize('arrays',[[],[np.ones(2)],[np.zeros((0,4))],[np.full((2,4),np.nan)],
                                   [np.ones((2,4)),np.ones((2,3))],[np.full((2,4),1e200)]])
def test_invalid_statistics_rejected(arrays):
    with pytest.raises(ValueError):exp.affine_statistics(arrays)



@pytest.mark.parametrize('floor',[0,-1,float('nan'),float('inf')])
def test_invalid_floor_rejected(floor):
    with pytest.raises(ValueError):exp.affine_statistics([np.ones((2,4))],floor)



def test_folded_logits_equal_standardized_and_original_model_unchanged():
    torch.set_num_threads(1);torch.manual_seed(8)
    model=exp.FingerspellingCTC(exp.CTCModelConfig(424,52,0,hidden_size=8,dropout=0)).eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=np.random.default_rng(8)
    x=rng.normal(size=(2,7,424)).astype(np.float32)
    mean=rng.normal(size=424).astype(np.float32)
    scale=rng.uniform(.05,2,424).astype(np.float32)
    folded=exp.fold_affine(model,mean,scale)
    with torch.inference_mode():
        a,la=model(torch.from_numpy((x-mean)/scale),[7,4])
        b,lb=folded(torch.from_numpy(x),[7,4])
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    assert torch.equal(la,lb) and all(torch.equal(v,state[k]) for k,v in model.state_dict().items())



def test_fold_bad_scale_rejected():
    model=exp.FingerspellingCTC(exp.CTCModelConfig(424,52,0,hidden_size=8))
    with pytest.raises(ValueError):exp.fold_affine(model,np.zeros(424),np.zeros(424))



def test_batches_same_targets_no_padding_standardization():
    rows=[(dict(video_id=str(n)),np.full((n,424),2,np.float32),np.array([1,31],np.int64)) for n in (3,5)]
    batch=next(exp.batches(rows,np.ones(424,np.float32),np.full(424,2,np.float32)))
    assert batch['features'].shape==(2,5,424)
    assert np.all(batch['features'][0,:3]==.5) and not batch['features'][0,3:].any()
    assert batch['targets'].tolist()==[1,31,1,31] and batch['input_lengths'].tolist()==[3,5]



def test_blank_counts_and_cer_include_only_real_frames():
    class Blank(torch.nn.Module):
        def forward(self,values,lengths):
            scores=torch.zeros((*values.shape[:2],52));scores[...,0]=5
            return scores,torch.as_tensor(lengths)
    rows=[(dict(video_id=str(n),text='가'),np.zeros((n,424),np.float32),np.array([1,31],np.int64)) for n in (3,5)]
    result=exp.score(Blank(),rows,torch.device('cpu'),np.zeros(424),np.ones(424))
    assert result['blank_argmax_fraction']==1 and result['D']==4 and result['N']==4
    assert result['jamo_cer']==1 and result['mean_prediction_length']==0



def synthetic_trainer():
    torch.set_num_threads(1)
    model=exp.FingerspellingCTC(exp.CTCModelConfig(424,52,0,hidden_size=8,dropout=0))
    trainer=exp.CTCTrainer(model,device='cpu')
    trainer.epoch=1;trainer.best_loss=2.
    trainer.history=[dict(epoch=1,train_loss=3.,validation_loss=2.)]
    with torch.no_grad():trainer.model.head[1].bias[0]=.4
    return trainer



def fake_score(model,*args):
    return dict(jamo_cer=float(model.head[1].bias[0].detach()))



def test_resume_after_last_checkpoint_before_selection(tmp_path,monkeypatch):
    trainer=synthetic_trainer()
    monkeypatch.setattr(exp,'score',fake_score)
    key=exp.repair_selection(trainer,tmp_path,[],np.zeros(424),np.ones(424))
    assert key[2]==1 and exp.load_checkpoint(tmp_path/'selected.pt')['epoch']==1
    assert exp.base.read(tmp_path/'selected.json')['epoch']==1



def test_resume_recovers_better_current_epoch_and_interrupted_json(tmp_path,monkeypatch):
    trainer=synthetic_trainer();monkeypatch.setattr(exp,'score',fake_score)
    exp.repair_selection(trainer,tmp_path,[],np.zeros(424),np.ones(424))
    trainer.epoch=2;trainer.best_loss=1.
    trainer.history.append(dict(epoch=2,train_loss=2.,validation_loss=1.))
    with torch.no_grad():trainer.model.head[1].bias[0]=.1
    before=torch.get_rng_state().clone()
    exp.repair_selection(trainer,tmp_path,[],np.zeros(424),np.ones(424))
    assert torch.equal(before,torch.get_rng_state())
    assert exp.load_checkpoint(tmp_path/'selected.pt')['epoch']==2
    exp.base.write(tmp_path/'selected.json',dict(key=[.4,2.,1],epoch=1))
    exp.repair_selection(trainer,tmp_path,[],np.zeros(424),np.ones(424))
    assert exp.base.read(tmp_path/'selected.json')['epoch']==2



def test_deployment_checkpoint_cannot_resume_training(tmp_path):
    from fingerspelling.ctc_checkpoint import checkpoint_payload,write_checkpoint
    trainer=synthetic_trainer();payload=checkpoint_payload(trainer,include_best=False)
    payload.update(optimizer_state={},deployment_only=True)
    path=tmp_path/'inference.pt';write_checkpoint(path,payload)
    weights={k:v.clone() for k,v in trainer.model.state_dict().items()}
    with pytest.raises((ValueError,KeyError)):trainer.resume(path)
    assert all(torch.equal(v,weights[k]) for k,v in trainer.model.state_dict().items())
