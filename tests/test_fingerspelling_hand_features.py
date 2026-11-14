"""Hand geometry and metadata compatibility, separate from real-data fitting."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fingerspelling_runtime import features,feature_schema
from fingerspelling_hand_features import hand_local_features,hand_local_schema


def sequence():
    x=np.random.default_rng(17).uniform(.2,.8,(8,53,2)).astype(np.float32)
    x[:,5]=(.3,.4);x[:,6]=(.7,.4)
    hand=np.stack([np.linspace(.4,.55,21),np.linspace(.5,.35,21)],axis=1)
    x[:,11:32]=hand;x[:,32:]=hand+(.1,0)
    return x,np.ones((8,53),bool)


def test_hand_translation_invariant_fingers_but_wrist_motion_preserved():
    x,m=sequence();base=features(x,m);before=base.copy();a=hand_local_features(base).reshape(8,53,8)
    x[:,11:32,0]+=np.arange(8,dtype=np.float32)[:,None]*.01
    b=hand_local_features(features(x,m)).reshape(8,53,8)
    np.testing.assert_allclose(a[:,12:32,:2],b[:,12:32,:2],atol=3e-6)
    assert np.any(b[1:,11,5]!=0) and not np.array_equal(a[:,11,:2],b[:,11,:2])
    np.testing.assert_array_equal(base,before)


def test_one_hand_missing_keeps_other_and_original_observation_mask():
    x,m=sequence();m[:,32:]=False
    base=features(x,m).reshape(8,53,8);out=hand_local_features(base.reshape(8,424)).reshape(8,53,8)
    assert out.shape==base.shape and np.isfinite(out).all()
    assert not out[:,32:,:2].any() and not out[:,32:,4].any()
    assert np.array_equal(out[:,:,2],base[:,:,2])
    assert np.array_equal(out[:,:11],base[:,:11])


def test_missing_or_zero_palm_falls_back_to_global_wrist_only():
    x,m=sequence();x[:,11:32]=x[:,11:12]
    out=hand_local_features(features(x,m)).reshape(8,53,8)
    assert not out[:,12:32,4].any() and out[:,11,4].all() and np.isfinite(out).all()


def test_uniform_coordinate_scale_and_translation_invariance():
    x,m=sequence()
    a=hand_local_features(features(x,m))
    b=hand_local_features(features(x*.75+.1,m))
    np.testing.assert_allclose(a,b,atol=5e-6)


def test_extreme_wrist_motion_rejected_before_float32_overflow():
    values=np.zeros((2,53,8),np.float32);values[:,:,4]=1
    values[0,11,0]=np.finfo(np.float32).max
    values[1,11,0]=-np.finfo(np.float32).max
    with pytest.raises(ValueError,match='overflow'):hand_local_features(values.reshape(2,424))


@pytest.mark.parametrize('bad',[np.zeros((0,424),np.float32),np.zeros((2,425),np.float32),
                               np.full((2,424),np.nan,np.float32),np.full((2,424),np.inf,np.float32),np.zeros((2,424),np.float64)])
def test_invalid_matrix_rejected(bad):
    with pytest.raises(ValueError):hand_local_features(bad)


def test_schema_is_explicit_and_base_schema_unchanged():
    old=feature_schema();new=hand_local_schema()
    assert new['hand_local']['version']==1 and new['input_size']==424
    assert feature_schema()==old and 'hand_local' not in old


def test_new_checkpoint_loads_matching_variant(tmp_path):
    import torch
    from fingerspelling.ctc_model import FingerspellingCTC,CTCModelConfig
    from fingerspelling.ctc_training import CTCTrainer
    from fingerspelling.ctc_checkpoint import checkpoint_payload,write_checkpoint
    from fingerspelling_runtime import FingerspellingReader,vocabulary
    model=FingerspellingCTC(CTCModelConfig(424,52,0,hidden_size=8))
    trainer=CTCTrainer(model,vocabulary=dict(token_ids=dict(vocabulary().token_ids),blank_id=0),feature_schema=hand_local_schema())
    path=tmp_path/'hand.pt';write_checkpoint(path,checkpoint_payload(trainer))
    reader=FingerspellingReader(path)
    assert reader.feature_transform is hand_local_features
    x,m=sequence();assert isinstance(reader(x,m),str)
    payload=checkpoint_payload(trainer);payload['feature_schema']['hand_local']['source_sha256']='wrong'
    write_checkpoint(tmp_path/'wrong.pt',payload)
    with pytest.raises(ValueError):FingerspellingReader(tmp_path/'wrong.pt')
