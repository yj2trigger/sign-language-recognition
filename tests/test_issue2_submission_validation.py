"""Independent evidence and inspection must not silently relabel observations."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inspect_issue2_semantics import hand_measures, gif_delays
from validate_issue2_submission import api_cases, api_failures, fields, signer_audit


def test_recursive_metadata_fields_include_actual_candidates():
    result=fields({'metaData':{'participant_id':'x'},'data':[{'attributes':[{'name':'sample'}]}]})
    assert 'metaData.participant_id' in result and 'data[].attributes[].name' in result


def test_numeric_probes_do_not_change_original_arrays():
    xy=np.ones((3,53,2),np.float32);mask=np.ones((3,53),bool)
    result=api_cases(xy,mask)
    assert len(result)==12 and result['long'][0].shape==(30,53,2)
    assert np.isfinite(xy).all() and mask.all()
    assert result['wrong_mask_dtype'][1].dtype==np.uint8


def test_hand_geometry_translation_scale_invariance_and_missing():
    points=np.arange(42,dtype=np.float64).reshape(21,2)/100
    mask=np.ones(21,bool)
    a=hand_measures(points,mask);b=hand_measures(points*3+7,mask)
    np.testing.assert_allclose(a['joint_angles_degrees'],b['joint_angles_degrees'],atol=1e-5)
    np.testing.assert_allclose(a['adjacent_tip_distances_palm_units'],b['adjacent_tip_distances_palm_units'])
    assert a['palm_facing_3D'].startswith('UNRESOLVED')
    missing=hand_measures(points,np.zeros(21,bool))
    assert missing['adjacent_tip_distances_palm_units']==[None]*4
    assert missing['valid_joint_count']==0


def test_zero_palm_has_no_invented_scale():
    result=hand_measures(np.zeros((21,2)),np.ones(21,bool))
    assert result['adjacent_tip_distances_palm_units']==[None]*4
    assert result['joint_angles_degrees']==[[None,None]]*5


@pytest.mark.parametrize('value',[float('nan'),float('inf'),1e300])
def test_invalid_or_unrepresentable_geometry_rejected(value):
    points=np.zeros((21,2));points[0,0]=value
    with pytest.raises(ValueError):hand_measures(points,np.ones(21,bool))


def test_mask_dtype_is_not_silently_coerced():
    with pytest.raises(ValueError):hand_measures(np.zeros((21,2)),np.ones(21,np.uint8))


def test_gif_cumulative_time_quantization_does_not_drift():
    times=np.arange(151)/15
    delays=gif_delays(times)
    assert all(d%10==0 and d>=10 for d in delays) and delays[-1]==10
    assert np.max(np.abs(np.cumsum(delays[:-1])-times[1:]*1000))<=5
    with pytest.raises(ValueError):gif_delays([0.,.001])


def test_api_failure_report_cannot_hide_gpu_or_contract_mismatch():
    cases=api_cases(np.zeros((2,53,2),np.float32),np.ones((2,53),bool))
    errors={'nan','inf','wrong_shape','wrong_xy_dtype','wrong_mask_dtype'}
    probes=[dict(case=name,result='ERROR' if name in errors else 'RETURN',output_type='str',output='') for name in cases]
    sample=dict(video_id='fixture',output_type='str',CPU_GPU_string_equal=True)
    assert api_failures([sample],probes)==[]
    sample['CPU_GPU_string_equal']=False
    assert api_failures([sample],probes)
    assert api_failures([],probes[:-1])


def test_url_proxy_is_never_promoted_to_performer_id(tmp_path):
    video='NIA_SL_FS0001_CROWD18_F'
    videos=tmp_path/'source.zip'
    with zipfile.ZipFile(videos,'w') as archive:archive.writestr('CROWD/18/'+video+'.mp4',b'fixture only')
    raw={'metaData':{'name':video+'.mp4','url':'https://example.org/1027_example0001/'+video+'.mp4',
        'duration':2.,'exportedOn':'fixture'},'data':[{'start':0.,'end':1.,'attributes':[{'name':'sample'}]}]}
    with zipfile.ZipFile(tmp_path/'labels_morpheme.zip','w') as archive:archive.writestr('sample.json',json.dumps(raw))
    previous=tmp_path/'outputs/mediapipe-ctc-performance/data-expansion-01';previous.mkdir(parents=True)
    (previous/'progress.json').write_text(json.dumps({'config':{'video_path':str(videos)}}),encoding='utf-8')
    (previous/'training-config.json').write_text(json.dumps({'splits':{'train':[{'video_id':video}],
        'validation':[{'video_id':video}],'test':[]}}),encoding='utf-8')
    blocks=tmp_path/'blocks';(blocks/'videos').mkdir(parents=True)
    result=signer_audit(tmp_path,blocks)
    assert result['url_pattern_matched']==1 and result['url_pattern_unmatched']==0
    assert not result['reliable_performer_id_available'] and result['jamo_CER'] is None
    assert result['unverified_url_prefix_split_overlap']['train:validation']==1
    assert result['video_id_overlap']['train:validation']==[video]


@pytest.mark.parametrize('reference,prediction,counts',[
    ('\uac00\ub098','\uac00\ub2e4',(1,0,0)),
    ('\uac00\ub098','\uac00',(0,1,0)),
    ('\uac00','\uac00\ub098',(0,0,1)),
    ('\u1100\u1161','\uac00',(0,0,0)),
])
def test_existing_cer_edit_types_and_nfc(reference,prediction,counts):
    from fingerspelling.cer_metrics import edit_counts
    result=edit_counts(reference,prediction)
    assert (result.substitutions,result.deletions,result.insertions)==counts


def test_jamo_units_are_not_hangul_syllable_units():
    from fingerspelling.cer_metrics import edit_counts
    assert edit_counts('\u3131\u314f','\u3131').cer==.5
    assert edit_counts('\uac00','').cer==1.
    assert edit_counts('\u3131\u314f','\uac00').cer>0


def test_cer_aggregate_is_reference_length_weighted():
    from fingerspelling.cer_metrics import edit_counts
    total=edit_counts('ABC','AB')+edit_counts('D','')
    assert total.cer==.5 and total.reference_characters==4
