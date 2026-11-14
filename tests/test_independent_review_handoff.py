import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_independent_review_handoff import eligibility, index_unique, mapping_valid, review_valid, write_csv, verified_existing_video

TRAIN='NIA_SL_FS0001_CROWD18_F'
EVAL='NIA_SL_FS0002_CROWD18_F'
SHA='a'*64


def mapping(vid,person):
    return dict(video_id=vid,performer_id=person,mapping_status='VERIFIED',mapping_method='official_metadata',
        mapping_source='synthetic provider fixture',approved_by='fixture-reviewer',evidence_sha256=SHA)


def review():
    return dict(video_id=EVAL,source_video_id=EVAL,start=0.,end=1.,review_status='VERIFIED',
        transcript_text='synthetic fixture',transcript_basis='human_independent_reading',reviewer='fixture-reviewer',
        evidence_source='fixture only',evidence_sha256=SHA,feature_sha256='b'*64,feature_contract_verified='true')


def test_no_verified_mapping_no_claimed_evaluation():
    result=eligibility([TRAIN],[],[review()])
    assert result['training_mapping_missing']==1 and result['declared_ready_count']==0
    assert result['jamo_CER'] is None and result['independent_performer_count'] is None


def test_complete_declaration_is_not_a_measured_cer():
    result=eligibility([TRAIN],[mapping(TRAIN,'p1'),mapping(EVAL,'p2')],[review()])
    assert result['declared_ready_count']==1 and result['independent_evaluation_videos']==0
    assert result['jamo_CER'] is None


@pytest.mark.parametrize('method',['url','filename','biometric'])
def test_untrusted_identity_method_is_blocked(method):
    row=mapping(EVAL,'p2');row['mapping_method']=method
    assert not mapping_valid(row)


def test_training_performer_overlap_and_duplicate_feature_blocked():
    result=eligibility([TRAIN],[mapping(TRAIN,'p1'),mapping(EVAL,'p1')],[review()],['b'*64])
    reasons=result['decisions'][0]['reasons']
    assert 'TRAIN_PERFORMER_OVERLAP' in reasons and 'TRAIN_FEATURE_DUPLICATE' in reasons


def test_source_video_alias_overlap_blocked():
    row=review();row['source_video_id']=TRAIN+'_'
    result=eligibility([TRAIN],[mapping(TRAIN,'p1'),mapping(EVAL,'p2')],[row])
    assert 'TRAIN_SOURCE_OVERLAP' in result['decisions'][0]['reasons']


def test_duplicate_mapping_or_review_is_not_silently_merged():
    with pytest.raises(ValueError):index_unique([mapping(EVAL,'p1'),mapping(EVAL+'_','p2')])


@pytest.mark.parametrize('status',['UNCERTAIN','MISMATCH','NOT_FINGERSPELLING','UNREVIEWED'])
def test_unverified_label_cannot_enter_evaluation(status):
    row=review();row['review_status']=status
    assert not review_valid(row)


def test_model_prediction_is_not_an_independent_transcript():
    row=review();row['transcript_basis']='ctc_prediction'
    assert not review_valid(row)


@pytest.mark.parametrize('start,end',[(float('nan'),1.),(0.,float('inf')),(2.,1.),(-1.,1.)])
def test_invalid_segment_is_rejected(start,end):
    row=review();row.update(start=start,end=end)
    assert not review_valid(row)


def test_unknown_status_is_rejected_and_inputs_unchanged():
    row=review();row['review_status']='SUSPECTED_MISMATCH';before=copy.deepcopy(row)
    with pytest.raises(ValueError):eligibility([TRAIN],[],[row])
    assert row==before


def test_existing_reviewer_csv_cannot_be_overwritten(tmp_path):
    path=tmp_path/'review.csv';write_csv(path,[{'x':'original'}],['x'])
    before=path.read_bytes()
    with pytest.raises(FileExistsError):write_csv(path,[{'x':'changed'}],['x'])
    assert path.read_bytes()==before


def test_performer_whitespace_cannot_bypass_overlap():
    result=eligibility([TRAIN],[mapping(TRAIN,'p1'),mapping(EVAL,' p1 ')],[review()])
    assert 'TRAIN_PERFORMER_OVERLAP' in result['decisions'][0]['reasons']


@pytest.mark.parametrize('identifier',[-1,False,' -1 ',' UNKNOWN ',' ',None])
def test_unknown_or_wrong_typed_performer_rejected(identifier):
    assert not mapping_valid(mapping(EVAL,identifier))


@pytest.mark.parametrize('field',['reviewer','evidence_source','transcript_text'])
def test_whitespace_only_transcript_evidence_rejected(field):
    row=review();row[field]='  '
    assert not review_valid(row)


def test_existing_source_link_requires_real_crc_and_never_modifies_file(tmp_path):
    import zlib
    path=tmp_path/'existing.mp4';path.write_bytes(b'synthetic only')
    original=path.read_bytes();row=dict(bytes=len(original),crc=zlib.crc32(original))
    assert verified_existing_video(path,row)
    row['crc']+=1
    assert not verified_existing_video(path,row) and path.read_bytes()==original
