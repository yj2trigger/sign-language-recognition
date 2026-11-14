"""Review imagery is timestamp-grounded; semantic status stays human-only."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import build_video_label_review as review


def test_contact_sheet_uses_actual_times_and_full_decode(monkeypatch):
    class Capture:
        def __init__(self):self.index=0
        def get(self,key):
            if key==review.cv2.CAP_PROP_FRAME_COUNT:return 6
            return (self.index-1)*1000/15
        def read(self):
            if self.index==6:return False,None
            self.index+=1
            return True,np.full((100,100,3),self.index*20,np.uint8)
        def release(self):pass
    monkeypatch.setattr(review.cv2,'VideoCapture',lambda path:Capture())
    xy=np.full((6,53,2),.5,np.float32);mask=np.ones((6,53),bool);times=np.arange(6)/15
    sheet,pts=review.contact_sheet('synthetic',xy,mask,times,(178,100))
    assert sheet.shape==(800,1440,3) and np.allclose(pts,times)


def test_review_rejects_unmatched_cached_timestamp(monkeypatch):
    class Capture:
        def get(self,key):return 2 if key==review.cv2.CAP_PROP_FRAME_COUNT else 1000.
        def read(self):return True,np.zeros((20,20,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr(review.cv2,'VideoCapture',lambda path:Capture())
    with pytest.raises(ValueError,match='PTS_not_found'):
        review.contact_sheet('synthetic',np.zeros((2,53,2)),np.ones((2,53),bool),np.array([0.,.1]),(36,20))


def test_review_rejects_invalid_sequence():
    with pytest.raises(ValueError):review.contact_sheet('synthetic',np.zeros((0,53,2)),np.zeros((0,53),bool),np.zeros(0),(36,20))


def test_video_resume_crc_and_partial_preservation(tmp_path):
    import zipfile,zlib
    payload=b'owned synthetic video bytes'
    row=dict(video_id='sample',member='sample.mp4',bytes=len(payload),crc=zlib.crc32(payload))
    archive_path=tmp_path/'source.zip'
    with zipfile.ZipFile(archive_path,'w') as archive:archive.writestr(row['member'],payload)
    with zipfile.ZipFile(archive_path) as archive:
        first=review.review_video(archive,row,tmp_path)
        assert review.review_video(archive,row,tmp_path)==first
        first.write_bytes(b'partial')
        second=review.review_video(archive,row,tmp_path)
        assert first.read_bytes()==b'partial' and second!=first and second.read_bytes()==payload


def test_video_resume_preserves_unowned_file(tmp_path):
    import zipfile,zlib
    payload=b'video';row=dict(video_id='sample',member='sample.mp4',bytes=5,crc=zlib.crc32(payload))
    (tmp_path/'sample.mp4').write_bytes(b'other work')
    with zipfile.ZipFile(tmp_path/'source.zip','w') as archive:archive.writestr('sample.mp4',payload)
    with zipfile.ZipFile(tmp_path/'source.zip') as archive:
        recovered=review.review_video(archive,row,tmp_path)
    assert recovered.name!='sample.mp4' and (tmp_path/'sample.mp4').read_bytes()==b'other work'
