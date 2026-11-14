"""Private original-frame contact sheets for expert review; never auto-verify labels."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys
import zipfile
import uuid

import cv2
import numpy as np
import run_mediapipe_ctc as base


def review_video(archive,row,temp):
    """Reuse only journal-owned, CRC-verified files; preserve partial/unowned files."""
    video=temp/(row['video_id']+'.mp4')
    owner=video.with_suffix('.owner.json')
    identity={k:row[k] for k in ('video_id','member','bytes','crc')}
    if video.exists():
        if owner.exists() and base.read(owner)==identity:
            try:
                base.checked_temporary(video,row)
                return video
            except (OSError,ValueError):pass
        video=temp/(row['video_id']+'-'+uuid.uuid4().hex+'.mp4')
        owner=video.with_suffix('.owner.json')
    base.write(owner,identity)
    with archive.open(row['member']) as source,video.open('xb') as target:
        shutil.copyfileobj(source,target,1024**2)
    base.checked_temporary(video,row)
    return video


def contact_sheet(video,xy,mask,times,fitted_wh):
    if (xy.shape!=(len(times),53,2) or mask.shape!=xy.shape[:2] or mask.dtype!=np.bool_
            or len(times)<2 or not np.isfinite(xy).all() or not np.isfinite(times).all()
            or not (np.diff(times)>0).all()):raise ValueError('invalid_review_feature_sequence')
    selected=np.unique(np.linspace(0,len(times)-1,6,dtype=int))
    wanted=list(selected);panels=[];observed=[]
    cap=cv2.VideoCapture(str(video));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded+=1;pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not wanted:continue
            i=wanted[0]
            if pts<float(times[i])-1e-5:continue
            if abs(pts-float(times[i]))>1e-5:raise ValueError('cached_PTS_not_found_in_original')
            wanted.pop(0);h,w=frame.shape[:2];fw,fh=fitted_wh
            points=xy[i,11:][mask[i,11:]].astype(np.float64)
            if len(points):
                points[:,0]=points[:,0]*fw-(fw-w)//2
                points[:,1]=points[:,1]*fh-(fh-h)//2
                x0,y0=np.maximum(np.floor(points.min(0)-64),0).astype(int)
                x1,y1=np.minimum(np.ceil(points.max(0)+64),(w,h)).astype(int)
                crop=frame[y0:y1,x0:x1]
                if not crop.size:crop=frame
            else:crop=frame
            factor=min(480/crop.shape[1],360/crop.shape[0])
            image=cv2.resize(crop,(max(1,round(crop.shape[1]*factor)),max(1,round(crop.shape[0]*factor))))
            panel=np.full((400,480,3),235,np.uint8);ih,iw=image.shape[:2]
            panel[:ih,:iw]=image;cv2.putText(panel,f'PTS {pts:.6f}s',(8,388),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
            panels.append(panel);observed.append(float(pts))
    finally:cap.release()
    if decoded!=expected or wanted or len(panels)!=len(selected):raise ValueError('incomplete_review_decode')
    while len(panels)<6:panels.append(np.zeros((400,480,3),np.uint8))
    return np.vstack((np.hstack(panels[:3]),np.hstack(panels[3:]))),observed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=12)
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.drive.upper()!='D:' or base.ROOT in args.output.parents or not 1<=args.count<=24:p.error('invalid private output')
    args.output.mkdir(parents=True,exist_ok=True)
    config=base.read(args.dataset/'training-config.json');progress=base.read(args.dataset/'progress.json')
    if (args.videos.stat().st_size!=progress['config']['video_size']
            or args.videos.stat().st_mtime_ns!=progress['config']['video_mtime']):raise ValueError('review_archive_changed')
    from collections import Counter
    frequency=Counter(i for r in config['splits']['train'] for i in r['target_ids'])
    rows=sorted(config['splits']['train'],key=lambda r:(-sum(1/frequency[i] for i in set(r['target_ids'])),r['video_id']))[:args.count]
    state_path=args.output/'review-progress.json'
    contract=dict(dataset_sha256=base.digest(args.dataset/'training-config.json'),video_bytes=args.videos.stat().st_size,
        video_mtime=args.videos.stat().st_mtime_ns,ids=[r['video_id'] for r in rows],implementation_sha256=base.digest(__file__))
    state=base.read(state_path) if state_path.exists() else dict(contract=contract,completed={},failed={})
    if state['contract']!=contract:raise ValueError('review_contract_changed')
    temp=args.dataset.parents[2]/'original-video'/'temp'/args.output.name;temp.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.videos) as archive:
        for row in rows:
            vid=row['video_id'];image_path=args.output/(vid+'.jpg')
            if vid in state['completed']:
                if base.digest(image_path)!=state['completed'][vid]['sha256']:raise ValueError('review_image_changed')
                continue
            video=temp/(vid+'.mp4');owned=False
            try:
                data_path=args.dataset/'features'/(vid+'.npz')
                if base.feature_ok(data_path,row)!=config['feature_hashes'][vid]:raise ValueError('review_features_changed')
                if shutil.disk_usage(temp).free<row['bytes']+128*1024**2:raise OSError('review_disk_space')
                video=review_video(archive,row,temp);owned=True
                with np.load(data_path,allow_pickle=False) as data:
                    sheet,pts=contact_sheet(video,data['xy'],data['mask'],data['t'],progress['completed'][vid]['fitted_wh'])
                encoded=cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,92])[1]
                image_bytes=encoded.tobytes()
                if image_path.exists() and image_path.read_bytes()!=image_bytes:
                    image_path=args.output/(vid+'-'+uuid.uuid4().hex+'.jpg')
                if not image_path.exists():
                    with image_path.open('xb') as target:target.write(image_bytes)
                restored=cv2.imread(str(image_path))
                if restored is None or restored.shape!=sheet.shape:raise ValueError('review_image_roundtrip')
                state['completed'][vid]=dict(sha256=base.digest(image_path),image=image_path.name,PTS=pts,
                    decoded_all_frames=True,CRC_verified=True,human_status='uncertain')
                state['failed'].pop(vid,None);base.write(state_path,state)
                if owned:video.unlink()
            except (OSError,ValueError,RuntimeError) as exc:
                state['failed'][vid]=dict(reason=str(exc),temporary_preserved=str(video) if owned else None)
                base.write(state_path,state)
    csv_path=args.output/'expert-review.csv'
    if not csv_path.exists():
        with csv_path.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['video_id','original_label','jamo','start','end','review_status','corrected_label','review_notes','performer_verified'])
            writer.writeheader()
            for row in rows:writer.writerow(dict(video_id=row['video_id'],original_label=row['text'],
                jamo=''.join(base.tokens_for(row['target_ids'])),start=row['start'],end=row['end'],
                review_status='uncertain',corrected_label='',review_notes='',performer_verified=False))
    pieces=['<!doctype html><meta charset="utf-8"><title>Private label review</title>',
            '<h1>전문가 검토용 · 모든 의미 검증 상태 uncertain</h1><p>원본 RGB의 손 영역을 실제 PTS로 표시했습니다. 좌우 반전 없음. 지문자 철자·라벨 구간·수행자 근거를 검토하세요. JPEG는 원본 영상 전체를 대체하지 않습니다.</p>']
    for row in rows:
        pieces.append('<h2>'+html.escape(row['video_id']+' / '+row['text'])+'</h2><p>'+html.escape(''.join(base.tokens_for(row['target_ids'])))+'</p>')
        if row['video_id'] in state['completed']:pieces.append('<img style="max-width:100%" src="'+html.escape(state['completed'][row['video_id']]['image'])+'">')
    (args.output/'index.html').write_text('\n'.join(pieces),encoding='utf-8')
    print(json.dumps(dict(selected=len(rows),images=len(state['completed']),failed=len(state['failed']),verified_labels=0)),flush=True)


if __name__=='__main__':main()
