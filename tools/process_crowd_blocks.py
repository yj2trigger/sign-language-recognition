"""Resumable observed-frame CROWD audit. Blocks are recovery units, never letters."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
ATTEMPTS = 11
BLOCK_SIZE = 15


class Unrecoverable(ValueError):
    """A structural incompatibility that retries cannot repair."""


class DecodeError(RuntimeError):
    pass


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    """Journal-only retries never repeat successful landmark extraction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    for attempt in range(ATTEMPTS):
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return attempt
        except (PermissionError, OSError):
            if attempt == ATTEMPTS-1:
                raise
            time.sleep(min(.1*2**attempt, 2.))


def retry(operation, events=None):
    """At most one initial attempt and ten additional attempts."""
    for number in range(1, ATTEMPTS+1):
        try:
            return operation(), number
        except Unrecoverable:
            raise
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, MemoryError, cv2.error, subprocess.SubprocessError) as exc:
            if events is not None:
                events.append(dict(attempt=number, timestamp=now(), error=type(exc).__name__, cause=str(exc)))
            if number == ATTEMPTS:
                raise
            gc.collect()
            time.sleep(min(.1*2**(number-1), 2.))


def stage_retry(operation,state,path,key):
    stage=state.setdefault('stages',{}).setdefault(key,dict(attempts=0,errors=[],status='PENDING'))
    if stage['status'] in ('FAILED_AFTER_RETRIES','SKIPPED_UNRECOVERABLE'):
        raise Unrecoverable('previously_exhausted_stage_'+key)
    while stage['attempts']<ATTEMPTS:
        stage['attempts']+=1;stage['status']='RETRYING';stage['last_attempt_at']=now()
        atomic_json(path,state)
        try:
            value=operation();stage['status']='SUCCESS';atomic_json(path,state)
            return value,stage['attempts']
        except Unrecoverable as exc:
            stage.update(status='SKIPPED_UNRECOVERABLE',error=str(exc));atomic_json(path,state);raise
        except (OSError,RuntimeError,ValueError,MemoryError,zipfile.BadZipFile,cv2.error,subprocess.SubprocessError) as exc:
            stage['errors'].append(dict(attempt=stage['attempts'],timestamp=now(),error=type(exc).__name__,cause=str(exc)))
            if stage['attempts']==ATTEMPTS:
                stage['status']='FAILED_AFTER_RETRIES';atomic_json(path,state);raise
            atomic_json(path,state);gc.collect();time.sleep(min(.1*2**(stage['attempts']-1),2.))
    raise Unrecoverable('stage_attempt_budget_exhausted_'+key)


def sampling_indices(times):
    times = np.asarray(times, np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or times[0] < 0 or not (np.diff(times) > 0).all():
        raise Unrecoverable('invalid_actual_PTS')
    grid = times[0] + np.arange(int(np.floor((times[-1]-times[0])*15))+1)/15
    selected = np.unique(np.searchsorted(times, grid-1e-7))
    # Include the actual endpoint for whole-video coverage; never duplicate a frame.
    return np.unique(np.append(selected, len(times)-1)).astype(np.int64)


def block_ranges(length):
    return [(start, min(start+BLOCK_SIZE, length)) for start in range(0, length, BLOCK_SIZE)]


def validate_block(path, indices, positions, times, schema):
    with np.load(path, allow_pickle=False) as data:
        count = len(indices)
        if not 0 < count <= BLOCK_SIZE or data['xy'].shape != (count,53,2) or data['mask'].shape != (count,53):
            raise ValueError('block_shape_changed')
        if data['xy'].dtype != np.float32 or data['mask'].dtype != np.bool_ or not np.isfinite(data['xy']).all():
            raise ValueError('invalid_observed_coordinates')
        if str(data['schema'].item()) != schema:
            raise Unrecoverable('block_schema_changed')
        for key, expected in [('source_index',indices),('selected_index',positions),('t',times)]:
            if not np.array_equal(data[key], expected):
                raise ValueError('block_actual_frame_identity_changed')
        if not (np.diff(data['t']) > 0).all() or data['origin'].shape != (count,) or data['context'].shape != (count,):
            raise ValueError('invalid_block_timing_or_provenance')
    return digest(path)


def geometry(xy, mask, times, selected, contexts, aspect):
    """Observable angles and motion hypotheses, without a handshape-to-jamo rule."""
    count = len(xy)
    angles = np.full((count,2,5,2), np.nan, np.float64)
    speeds = np.full((count,2), np.nan, np.float64)
    shape_changes = np.full((count,2), np.nan, np.float64)
    context_change = np.zeros(count, bool)
    if count > 1:
        context_change[1:] = contexts[1:] != contexts[:-1]
    for hand, offset in enumerate((11,32)):
        points = xy[:,offset:offset+21].astype(np.float64).copy()
        points[:,:,0] *= aspect
        valid = mask[:,offset:offset+21]
        palm = np.linalg.norm(points[:,9]-points[:,0], axis=1)
        for finger in range(5):
            chain = list(range(1+4*finger,5+4*finger))
            for joint, (a,b,c) in enumerate((chain[:3],chain[1:])):
                u, v = points[:,a]-points[:,b], points[:,c]-points[:,b]
                denominator = np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
                usable = valid[:,a]&valid[:,b]&valid[:,c]&(denominator>1e-12)
                cosine = (u*v).sum(axis=1)/np.where(usable,denominator,1.)
                angles[usable,hand,finger,joint] = np.degrees(np.arccos(np.clip(cosine[usable],-1.,1.)))
        for i in range(1,count):
            if selected[i] != selected[i-1]+1 or context_change[i]:
                continue
            dt = times[i]-times[i-1]
            if dt <= 0:
                continue
            if valid[i,0] and valid[i-1,0] and min(palm[i],palm[i-1]) > 1e-6:
                speeds[i,hand] = np.linalg.norm(points[i,0]-points[i-1,0])/max(palm[i],palm[i-1])/dt
            difference = np.abs(angles[i,hand]-angles[i-1,hand])
            if np.isfinite(difference).any():
                shape_changes[i,hand] = np.nanmean(difference)/dt
    stable = np.any((speeds<=.5)&(shape_changes<=25),axis=1)
    transition = np.any((speeds>3)|(shape_changes>150),axis=1)
    def spans(flags):
        result=[]
        start=None
        for i in range(count+1):
            continues = i<count and flags[i] and (start is None or selected[i]==selected[i-1]+1)
            if start is not None and not continues:
                result.append(dict(start_selected_frame=int(selected[start])+1,end_selected_frame=int(selected[i-1])+1,
                                   start_PTS=float(times[start]),end_PTS=float(times[i-1]),frames=i-start))
                start=None
            if i<count and flags[i] and start is None:
                start=i
        return result
    fingers=[]
    for hand in range(2):
        fingers.append([dict(valid_angles=int(np.isfinite(angles[:,hand,f]).sum()),
            extended_angles=int((angles[:,hand,f]>=150).sum()),bent_angles=int((angles[:,hand,f]<=100).sum())) for f in range(5)])
    return dict(left_detected_frames=int(mask[:,11:32].any(axis=1).sum()),
                right_detected_frames=int(mask[:,32:53].any(axis=1).sum()),
                finger_angles=fingers,stable_observation_spans=spans(stable),transition_observation_spans=spans(transition),
                tracking_context_change_edges=np.flatnonzero(context_change).tolist(),
                handshape_to_jamo_candidates=[],semantic_status='UNCERTAIN',
                interpretation='geometric hypotheses; spans may cross block boundaries; no letter per block')


def contact_sheet(snapshots,xy,mask,fitted_wh):
    panels=[]
    for local,image,pts in snapshots:
        height,width=image.shape[:2];fw,fh=fitted_wh
        points=xy[local,11:][mask[local,11:]].astype(np.float64)
        panel=np.full((260,320,3),235,np.uint8)
        if len(points):
            points[:,0]=points[:,0]*fw-(fw-width)//2;points[:,1]=points[:,1]*fh-(fh-height)//2
            left,top=np.maximum(np.floor(points.min(0)-32),0).astype(int)
            right,bottom=np.minimum(np.ceil(points.max(0)+32),(width,height)).astype(int)
            crop=image[top:bottom,left:right]
            if crop.size:
                scale=min(320/crop.shape[1],230/crop.shape[0]);view=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
                panel[:view.shape[0],:view.shape[1]]=view
        else:cv2.putText(panel,'No hand detected',(8,80),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cv2.putText(panel,f'PTS {pts:.6f}s',(8,250),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        panels.append(panel)
    return np.hstack(panels)


def verify_source(path, row):
    if path.stat().st_size != row['bytes']:
        raise ValueError('source_member_size_mismatch')
    crc=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2), b''):
            crc=zlib.crc32(chunk,crc)
    if crc != row['crc']:
        raise ValueError('source_member_CRC_mismatch')


def source_file(row, settings, state):
    old = settings['old_failed'].get(row['video_id'],{}).get('temporary_video_preserved')
    if old and Path(old).is_file():
        try:
            verify_source(Path(old),row)
            return Path(old), False
        except (OSError,ValueError) as exc:
            state['preserved_old_input_error']=dict(path=old,reason=str(exc))
    temporary = Path(settings['output'])/'temp'/row['video_id']
    temporary.mkdir(parents=True,exist_ok=True)
    existing=state.get('temporary_video')
    if existing and Path(existing).is_file():
        try:
            verify_source(Path(existing),row)
            return Path(existing), True
        except (OSError,ValueError):
            state.setdefault('preserved_partial_inputs',[]).append(existing)
    if shutil.disk_usage(temporary).free < row['bytes']+128*1024**2:
        raise OSError('insufficient_temporary_disk_space')
    path=temporary/(uuid.uuid4().hex+'.mp4')
    state['temporary_video']=str(path)
    atomic_json(temporary/'owner.json',dict(video_id=row['video_id'],member=row['member'],crc=row['crc'],bytes=row['bytes']))
    with zipfile.ZipFile(settings['archive']) as archive:
        with archive.open(row['member']) as source,path.open('xb') as target:
            shutil.copyfileobj(source,target,1024**2)
            target.flush();os.fsync(target.fileno())
    verify_source(path,row)
    return path, True


def acquire_source(row,settings,state,path):
    stage=state.get('stages',{}).get('source',{})
    if stage.get('status')=='SUCCESS':
        old=settings['old_failed'].get(row['video_id'],{}).get('temporary_video_preserved')
        existing=state.get('temporary_video') or old
        if existing and Path(existing).is_file():
            _,attempts=retry(lambda:verify_source(Path(existing),row),state.setdefault('source_revalidation_errors',[]))
            state['source_revalidation_attempts']=attempts
            return (Path(existing),existing!=old),stage['attempts']
        # A missing disposable input is a new restoration task, not a reason to
        # repeat successful blocks or reset the original extraction budget.
        return stage_retry(lambda:source_file(row,settings,state),state,path,'source_restore')
    return stage_retry(lambda:source_file(row,settings,state),state,path,'source')


def probe(path):
    capture=cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise DecodeError('video_open_failed')
        declared=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=float(capture.get(cv2.CAP_PROP_FPS))
        wh=[int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if min(wh)<1:raise Unrecoverable('invalid_video_resolution')
        times=[]
        while True:
            ok,frame=capture.read()
            if not ok:break
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if not times or (declared>0 and len(times)<declared):
            raise DecodeError('incomplete_video_decode')
        clock='OpenCV_decoded_PTS'
        values=np.asarray(times,np.float64)
        if not np.isfinite(values).all() or values[0]<0 or not (np.diff(values)>0).all():
            executable=shutil.which('ffprobe')
            if not executable:
                raise Unrecoverable('actual_PTS_unresolved_no_ffprobe_fallback')
            result=subprocess.run([executable,'-v','error','-select_streams','v:0','-show_frames',
                '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],capture_output=True,check=True,timeout=60)
            values=np.asarray([float(f['best_effort_timestamp_time']) for f in json.loads(result.stdout)['frames']],np.float64)
            if len(values)!=len(times):raise Unrecoverable('ffprobe_decoded_frame_count_disagrees')
            clock='ffprobe_best_effort_PTS_frame_order'
        indices=sampling_indices(values)
        return dict(times=values.tolist(),selected=indices.tolist(),source_frames=len(times),declared_frames=declared,
                    metadata_fps=fps if math.isfinite(fps) and fps>0 else None,resolution=wh,clock=clock,
                    observed_average_fps=(len(values)-1)/(values[-1]-values[0]) if len(values)>1 else None,
                    no_synthetic_frames=True,includes_actual_endpoint=True)
    finally:
        capture.release()


class Decoder:
    def __init__(self,path,plan):
        self.path=path;self.plan=plan;self.capture=None;self.cursor=0;self.sequential=False
    def reset(self):
        if self.capture is not None:self.capture.release()
        self.capture=cv2.VideoCapture(str(self.path));self.cursor=0
    def frame(self,index):
        if self.capture is None:self.reset()
        if index<self.cursor or index-self.cursor>120:
            if self.sequential:
                self.reset()
            else:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES,index);self.cursor=index
        while self.cursor<index:
            if not self.capture.grab():raise DecodeError('decoder_grab_failed')
            self.cursor+=1
        ok,frame=self.capture.read();self.cursor+=1
        if not ok:raise DecodeError('decoder_read_failed')
        if abs(self.capture.get(cv2.CAP_PROP_POS_FRAMES)-(index+1))>.5:
            raise DecodeError('decoded_frame_index_changed')
        if self.plan['clock']=='OpenCV_decoded_PTS' and abs(self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000-self.plan['times'][index])>1e-5:
            raise DecodeError('decoded_PTS_changed')
        return frame
    def close(self):
        if self.capture is not None:self.capture.release()


class Detector:
    def __init__(self,state):
        self.state=state;self.model=None;self.last=None;self.history=deque(maxlen=15);self.context=None
    def close(self):
        if self.model is not None:self.model.close()
        self.model=None;self.last=None
    def extract(self,image,position,events):
        import landmarks
        if self.model is None or self.last!=position-1:
            self.close()
            self.model=landmarks.mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=0)
            self.context=self.state['next_context'];self.state['next_context']+=1
            for old_position,old_image in self.history:
                landmarks.process_frame(self.model,old_image)
                events.append(int(old_position))
        xy,mask,z,wh=landmarks.process_frame(self.model,image)
        self.last=position
        return xy,mask,self.context,wh


def reusable(row,settings,plan):
    previous=Path(settings['previous'])
    completed=settings['old_completed'].get(row['video_id'])
    if completed is None:return {},None
    old_row=settings['old_rows'][row['video_id']]
    if old_row['member']!=row['member'] or old_row['crc']!=row['crc'] or not completed.get('member_crc_checked'):
        return {},'old_member_contract_differs'
    path=previous/'features'/(row['video_id']+'.npz')
    if digest(path)!=completed['sha256']:raise Unrecoverable('previous_feature_SHA_changed')
    with np.load(path,allow_pickle=False) as data:
        xy,mask,times=data['xy'].copy(),data['mask'].copy(),data['t'].astype(np.float64)
    if xy.shape!=(len(times),53,2) or mask.shape!=(len(times),53) or mask.dtype!=np.bool_ or not np.isfinite(xy).all():
        raise Unrecoverable('previous_feature_schema_changed')
    actual=np.asarray(plan['times']);selected=set(plan['selected']);result={}
    for i,pts in enumerate(times):
        source=int(np.searchsorted(actual,pts-1e-6))
        if source<len(actual) and abs(actual[source]-pts)<=1e-6 and source in selected:
            result[source]=(xy[i],mask[i])
    return result,completed['sha256']


def block_job(row,settings,state,plan,number,start,end,decoder,detector,reused):
    directory=Path(settings['output'])/'videos'/row['video_id']
    key=str(number)
    record=state['blocks'].setdefault(key,dict(block=number,status='PENDING',attempts=0,errors=[],
        first_attempt_at=None,semantic_status='UNCERTAIN',start_frame=start+1,end_frame=end))
    indices=np.asarray(plan['selected'][start:end],np.int64)
    positions=np.arange(start,end,dtype=np.int64);times=np.asarray(plan['times'],np.float64)[indices]
    record.update(start_source_frame=int(indices[0])+1,end_source_frame=int(indices[-1])+1,PTS_start=float(times[0]),PTS_end=float(times[-1]),actual_frames=len(indices))
    if record['status']=='VERIFIED_TECHNICAL':
        try:
            if (validate_block(record['path'],indices,positions,times,settings['schema_hash'])==record['sha256']
                    and digest(record['review_image'])==record['review_image_sha256']):
                detector.close();detector.history.clear()
                return
        except (OSError,ValueError,RuntimeError):
            pass
        record.setdefault('preserved_artifacts',[]).append(dict(path=record['path'],sha256=record['sha256']))
        record['status']='PENDING'
    if record['status'] in ('FAILED_AFTER_RETRIES','SKIPPED_UNRECOVERABLE'):
        return
    candidate=None;image_bytes=None;image_path=None;snapshots=[]
    prefix=list(detector.history)
    while record['attempts']<ATTEMPTS:
        record['attempts']+=1;record['first_attempt_at']=record['first_attempt_at'] or now()
        record['status']='RETRYING';record['last_attempt_at']=now()
        atomic_json(directory/'state.json',state)
        try:
            if candidate is None:
                if not detector.history and start:
                    for old in range(max(0,start-15),start):
                        detector.history.append((old,decoder.frame(plan['selected'][old])))
                    prefix=list(detector.history)
                arrays=[];masks=[];contexts=[];origins=[];context_frames=[];snapshots=[]
                snapshot_positions={0,len(indices)//2,len(indices)-1}
                fitted=None
                for position,index in zip(positions,indices):
                    image=decoder.frame(int(index))
                    if int(index) in reused:
                        xy,mask=reused[int(index)];context=0;origin=1;detector.last=None
                    else:
                        xy,mask,context,fitted=detector.extract(image,int(position),context_frames);origin=0
                    arrays.append(xy);masks.append(mask);contexts.append(context);origins.append(origin)
                    local=len(arrays)-1
                    if local in snapshot_positions:snapshots.append((local,image.copy(),float(times[local])))
                    detector.history.append((int(position),image))
                candidate=directory/(f'block-{number:05d}-'+uuid.uuid4().hex+'.npz')
                with candidate.open('xb') as stream:
                    np.savez_compressed(stream,xy=np.asarray(arrays,np.float32),mask=np.asarray(masks,bool),t=times,
                        source_index=indices,selected_index=positions,origin=np.asarray(origins,np.uint8),
                        context=np.asarray(contexts,np.int64),schema=np.asarray(settings['schema_hash']))
                    stream.flush();os.fsync(stream.fileno())
                record['context_only_selected_frames']=context_frames
                record['context_only_extractions']=len(context_frames)
                record['reused_frames']=int(sum(origins))
                if fitted:state['fitted_wh']=list(fitted)
                sheet=contact_sheet(snapshots,np.asarray(arrays),np.asarray(masks),state['fitted_wh'])
                ok,encoded=cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,88])
                if not ok:raise RuntimeError('review_image_encode_failed')
                image_bytes=encoded.tobytes();image_path=candidate.with_suffix('.jpg')
            sha=validate_block(candidate,indices,positions,times,settings['schema_hash'])
            if image_bytes is None:
                with np.load(candidate,allow_pickle=False) as saved:
                    sheet=contact_sheet(snapshots,saved['xy'],saved['mask'],state['fitted_wh'])
                ok,encoded=cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,88])
                if not ok:raise RuntimeError('review_image_encode_failed')
                image_bytes=encoded.tobytes();image_path=candidate.with_suffix('.jpg')
            if image_path.exists() and image_path.read_bytes()!=image_bytes:
                image_path=candidate.with_name(candidate.stem+'-'+uuid.uuid4().hex+'.jpg')
            if not image_path.exists():
                with image_path.open('xb') as image_stream:image_stream.write(image_bytes)
            restored=cv2.imread(str(image_path))
            if restored is None or restored.shape[0]!=260:raise RuntimeError('review_image_roundtrip_failed')
            with np.load(candidate,allow_pickle=False) as observed:
                fitted=state.get('fitted_wh',[16,9])
                record['analysis']=geometry(observed['xy'],observed['mask'],observed['t'],observed['selected_index'],
                                            observed['context'],fitted[0]/fitted[1])
            record.update(status='VERIFIED_TECHNICAL',path=str(candidate),sha256=sha,verified_at=now(),
                retries=record['attempts']-1,error_code=None,recovery='verified retry candidate reused' if record['attempts']>1 else None)
            record.update(review_image=str(image_path),review_image_sha256=digest(image_path))
            atomic_json(directory/'state.json',state)
            return
        except Unrecoverable as exc:
            record.update(status='SKIPPED_UNRECOVERABLE',error_code=str(exc),error_cause=type(exc).__name__)
            record['errors'].append(dict(attempt=record['attempts'],timestamp=now(),cause=str(exc),recovery='structural stop'))
            break
        except (OSError,RuntimeError,ValueError,MemoryError,zipfile.BadZipFile,cv2.error) as exc:
            record['errors'].append(dict(attempt=record['attempts'],timestamp=now(),error=type(exc).__name__,cause=str(exc),
                recovery='reopen decoder sequentially; reset tracker with previous observed context; preserve candidate'))
            record.update(error_code=type(exc).__name__,error_cause=str(exc))
            detector.close();detector.history=deque(prefix,maxlen=15);decoder.sequential=True;decoder.reset()
            if candidate is not None:
                try:validate_block(candidate,indices,positions,times,settings['schema_hash'])
                except (ValueError,zipfile.BadZipFile):
                    record.setdefault('preserved_partial_artifacts',[]).append(str(candidate));candidate=None
                except OSError:pass
            if record['attempts']<ATTEMPTS:
                time.sleep(min(.1*2**(record['attempts']-1),2.))
    if record['status']!='SKIPPED_UNRECOVERABLE':record['status']='FAILED_AFTER_RETRIES'
    record['retries']=max(0,record['attempts']-1)
    atomic_json(directory/'state.json',state)


def label_review_context(label):
    from fingerspelling.jamo import decompose_hangul
    result=dict(label)
    try:
        result.update(jamo=decompose_hangul(label['text']),jamo_status='SUPPORTED')
    except ValueError as exc:
        result.update(jamo=None,jamo_status='UNSUPPORTED',jamo_error=str(exc))
    return result


def video_summary(row,state,settings):
    data=[]
    for record in state['blocks'].values():
        if record['status']=='VERIFIED_TECHNICAL':
            with np.load(record['path'],allow_pickle=False) as archive:
                data.append({k:archive[k].copy() for k in ('xy','mask','t','selected_index','context')})
    result=dict(video_id=row['video_id'],original_labels=[label_review_context(label) for label in row['labels']],semantic_status='UNCERTAIN',
        semantic_verified=False,label_mismatch_suspected=False,block_is_letter=False)
    if data:
        combined={k:np.concatenate([d[k] for d in data]) for k in data[0]}
        order=np.argsort(combined['selected_index']);combined={k:v[order] for k,v in combined.items()}
        aspect=state.get('fitted_wh',[16,9]);result['geometry']=geometry(combined['xy'],combined['mask'],combined['t'],
            combined['selected_index'],combined['context'],aspect[0]/aspect[1])
        actual_times=np.asarray(load(Path(settings['output'])/'videos'/row['video_id']/'plan.json')['times'])
        result['label_time_overlap']=[dict(start=label['start'],end=label['end'],
            overlaps=bool(np.any((actual_times>=label['start'])&(actual_times<label['end'])))) for label in row['labels']]
        result['label_mismatch_suspected']=bool(row['labels'] and not any(r['overlaps'] for r in result['label_time_overlap']))
        result['mismatch_evidence']='no actual decoded source frame in any label time interval' if result['label_mismatch_suspected'] else None
    atomic_json(Path(settings['output'])/'videos'/row['video_id']/'summary.json',result)
    import html
    pieces=['<!doctype html><meta charset="utf-8"><title>Observed block review</title>',
            '<h1>'+html.escape(row['video_id'])+'</h1><p>Semantic review UNCERTAIN. Actual observed PTS. Blocks are not letters.</p>']
    for label in result['original_labels']:
        pieces.append('<h2>'+html.escape(label['text'])+'</h2><p>'+html.escape(''.join(label['jamo']) if label['jamo'] is not None else 'Jamo unsupported: '+label['jamo_error'])+'</p>')
    for record in state['blocks'].values():
        if record['status']=='VERIFIED_TECHNICAL' and record.get('review_image'):
            pieces.append('<p>Block '+str(record['block'])+' / '+str(record['PTS_start'])+'–'+str(record['PTS_end'])+' s</p><img src="'+html.escape(Path(record['review_image']).name)+'">')
    (Path(settings['output'])/'videos'/row['video_id']/'review.html').write_text('\n'.join(pieces),encoding='utf-8')


def process_video(row,settings):
    cv2.setNumThreads(1)
    directory=Path(settings['output'])/'videos'/row['video_id'];directory.mkdir(parents=True,exist_ok=True)
    path=directory/'state.json'
    state=load(path) if path.exists() else dict(video_id=row['video_id'],schema_hash=settings['schema_hash'],
        source_crc=row['crc'],source_bytes=row['bytes'],technical_status='PENDING',semantic_status='UNCERTAIN',
        first_attempt_at=now(),blocks={},next_context=1,preparation_errors=[])
    if state['schema_hash']!=settings['schema_hash'] or state['source_crc']!=row['crc']:
        return dict(video_id=row['video_id'],technical_status='SKIPPED_UNRECOVERABLE',reason='state_source_or_schema_changed')
    state['worker_pid']=os.getpid()
    atomic_json(path,state)
    source=None;owned=False;decoder=None;detector=None
    try:
        plan_path=directory/'plan.json'
        plan=load(plan_path) if plan_path.exists() else None
        if plan is not None and all(state['blocks'].get(str(i+1),{}).get('status')=='VERIFIED_TECHNICAL' for i in range(len(block_ranges(len(plan['selected']))))):
            repair=[]
            for i,(start,end) in enumerate(block_ranges(len(plan['selected'])),1):
                record=state['blocks'][str(i)];indices=np.asarray(plan['selected'][start:end],np.int64)
                try:
                    validate,read_attempts=retry(lambda:validate_block(record['path'],indices,np.arange(start,end),np.asarray(plan['times'])[indices],settings['schema_hash']),record.setdefault('read_verification_errors',[]))
                    record['read_verification_attempts']=read_attempts
                    if validate!=record['sha256'] or digest(record['review_image'])!=record['review_image_sha256']:
                        repair.append(i)
                except (ValueError,OSError):repair.append(i)
            if not repair:
                state['technical_status']='SUCCESS';retry(lambda:video_summary(row,state,settings),state['preparation_errors']);atomic_json(path,state)
                return dict(video_id=row['video_id'],technical_status='SUCCESS',resumed_without_extraction=True,blocks=len(state['blocks']))
            state['repair_block_numbers']=repair
        source_result,attempts=acquire_source(row,settings,state,path)
        source,owned=source_result;state['source_preparation_attempts']=attempts;state['source_CRC_verified']=True
        atomic_json(path,state)
        if plan is None:
            plan,attempts=stage_retry(lambda:probe(source),state,path,'probe')
            state['probe_attempts']=attempts;atomic_json(plan_path,plan)
        state['total_blocks']=len(block_ranges(len(plan['selected'])));state['selected_frames']=len(plan['selected'])
        state['actual_source_frames']=plan['source_frames'];state['whole_source_decoded']=True
        reused,cache_sha=reusable(row,settings,plan)
        state['reused_cache_sha256']=cache_sha
        width,height=plan['resolution'];ratio=16/9
        state['fitted_wh']=([width,height] if abs(width/height-ratio)<1e-3 else
            [round(height*ratio),height] if width/height<ratio else [width,round(width/ratio)])
        decoder=Decoder(source,plan);detector=Detector(state)
        for number,(start,end) in enumerate(block_ranges(len(plan['selected'])),1):
            block_job(row,settings,state,plan,number,start,end,decoder,detector,reused)
        values=[record['status'] for record in state['blocks'].values()]
        state['technical_status']='SUCCESS' if values and all(v=='VERIFIED_TECHNICAL' for v in values) else (
            'FAILED_AFTER_RETRIES' if 'FAILED_AFTER_RETRIES' in values else 'SKIPPED_UNRECOVERABLE')
        retry(lambda:video_summary(row,state,settings),state['preparation_errors'])
        state['finished_at']=now();atomic_json(path,state)
        if owned and state['technical_status']=='SUCCESS':
            decoder.close();decoder=None
            verify_source(source,row)
            assert source.resolve().is_relative_to((Path(settings['output'])/'temp').resolve())
            try:
                retry(source.unlink,state.setdefault('cleanup_errors',[]))
                state['temporary_cleanup']='SUCCESS'
            except OSError as exc:
                state['temporary_cleanup']='PRESERVED_AFTER_RETRIES'
                state['cleanup_error']=str(exc)
            atomic_json(path,state)
    except Unrecoverable as exc:
        exhausted=any(stage.get('status')=='FAILED_AFTER_RETRIES' for stage in state.get('stages',{}).values())
        state.update(technical_status='FAILED_AFTER_RETRIES' if exhausted else 'SKIPPED_UNRECOVERABLE',error_code=str(exc),finished_at=now())
        atomic_json(path,state)
    except Exception as exc:
        state.update(technical_status='FAILED_AFTER_RETRIES',error_code=type(exc).__name__,error_cause=str(exc),
            traceback=traceback.format_exc(),finished_at=now())
        atomic_json(path,state)
    finally:
        if decoder is not None:decoder.close()
        if detector is not None:detector.close()
    return dict(video_id=row['video_id'],technical_status=state['technical_status'],blocks=len(state['blocks']),reason=state.get('error_code'))


def inventory(settings):
    labels={}
    with zipfile.ZipFile(settings['labels']) as archive:
        for member in archive.infolist():
            if not member.filename.endswith('.json'):continue
            raw=json.loads(archive.read(member).decode('utf-8-sig'))
            key=Path(raw['metaData']['name']).stem.rstrip('_')
            entries=[]
            for entry in raw.get('data',[]):
                for attribute in entry.get('attributes',[]):
                    if isinstance(attribute.get('name'),str):
                        entries.append(dict(text=attribute['name'],start=float(entry['start']),end=float(entry['end'])))
            if key in labels:raise Unrecoverable('duplicate_label_video_ID')
            labels[key]=entries
    result=[];seen=set()
    with zipfile.ZipFile(settings['archive']) as archive:
        for member in archive.infolist():
            if not member.filename.lower().endswith('.mp4'):continue
            key=Path(member.filename).stem.rstrip('_');unique=key
            if key in seen:unique=key+'__'+hashlib.sha256(member.filename.encode()).hexdigest()[:12]
            seen.add(unique)
            result.append(dict(video_id=unique,canonical_id=key,member=member.filename,bytes=member.file_size,crc=member.CRC,
                               labels=labels.get(key,[])))
    return result


def review_catalog(settings,rows):
    """Human review index; never promotes model/geometric hypotheses to truth."""
    import csv
    import html
    entries=[]
    output=Path(settings['output'])
    for row in rows:
        folder=output/'videos'/row['video_id']
        state=load(folder/'state.json') if (folder/'state.json').exists() else {}
        summary=load(folder/'summary.json') if (folder/'summary.json').exists() else {}
        entries.append(dict(video_id=row['video_id'],original_label=' | '.join(label['text'] for label in row['labels']),
            review_status='uncertain',corrected_label='',review_notes='',
            technical_status=state.get('technical_status','PENDING'),
            label_time_overlap=summary.get('label_time_overlap',[])))
    destination=output/'review-template.csv'
    # Re-running an audit must never overwrite a human's existing review edits.
    if not destination.exists():
        with destination.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['video_id','original_label','review_status','corrected_label','review_notes'])
            writer.writeheader()
            for entry in entries:writer.writerow({key:entry[key] for key in writer.fieldnames})
    parts=['<!doctype html><meta charset="utf-8"><title>CROWD review inventory</title>',
        '<h1>CROWD observed-frame review</h1><p>All semantic decisions remain UNCERTAIN. Blocks are not letters.</p>',
        '<table><tr><th>Video ID</th><th>Original label</th><th>Technical state</th><th>Semantic state</th></tr>']
    for entry in entries:
        url='videos/'+entry['video_id']+'/review.html'
        parts.append('<tr><td><a href="'+html.escape(url)+'">'+html.escape(entry['video_id'])+'</a></td><td>'+
            html.escape(entry['original_label'])+'</td><td>'+html.escape(entry['technical_status'])+'</td><td>UNCERTAIN</td></tr>')
    parts.append('</table>')
    (output/'review-index.html').write_text('\n'.join(parts),encoding='utf-8')


def audit(settings,rows,verify=False):
    totals=dict(target_videos=len(rows),attempted_videos=0,technical_success_videos=0,failed_after_retries_videos=0,
        structurally_excluded_videos=0,unprocessed_videos=0,total_blocks=0,successful_blocks=0,retried_successful_blocks=0,
        failed_blocks=0,excluded_blocks=0,unprocessed_blocks=0,semantic_verified_videos=0,label_mismatch_suspected_videos=0,
        semantic_uncertain_videos=0,unplanned_videos=0,reused_observed_frames=0,context_only_extractions=0)
    lists=dict(failed_videos=[],unprocessed_videos=[],unplanned_videos=[],failed_blocks=[],label_mismatch_suspected=[],artifact_verification_errors=[])
    for row in rows:
        folder=Path(settings['output'])/'videos'/row['video_id'];path=folder/'state.json'
        if not path.exists():
            totals['unprocessed_videos']+=1;lists['unprocessed_videos'].append(dict(video_id=row['video_id'],reason='no durable worker start'))
            totals['unplanned_videos']+=1;lists['unplanned_videos'].append(row['video_id'])
            continue
        state=load(path);totals['attempted_videos']+=1;totals['semantic_uncertain_videos']+=1
        status=state['technical_status']
        if status=='SUCCESS':totals['technical_success_videos']+=1
        elif status=='FAILED_AFTER_RETRIES':
            totals['failed_after_retries_videos']+=1;lists['failed_videos'].append(dict(video_id=row['video_id'],reason=state.get('error_code')))
        elif status=='SKIPPED_UNRECOVERABLE':
            totals['structurally_excluded_videos']+=1;lists['failed_videos'].append(dict(video_id=row['video_id'],reason=state.get('error_code')))
        else:
            totals['unprocessed_videos']+=1;lists['unprocessed_videos'].append(dict(video_id=row['video_id'],reason=status))
        plan_path=folder/'plan.json'
        if not plan_path.exists():
            totals['unplanned_videos']+=1;lists['unplanned_videos'].append(row['video_id']);continue
        plan=load(plan_path)
        artifact_failure=False
        for number,(start,end) in enumerate(block_ranges(len(plan['selected'])),1):
            totals['total_blocks']+=1;record=state['blocks'].get(str(number),{});value=record.get('status')
            if value=='VERIFIED_TECHNICAL':
                if verify:
                    indices=np.asarray(plan['selected'][start:end],np.int64)
                    def verify_artifact():
                        if validate_block(record['path'],indices,np.arange(start,end),np.asarray(plan['times'])[indices],settings['schema_hash'])!=record['sha256']:
                            raise ValueError('saved_block_SHA_changed')
                        if digest(record['review_image'])!=record['review_image_sha256']:
                            raise ValueError('saved_review_image_SHA_changed')
                    try:retry(verify_artifact)
                    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile,KeyError) as exc:
                        artifact_failure=True
                        kind='excluded_blocks' if isinstance(exc,Unrecoverable) else 'failed_blocks'
                        totals[kind]+=1
                        detail=dict(video_id=row['video_id'],block=number,error=type(exc).__name__,reason=str(exc),record=record)
                        lists['artifact_verification_errors'].append(detail);lists['failed_blocks'].append(detail)
                        continue
                totals['successful_blocks']+=1;totals['retried_successful_blocks']+=record.get('attempts',1)>1
                totals['reused_observed_frames']+=record.get('reused_frames',0)
                totals['context_only_extractions']+=record.get('context_only_extractions',0)
            elif value=='FAILED_AFTER_RETRIES':totals['failed_blocks']+=1
            elif value=='SKIPPED_UNRECOVERABLE':totals['excluded_blocks']+=1
            else:totals['unprocessed_blocks']+=1
            if value in ('FAILED_AFTER_RETRIES','SKIPPED_UNRECOVERABLE'):
                lists['failed_blocks'].append(dict(video_id=row['video_id'],block=number,record=record))
        if artifact_failure and status=='SUCCESS':
            totals['technical_success_videos']-=1;totals['failed_after_retries_videos']+=1
            lists['failed_videos'].append(dict(video_id=row['video_id'],reason='artifact_verification_failed'))
        summary=folder/'summary.json'
        if summary.exists() and load(summary).get('label_mismatch_suspected'):
            totals['label_mismatch_suspected_videos']+=1;lists['label_mismatch_suspected'].append(row['video_id'])
    totals['failed_videos']=totals['failed_after_retries_videos']+totals['structurally_excluded_videos']
    totals['partition_verified']=totals['total_blocks']==sum(totals[k] for k in ('successful_blocks','failed_blocks','excluded_blocks','unprocessed_blocks'))
    totals['block_total_complete']=totals['unplanned_videos']==0 and totals['unprocessed_videos']==0
    assert totals['partition_verified']
    atomic_json(Path(settings['output'])/'audit.json',totals)
    atomic_json(Path(settings['output'])/'failure-and-unprocessed-lists.json',lists)
    review_catalog(settings,rows)
    return totals


def run(args):
    import landmarks
    previous=load(args.previous/'progress.json');configuration=previous['config']
    archive=Path(configuration['video_path'])
    labels=[p for p in args.data.rglob('*morpheme.zip') if 'outputs' not in p.parts]
    if len(labels)!=1:raise Unrecoverable('ambiguous_morpheme_source')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'temp').mkdir(exist_ok=True)
    schema=dict(version='crowd-observed-block15-v1',target_fps=15,block_size=15,endpoint=True,
        landmarks_sha256=digest(landmarks.__file__),mediapipe=landmarks.mp.__version__,opencv=cv2.__version__,
        extraction=dict(model_complexity=0,static_image_mode=False,mirror=False,context_frames=15),
        coordinate_contract='normalized fitted16x9 xy53; MP left/right unchanged',
        origins='existing verified video-tracker frames or current contiguous-tracker frames; explicit context IDs',
        geometry=dict(extended_angle=150,bent_angle=100,stable_palm_lengths_s=.5,stable_degrees_s=25,
                      transition_palm_lengths_s=3,transition_degrees_s=150),
        archive_bytes=archive.stat().st_size,archive_mtime_ns=archive.stat().st_mtime_ns,labels_sha256=digest(labels[0]))
    if schema['landmarks_sha256']!=configuration['landmarks_sha256'] or schema['mediapipe']!=configuration['mediapipe'] or schema['opencv']!=configuration['opencv']:
        raise Unrecoverable('old_feature_preprocessor_compatibility_unconfirmed')
    settings=dict(output=str(args.output.resolve()),archive=str(archive),labels=str(labels[0]),previous=str(args.previous),
        old_completed=previous['completed'],old_rows={r['video_id']:r for r in previous['selected']},old_failed=previous['failed'])
    settings['schema_hash']=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    contract=args.output/'schema.json'
    if contract.exists() and load(contract)!=schema:raise Unrecoverable('run_schema_or_input_changed')
    if not contract.exists():atomic_json(contract,schema)
    rows=inventory(settings)
    inventory_path=args.output/'inventory.json'
    if inventory_path.exists() and load(inventory_path)!=rows:raise Unrecoverable('inventory_changed')
    if not inventory_path.exists():atomic_json(inventory_path,rows)
    source_guard=args.output/'source-guard.json'
    if not source_guard.exists():
        print('Hashing original video ZIP read-only',flush=True)
        atomic_json(source_guard,dict(video_zip_sha256=digest(archive),labels_sha256=digest(labels[0])))
    if args.audit_only:
        print(json.dumps(audit(settings,rows,verify=True)),flush=True)
        guard=load(source_guard)
        assert digest(archive)==guard['video_zip_sha256'] and digest(labels[0])==guard['labels_sha256']
        atomic_json(args.output/'source-integrity.json',dict(original_video_zip_unchanged=True,original_label_zip_unchanged=True,checked_at=now()))
        return
    chosen=[r for r in rows if not args.ids or r['video_id'] in args.ids]
    if args.limit:chosen=chosen[:args.limit]
    # Prior low-fps failures are revisited early, without excluding other videos.
    chosen.sort(key=lambda r:(r['video_id'] not in settings['old_failed'],r['video_id']))
    pending=deque(chosen);completed=[];dispatch_failures=[]
    dispatch_path=args.output/'worker-dispatch-state.json'
    dispatches=load(dispatch_path) if dispatch_path.exists() else {}
    while pending:
        executor=ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=20)
        active={}
        broken=False
        try:
            while pending or active:
                while pending and len(active)<args.workers:
                    row=pending.popleft()
                    if dispatches.get(row['video_id'],0)>=ATTEMPTS:
                        dispatch_failures.append(dict(video_id=row['video_id'],reason='worker_dispatch_exhausted',attempts=ATTEMPTS))
                        completed.append(dispatch_failures[-1]);continue
                    try:
                        active[executor.submit(process_video,row,settings)]=row
                    except Exception as exc:
                        dispatches[row['video_id']]=dispatches.get(row['video_id'],0)+1
                        atomic_json(dispatch_path,dispatches)
                        if dispatches[row['video_id']]<ATTEMPTS:pending.appendleft(row)
                        else:
                            failure=dict(video_id=row['video_id'],technical_status='FAILED_AFTER_RETRIES',
                                error=str(exc),attempts=ATTEMPTS)
                            completed.append(failure);dispatch_failures.append(failure)
                        broken=True;break
                if broken:
                    pending.extend(active.values());break
                if not active:break
                done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    row=active.pop(future)
                    try:
                        result=future.result();completed.append(result)
                        print(json.dumps(dict(completed=len(completed),target=len(chosen),**result)),flush=True)
                    except Exception as exc:
                        dispatches[row['video_id']]=dispatches.get(row['video_id'],0)+1
                        atomic_json(dispatch_path,dispatches)
                        if dispatches[row['video_id']]<ATTEMPTS:pending.append(row)
                        else:
                            failure=dict(video_id=row['video_id'],technical_status='FAILED_AFTER_RETRIES',error=str(exc),attempts=ATTEMPTS)
                            completed.append(failure);dispatch_failures.append(failure)
                        broken=True
                if broken:
                    pending.extend(active.values());break
                if len(completed)%10==0 or not pending and not active:
                    atomic_json(args.output/'session-progress.json',dict(completed=len(completed),target=len(chosen),last=completed[-3:],timestamp=now()))
        finally:
            # Running futures cannot be cancelled. Join them before requeueing
            # their video IDs, otherwise two workers may mutate the same journal.
            executor.shutdown(wait=True,cancel_futures=broken)
        if not broken:break
    atomic_json(args.output/'worker-dispatch-failures.json',dispatch_failures)
    for failure in dispatch_failures:
        row=next(r for r in rows if r['video_id']==failure['video_id'])
        path=args.output/'videos'/row['video_id']/'state.json'
        state=load(path) if path.exists() else dict(video_id=row['video_id'],blocks={},schema_hash=settings['schema_hash'],source_crc=row['crc'])
        state.update(technical_status='FAILED_AFTER_RETRIES',error_code='worker_dispatch_exhausted',dispatch_failures=ATTEMPTS)
        atomic_json(path,state)
    print(json.dumps(audit(settings,rows,verify=True)),flush=True)
    original_guard=load(source_guard)
    print('Verifying original ZIP SHA after processing',flush=True)
    assert digest(archive)==original_guard['video_zip_sha256'] and digest(labels[0])==original_guard['labels_sha256']
    atomic_json(args.output/'source-integrity.json',dict(original_video_zip_unchanged=True,original_label_zip_unchanged=True,checked_at=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--ids',nargs='*');parser.add_argument('--limit',type=int);parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.output.drive.upper()!='D:' or args.output==ROOT or ROOT in args.output.parents or not 1<=args.workers<=4:
        parser.error('new external D output and 1..4 workers required')
    os.environ['TEMP']=str(args.output/'temp');os.environ['TMP']=os.environ['TEMP']
    os.environ['OMP_NUM_THREADS']='1';os.environ['TF_NUM_INTRAOP_THREADS']='1';os.environ['TF_NUM_INTEROP_THREADS']='1'
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'temp').mkdir(exist_ok=True)
    import msvcrt
    with (args.output/'run.lock').open('a+b') as lock:
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        run(args)


if __name__=='__main__':main()
