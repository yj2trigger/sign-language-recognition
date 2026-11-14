"""Use decoded presentation times, including variable frame rate videos."""
import cv2
import numpy as np
import landmarks


def timestamp_indices(times, target_fps=15):
    times=np.asarray(times,dtype=np.float64)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('invalid_decoded_timestamps')
    if times[0]<0 or not np.isfinite(target_fps) or target_fps<=0:
        raise ValueError('invalid_sampling_clock')
    grid=np.arange(int(np.floor((times[-1]-times[0])*target_fps))+1)/target_fps+times[0]
    # One actual frame at or after each 15Hz tick. Never duplicate a source frame.
    return np.unique(np.searchsorted(times,grid-1e-7)).astype(np.int64)


def probe_video(path):
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wh=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if not np.isfinite(fps) or fps<15 or count<2 or min(wh)<1:
            raise ValueError('invalid_video_metadata')
        times=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if len(times)!=count:
            raise ValueError('incomplete_video_decode')
        indices=timestamp_indices(times)
        times=np.asarray(times)
        return dict(src_fps=fps,source_frames=count,resolution=wh,times=times,indices=indices,
                    cfr_error_seconds=float(np.max(np.abs(times-np.arange(count)/fps))))
    finally:cap.release()


def extract_timestamped(path,probe):
    cap=cv2.VideoCapture(str(path))
    selected=set(probe['indices'].tolist())
    xys,masks,times=[],[],[]
    try:
        with landmarks.mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=0) as holistic:
            for index in range(probe['source_frames']):
                if index not in selected:
                    if not cap.grab():raise ValueError('incomplete_second_decode')
                    continue
                ok,frame=cap.read()
                if not ok:raise ValueError('incomplete_second_decode')
                pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if abs(pts-probe['times'][index])>1e-6:raise ValueError('unstable_decoder_timestamp')
                xy,mask,z,wh=landmarks.process_frame(holistic,frame)
                xys.append(xy);masks.append(mask);times.append(pts)
    finally:cap.release()
    return dict(xy=np.stack(xys),mask=np.stack(masks),t=np.asarray(times,dtype=np.float64),
                wh=wh,src_fps=probe['src_fps'])
