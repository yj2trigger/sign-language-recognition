"""Optional 424-wide wrist/palm representation; preserve global wrist motion."""
import hashlib
from pathlib import Path
import numpy as np


def hand_local_schema():
    from fingerspelling_runtime import feature_schema
    return dict(feature_schema(),hand_local=dict(version=1,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        palm_joint=9,minimum_scale=1e-6,wrist_position='global_shoulder_normalized'))


def hand_local_features(values):
    values=np.asarray(values)
    if values.ndim!=2 or values.shape[1]!=424 or not len(values) or values.dtype!=np.float32 or not np.isfinite(values).all():
        raise ValueError('invalid_base_feature_matrix')
    output=values.reshape(len(values),53,8).copy()
    if not np.isin(output[:,:,4],(0.,1.)).all():raise ValueError('invalid_available_channel')
    for start in (11,32):
        hand=output[:,start:start+21]
        available=hand[:,:,4].astype(bool)
        xy=hand[:,:,:2].astype(np.float64)
        scale=np.linalg.norm(xy[:,9]-xy[:,0],axis=1)
        reference=available[:,0]&available[:,9]&(scale>1e-6)
        relative=(xy-xy[:,:1])/np.where(reference,scale,1.)[:,None,None]
        usable=available.copy();usable[:,1:]&=reference[:,None]
        relative[:,0]=xy[:,0]  # Keep shoulder-relative whole-hand position.
        relative[~usable]=0.
        motion=np.zeros_like(relative)
        valid_motion=np.zeros_like(usable)
        valid_motion[1:]=usable[1:]&usable[:-1]
        motion[1:]=relative[1:]-relative[:-1];motion[~valid_motion]=0.
        if (not np.isfinite(relative).all() or not np.isfinite(motion).all()
                or max(np.abs(relative).max(),np.abs(motion).max())>np.finfo(np.float32).max):
            raise ValueError('hand_local_overflow')
        hand[:,:,:2]=relative;hand[:,:,4]=usable
        hand[:,:,5:7]=motion;hand[:,:,7]=valid_motion
    return output.reshape(len(values),424)
