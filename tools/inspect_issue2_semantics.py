"""Continuous 2-D geometry review from existing blocks; no CTC or video decoder."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
         (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
PROBES = {'NIA_SL_FS0001_CROWD18_F':[1.4666667,1.9333333,2.9333333,3.4666667,3.9333333,4.4666667,5.4666667],
          'NIA_SL_FS0001_CROWD19_F':[3.4666667,3.9333333,4.4666667,5.,5.4666667,5.9333333,6.9333333],
          'NIA_SL_FS0735_CROWD18_F':[1.2666667,1.7666667,2.4666667,3.7]}


def gif_delays(times):
    times=np.asarray(times,np.float64)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('invalid observed PTS')
    ticks=np.rint((times-times[0])*100).astype(np.int64)
    delays=np.diff(ticks)*10
    if (delays<10).any():raise ValueError('GIF cannot resolve these observation intervals')
    return delays.tolist()+[10]


def hand_measures(points, valid, aspect=16/9):
    points=np.asarray(points);valid=np.asarray(valid)
    if points.shape!=(21,2) or points.dtype.kind not in 'fiu' or valid.shape!=(21,) or valid.dtype!=np.bool_:
        raise ValueError('expected 21 real xy points and boolean mask')
    if not np.isfinite(points).all() or (np.abs(points)>np.finfo(np.float32).max).any():
        raise ValueError('invalid coordinates')
    if not np.isfinite(aspect) or aspect<=0:raise ValueError('invalid aspect')
    points=points.astype(np.float64,copy=True)
    points[:,0]*=aspect
    palm=np.linalg.norm(points[9]-points[0])
    angles=[]
    for finger in range(5):
        chain=list(range(1+4*finger,5+4*finger));pair=[]
        for a,b,c in (chain[:3],chain[1:]):
            u=points[a]-points[b];v=points[c]-points[b]
            denominator=np.linalg.norm(u)*np.linalg.norm(v)
            pair.append(float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denominator,-1,1))))
                        if valid[[a,b,c]].all() and denominator>1e-12 else None)
        angles.append(pair)
    distances=[]
    for a,b in ((4,8),(8,12),(12,16),(16,20)):
        distances.append(float(np.linalg.norm(points[a]-points[b])/palm)
            if valid[[0,9,a,b]].all() and palm>1e-6 else None)
    return dict(joint_angles_degrees=angles,adjacent_tip_distances_palm_units=distances,
        valid_joint_count=int(valid.sum()),
        projected_palm_signed_area=float(np.cross(points[5]-points[0],points[17]-points[0]))
            if valid[[0,5,17]].all() else None,
        palm_facing_3D='UNRESOLVED: xy has no depth or RGB palm/back classification',
        confidence='boolean detection mask only; not a calibrated joint confidence')


def inspect(blocks,output):
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for video_id,probes in PROBES.items():
        folder=blocks/'videos'/video_id
        state=json.loads((folder/'state.json').read_text(encoding='utf-8'))
        parts=[]
        for block in sorted(state['blocks'].values(),key=lambda item:item['block']):
            if block['status']!='VERIFIED_TECHNICAL':raise ValueError('incomplete selected video')
            with np.load(block['path'],allow_pickle=False) as data:
                parts.append({key:data[key].copy() for key in ('xy','mask','t','selected_index','context')})
        combined={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
        xy=combined['xy'];mask=combined['mask'];times=combined['t']
        if not np.array_equal(combined['selected_index'],np.arange(len(times))):raise ValueError('selected observation gap')
        observations=[]
        for probe in probes:
            index=int(np.argmin(np.abs(times-probe)))
            if abs(times[index]-probe)>1e-5:raise ValueError('probe not an actual saved PTS')
            observations.append(dict(PTS=float(times[index]),selected_frame=index+1,
                left=hand_measures(xy[index,11:32],mask[index,11:32]),
                right=hand_measures(xy[index,32:53],mask[index,32:53])))
        frames=[]
        # Static fitted-coordinate view, unchanged anatomical order and image y-axis.
        for i,(points,valid) in enumerate(zip(xy,mask)):
            image=Image.new('RGB',(640,400),'white');draw=ImageDraw.Draw(image)
            draw.text((8,8),f'{video_id} PTS {times[i]:.6f} block {i//15+1}',fill='black')
            for offset,color in ((11,'blue'),(32,'red')):
                hand=points[offset:offset+21]*np.array([640,360])+[0,30]
                for a,b in EDGES:
                    if valid[offset+a] and valid[offset+b]:draw.line([tuple(hand[a]),tuple(hand[b])],fill=color,width=2)
                for point,observed in zip(hand,valid[offset:offset+21]):
                    if observed:draw.ellipse((point[0]-2,point[1]-2,point[0]+2,point[1]+2),fill=color)
            frames.append(image)
        durations=gif_delays(times)
        frames[0].save(output/(video_id+'.gif'),save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
        with Image.open(output/(video_id+'.gif')) as restored:
            if restored.n_frames!=len(times):raise ValueError('GIF observation count changed')
            actual=[]
            for index in range(restored.n_frames):
                restored.seek(index);actual.append(restored.info.get('duration',0))
            if actual!=durations:raise ValueError('GIF duration readback changed')
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        records.append(dict(video_id=video_id,original_labels=summary['original_labels'],observed_frames=len(times),
            source_blocks=len(parts),probe_count=len(probes),probes=observations,
            status='UNCERTAIN',independent_reading='official chart plus RGB/geometry comparison; no model prediction',
            missing_selected_observations=0,context_changes=int((np.diff(combined['context'])!=0).sum()),
            observed_interval_range_seconds=[float(np.diff(times).min()),float(np.diff(times).max())],
            observed_PTS=times.tolist(),animation_last_display_ms=10,
            max_GIF_cumulative_error_ms=float(np.max(np.abs(np.cumsum(actual[:-1])-(times[1:]-times[0])*1000))) if len(times)>1 else 0.,
            animation_timing='10ms GIF quantization of cumulative observed PTS; final 10ms is display only; original PTS retained'))
    result=dict(video_attempts=len(records),jamo_probe_attempts=sum(r['probe_count'] for r in records),
        verified=0,suspected_mismatch=0,uncertain=len(records),technical_failure=0,label_corrections=0,
        records=records,interpretation='Geometric measurements are not a complete alphabet classifier or independent transcript.')
    (output/'semantics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().drive.upper()!='D:':parser.error('D output required')
    result=inspect(args.blocks,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
