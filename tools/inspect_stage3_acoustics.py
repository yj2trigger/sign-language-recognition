"""Local label-free posterior/hand-feature ambiguity diagnostics; no correction."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

import run_issue2_stage3 as stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    if args.output.drive.upper()!='D:' or args.output.exists():parser.error('new local D output required')
    stage.torch.set_num_threads(2)
    data=stage.stage2.previous.training_data(args.source,stage.exp.base.read(args.source/'training-config.json'))
    pairs=Counter();frames=0;ambiguous=0;availability=[]
    for row,x,y,logits in stage.logits_for(args.checkpoint,data['validation']):
        top=np.argsort(logits[:,1:],axis=1)[:,-2:]+1
        nonblank=logits.argmax(-1)!=0
        margin=logits[np.arange(len(x)),top[:,1]]-logits[np.arange(len(x)),top[:,0]]
        low=nonblank&(margin<.5)
        for a,b in top[low]:pairs['/'.join(sorted(stage.exp.base.tokens_for([int(a),int(b)])))]+=1
        frames+=int(nonblank.sum());ambiguous+=int(low.sum())
        availability.append(float(x.reshape(len(x),53,8)[:,11:,4].mean()))
    summary=dict(nonblank_frames=frames,ambiguous_nonblank_frames=ambiguous,
                 margin_threshold=.5,top2_nonblank_pairs=dict(pairs),
                 mean_hand_joint_availability=float(np.mean(availability)),
                 interpretation='posterior ambiguity is not verified visual sign confusion',
                 labels_used_for_correction=False,test_evaluated=False)
    stage.exp.base.write(args.output,summary)
    print(summary)


if __name__=='__main__':main()
