"""Compare cached predictions without training or reference-guided rewriting."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_mediapipe_ctc as base
from fingerspelling_runtime import render,vocabulary
from fingerspelling import jamo


def rendering_metrics(videos,references,mode):
    if not videos or not references:raise ValueError('empty_evaluation')
    if {v['video_id'] for v in videos}!=set(references):raise ValueError('prediction_reference_ID_mismatch')
    if len(videos)!=len(references):raise ValueError('duplicate_prediction_ID')
    counts=[];exact=0
    for video in videos:
        tokens=list(video['prediction_jamo'])
        text=render(vocabulary().encode(tokens),mode)
        if jamo.decompose_hangul(text)!=tokens:raise ValueError('rendering_changed_tokens')
        counts.append(base.edit_counts(references[video['video_id']],text))
        exact+=text==references[video['video_id']]
    return dict(cer=sum(c.errors for c in counts)/sum(c.reference_characters for c in counts),
        D=sum(c.deletions for c in counts),I=sum(c.insertions for c in counts),S=sum(c.substitutions for c in counts),
        N=sum(c.reference_characters for c in counts),exact_match=exact/len(videos),tokens_preserved=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.resolve().drive.upper()!='D:' or ROOT in args.output.resolve().parents:
        p.error('output must be outside repository on D drive')
    summary=base.read(args.experiment/'summary.json')
    selected=summary['selected']
    candidate=base.read(args.experiment/selected/'result.json')
    config=base.read(args.source/'training-config.json')
    references={split:{r['video_id']:r['text'] for r in rows} for split,rows in config['splits'].items()}
    validation={mode:rendering_metrics(candidate['validation']['videos'],references['validation'],mode)
                for mode in ('strict','prefix','resync')}
    # Only Validation enters the rendering choice. Test is read afterwards.
    mode=min(validation,key=lambda m:(validation[m]['cer'],('strict','prefix','resync').index(m)))
    heldout=base.read(args.experiment/'heldout-diagnostic.json')
    if heldout['selected']!=selected or heldout['checkpoint_sha256']!=base.digest(args.experiment/selected/'inference.pt'):
        raise ValueError('heldout_checkpoint_mismatch')
    results=dict(selected_model=selected,selected_render_mode=mode,validation=validation,
        train={m:rendering_metrics(candidate['train']['videos'],references['train'],m) for m in validation},
        test_diagnostic={m:rendering_metrics(heldout['videos'],references['test'],m) for m in validation},
        selection='Validation CER only; no token insertion/deletion or dictionary',test_previously_observed=True,
        performer_independent=False,checkpoint_sha256=heldout['checkpoint_sha256'])
    with args.output.open('x',encoding='utf-8') as stream:json.dump(results,stream,ensure_ascii=False,indent=2)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
