"""Build a blinded review queue from saved results. No video extraction or inference."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import zlib

STATUSES={'VERIFIED','MISMATCH','UNCERTAIN','NOT_FINGERSPELLING','UNREVIEWED'}
MAPPING_METHODS={'official_metadata','provider_confirmation'}
TRANSCRIPT_METHODS={'official_transcript','human_independent_reading'}


def canonical(value):
    if not isinstance(value,str) or not re.fullmatch(r'NIA_SL_FS\d{4}_CROWD\d{2}_F_?',value):
        raise ValueError('unsupported source video ID')
    return value.rstrip('_')


def index_unique(rows):
    result={}
    for row in rows:
        key=canonical(row['video_id'])
        if key in result:raise ValueError('duplicate or aliased video ID')
        result[key]=row
    return result


def nonempty_text(value):
    return isinstance(value,str) and bool(value.strip())


def is_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def mapping_valid(row):
    return bool(row and row.get('mapping_status')=='VERIFIED'
        and row.get('mapping_method') in MAPPING_METHODS and nonempty_text(row.get('performer_id'))
        and row['performer_id'].strip().casefold() not in {'-1','unknown','none','n/a','unidentified'}
        and nonempty_text(row.get('mapping_source')) and nonempty_text(row.get('approved_by'))
        and is_sha(row.get('evidence_sha256')))


def review_valid(row):
    if row.get('review_status') not in STATUSES:raise ValueError('invalid review status')
    if isinstance(row.get('start'),bool) or isinstance(row.get('end'),bool):return False
    try:
        start,end=float(row['start']),float(row['end'])
    except (KeyError,ValueError,TypeError):return False
    return bool(row['review_status']=='VERIFIED' and row.get('transcript_basis') in TRANSCRIPT_METHODS
        and nonempty_text(row.get('reviewer')) and nonempty_text(row.get('evidence_source')) and nonempty_text(row.get('transcript_text'))
        and is_sha(row.get('evidence_sha256'))
        and math.isfinite(start) and math.isfinite(end) and 0<=start<end)


def eligibility(training_ids,mapping_rows,review_rows,training_hashes=()):
    """Validate declarations; this cannot authenticate an external provider/reviewer."""
    train={canonical(value) for value in training_ids}
    if len(train)!=len(training_ids):raise ValueError('duplicate training video ID')
    mappings=index_unique(mapping_rows);reviews=index_unique(review_rows)
    missing_train=sorted(vid for vid in train if not mapping_valid(mappings.get(vid)))
    train_people={mappings[vid]['performer_id'].strip() for vid in train if mapping_valid(mappings.get(vid))}
    decisions=[]
    for vid,row in reviews.items():
        reasons=[]
        if row.get('review_status') not in STATUSES:raise ValueError('invalid review status')
        if missing_train:reasons.append('TRAIN_PERFORMER_MAPPING_INCOMPLETE')
        if not mapping_valid(mappings.get(vid)):reasons.append('UNVERIFIED_PERFORMER_MAPPING')
        elif mappings[vid]['performer_id'].strip() in train_people:reasons.append('TRAIN_PERFORMER_OVERLAP')
        if vid in train:reasons.append('TRAIN_VIDEO_OVERLAP')
        if row.get('source_video_id') and canonical(row['source_video_id']) in train:reasons.append('TRAIN_SOURCE_OVERLAP')
        if not review_valid(row):reasons.append('INDEPENDENT_TRANSCRIPT_UNVERIFIED')
        feature_hash=row.get('feature_sha256','')
        if is_sha(feature_hash) and feature_hash in training_hashes:reasons.append('TRAIN_FEATURE_DUPLICATE')
        if not is_sha(feature_hash) or row.get('feature_contract_verified')!='true':
            reasons.append('FEATURE_CONTRACT_UNVERIFIED')
        decisions.append(dict(video_id=vid,declared_ready=not reasons,reasons=reasons))
    return dict(training_mapping_missing=len(missing_train),missing_training_ids=missing_train,
        declared_ready_count=sum(d['declared_ready'] for d in decisions),decisions=decisions,
        independent_performer_count=None,independent_evaluation_videos=0,jamo_CER=None,hangul_CER=None,
        exact_match=None,interpretation='Declaration checks only; external authenticity and actual feature arrays require independent validation before inference.')


def write_csv(path,rows,columns):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({key:row.get(key,'') for key in columns})


def verified_existing_video(path,row):
    path=Path(path)
    if path.drive.upper()!='D:' or not path.is_file() or path.stat().st_size!=row['bytes']:return False
    crc=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):crc=zlib.crc32(chunk,crc)
    return crc==row['crc']


def build(blocks,previous,output):
    # A dedicated new directory prevents overwriting any reviewer's work.
    output.mkdir(parents=True,exist_ok=False)
    inventory=json.loads((blocks/'inventory.json').read_text(encoding='utf-8'))
    config=json.loads((previous/'training-config.json').read_text(encoding='utf-8'))
    prior=json.loads((previous/'progress.json').read_text(encoding='utf-8'))
    training=[row['video_id'] for row in config['splits']['train']]
    mapping=[];reviews=[];expected=[];links=[];original_links=0
    for row in inventory:
        vid=canonical(row['video_id'])
        mapping.append(dict(video_id=vid,mapping_status='UNCONFIRMED'))
        folder=blocks/'videos'/vid
        if not (folder/'state.json').exists():continue
        state=json.loads((folder/'state.json').read_text(encoding='utf-8'))
        if state['technical_status']!='SUCCESS':continue
        labels=row['labels'];start=min(label['start'] for label in labels);end=max(label['end'] for label in labels)
        reviewed=vid in {'NIA_SL_FS0001_CROWD18_F','NIA_SL_FS0001_CROWD19_F','NIA_SL_FS0735_CROWD18_F'}
        task='review-'+hashlib.sha256(vid.encode()).hexdigest()[:16]
        review=dict(video_id=vid,task_id=task,source_video_id=vid,start=start,end=end,
            review_status='UNCERTAIN' if reviewed else 'UNREVIEWED')
        reviews.append(review)
        expected.append(dict(video_id=vid,original_label=' | '.join(label['text'] for label in labels)))
        page=['<!doctype html><meta charset="utf-8"><title>'+task+'</title>',
            '<h1>'+task+'</h1><p>Transcribe independently before opening the separate label adjudication file.</p>',
            '<p>Reference interval '+str(start)+' - '+str(end)+' seconds; verify its boundaries yourself. Blocks are not letters.</p>']
        existing=prior.get('failed',{}).get(vid,{}).get('temporary_video_preserved')
        if existing and verified_existing_video(existing,row):
            page.append('<p>Existing CRC-verified original MP4; no new extraction.</p><video controls preload="none" src="'+
                html.escape(Path(existing).resolve().as_uri(),quote=True)+'"></video>')
            original_links+=1
        for record in sorted(state['blocks'].values(),key=lambda b:b['block']):
            if record['status']!='VERIFIED_TECHNICAL':raise ValueError('success video with incomplete blocks')
            image=Path(record['review_image'])
            if not image.is_file():raise ValueError('missing saved RGB contact sheet')
            page.append('<p>Block '+str(record['block'])+' PTS '+str(record['PTS_start'])+' - '+str(record['PTS_end'])+'</p>')
            page.append('<img loading="lazy" src="'+html.escape(image.resolve().as_uri(),quote=True)+'" alt="Saved RGB observations">')
        (output/(task+'.html')).write_text('\n'.join(page),encoding='utf-8')
        links.append('<li><a href="'+task+'.html">'+task+'</a></li>')
    write_csv(output/'performer-mapping-template.csv',mapping,
        ['video_id','performer_id','mapping_status','mapping_method','mapping_source','evidence_sha256','approved_by'])
    write_csv(output/'blind-transcription-template.csv',reviews,
        ['task_id','video_id','source_video_id','start','end','review_status','transcript_text','transcript_basis',
         'reviewer','evidence_source','evidence_sha256','feature_sha256','feature_contract_verified','review_notes'])
    write_csv(output/'adjudication-private-do-not-show-before-transcription.csv',expected,['video_id','original_label'])
    (output/'blind-index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Independent transcription queue</h1><ul>'+''.join(links)+'</ul>',encoding='utf-8')
    hashes={hashlib.sha256((previous/'features'/(vid+'.npz')).read_bytes()).hexdigest() for vid in training}
    result=eligibility(training,mapping,reviews,hashes)
    result['review_states']={status:sum(r['review_status']==status for r in reviews) for status in sorted(STATUSES)}
    result['mapping_template_rows']=len(mapping);result['blinded_review_pages']=len(reviews)
    result['verified_existing_original_video_links']=original_links
    (output/'readiness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.resolve().drive.upper()!='D:':parser.error('D output required')
    result=build(args.blocks,args.previous,args.output)
    print(json.dumps({key:value for key,value in result.items() if key not in ('decisions','missing_training_ids')},ensure_ascii=False))
