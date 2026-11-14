"""Evidence-linked metric eligibility; never infer a physical failure from PT rollback."""
from .common import *
from experiments.prm_five_groups_20261114.prepare_cache import progress_values
from experiments.prm_five_groups_20261114.recover_metrics import validate_events


def bounds(stage):
    return (stage.get('source_start_frame',stage.get('start_frame')),
            stage.get('source_end_frame',stage.get('end_frame')))


def merge_platforms(intervals):
    """Adjacent repeat actions share one drift anchor; do not absorb idle gaps."""
    merged=[]
    for segment in intervals:
        if merged and segment['start_frame']==merged[-1]['end_frame']+1 and segment['cause']==merged[-1]['cause']:
            merged[-1]['end_frame']=segment['end_frame']
            merged[-1]['source_segments'].append(segment)
        else:
            merged.append(dict(start_frame=segment['start_frame'],end_frame=segment['end_frame'],
                cause=segment['cause'],source_segments=[segment]))
    return merged


def build(row):
    a=read(row['annotation_path']); stages=a['stages']; task=row['task_id']
    evidence=[dict(path=row['annotation_path'],sha256=row['annotation_sha256'])]
    event=dict(schema='prm-recover-events-v1',review_status='verified',key=row['key'],
        video_sha256=row['video_hashes'][0],progress_sha256=row['progress_sha256'],total_frames=row['frames'],
        annotation_provenance=evidence,success=None,failure_onset_frame=None,trr_role=None,abnormal_plateaus=[],
        review_method='Deterministic extraction from accepted visual stage annotations; not a new human review',
        boundary_resolution_seconds=.2,success_evidence=[],failure_reason=None)
    if task=='R01':
        # User-approved semantic milestones end with both cups' final flip/place.
        required={1,2,3,4,5,6,7,8,9,10,11,12,13}
        if required.issubset({s['stage_id'] for s in stages}):
            event['success']=True;event['success_evidence']=['All 13 annotated milestones including restoration and final flip/place are present']
        failure=[s for s in stages if s['stage_id']==6 and '碰倒' in s['stage_name']]
        if len(failure)==1:
            event.update(failure_onset_frame=bounds(failure[0])[0],trr_role='r_positive',
                failure_evidence=failure[0])
    elif task=='R03':
        last={}
        for s in stages:
            if s['action'] in ('table_to_rack','recover_to_rack'):last[s['object_position']]=s
        if set(last)=={'left','middle','right'} and all(s['outcome']=='completed' for s in last.values()):
            event['success']=True;event['success_evidence']=list(last.values())
        failed=[s for s in stages if s.get('outcome')=='failed_bottle_fell']
        if failed:
            event['failure_reason']='Physical fall onset is not a separate annotated frame; return-action start and PT rollback at recovery must not substitute'
            event['failure_action_segments']=failed
            review_path=ROOT/'event-review'/row['sample_id']/'review.json'
            if review_path.exists():
                review=read(review_path)
                assert review['video_sha256']==row['video_hashes'][0] and not review['predictions_consulted']
                assert review['stage_annotation_sha256']==row['annotation_sha256']
                for path,digest in review['source_frame_sha256'].items():assert sha(path)==digest
                assert len(failed)==1
                start,end=bounds(failed[0]);assert start<=review['failure_onset_frame']<=end
                event.update(failure_onset_frame=review['failure_onset_frame'],trr_role='r_positive',
                    failure_reason=None,failure_evidence=review)
                event['annotation_provenance'].append(dict(path=str(review_path),sha256=sha(review_path)))
        else:event['trr_role']='normal';event['failure_reason']='No explicitly annotated fall'
    elif task=='R02':
        required={(c,a) for c in ('purple','red','green') for a in ('pick_up','open_zipper','close_zipper','put_down_and_retract')}
        actual={(s.get('object_color'),s.get('action')) for s in stages if s.get('object_occurrence')==1}
        # Do not assume color names; stage definitions provide the three annotated objects.
        colors={s.get('object_color') for s in stages if s.get('object_color')}
        required={(c,a) for c in colors for a in ('pick_up','open_zipper','close_zipper','put_down_and_retract')}
        if len(colors)==3 and required<=actual:
            event['success']=True;event['success_evidence']=['Each of three annotated objects has its complete first inspection cycle']
        event['trr_role']='normal';event['failure_reason']='Repeated inspections; no annotated regressive failure event'
    elif task=='R04':
        final={}
        for s in stages:final[s['target_position']]=s
        if set(final)=={'top','middle','bottom'} and all(s['stage_id'] in (2,4,6) for s in final.values()):
            event['success']=True;event['success_evidence']=list(final.values())
        event['trr_role']='normal';event['failure_reason']='Repeated inspections; no annotated regressive failure event'
    elif task=='R05':
        from experiments.prm_five_groups_20261114.recover_pour_labels import first_visit_mask
        _,segments=first_visit_mask(a)
        event['success']=True;event['success_evidence']=['First pour to each of left/middle/right completed under accepted stage annotation']
        event['trr_role']='normal';event['failure_reason']='Repeated pour; no annotated regressive failure event'
    else:raise ValueError(task)
    for s in stages:
        repeat=(task=='R02' and s.get('object_occurrence',0)>1 or
                task=='R04' and s.get('cycle_index',0)>1 or
                task=='R05' and s.get('target_occurrence',0)>1)
        if repeat:
            start,end=bounds(s)
            event['abnormal_plateaus'].append(dict(start_frame=start,end_frame=end,cause='repeated_action',
                stage_id=s['stage_id'],segment_index=s.get('segment_index'),source='accepted stage occurrence'))
    event['abnormal_plateaus']=merge_platforms(event['abnormal_plateaus'])
    validate_events(event,row,progress_values(row))
    return event


def main():
    receipt=[]
    for row in read(ROOT/'inputs.json')['records']:
        path=ROOT/'events'/(row['key'].replace('/','__')+'.json')
        if path.exists():
            validate_events(read(path),row,progress_values(row));event=read(path)
        else:event=build(row);save(path,event)
        receipt.append(dict(key=row['key'],success=event['success'],failure_onset_frame=event['failure_onset_frame'],
            failure_reason=event['failure_reason'],platform_segments=len(event['abnormal_plateaus']),sha256=sha(path)))
    save(ROOT/'event-readiness.json',dict(records=receipt,physical_failure_onsets_missing=[r['key'] for r in receipt
        if r['failure_reason'] and 'not a separate' in r['failure_reason']]))


if __name__=='__main__':main()
