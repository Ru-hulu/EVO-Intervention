"""Audit every frozen input and result; aggregate only complete trajectory coverage."""
import math
from collections import Counter
from .common import *
from .worker import validate_prefix
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics
from experiments.prm_five_groups_20261114.prepare_cache import progress_values
from experiments.prm_five_groups_20261114.recover_metrics import score_recover,validate_events

METRICS=('pearson','spearman','successful_endpoint_score','fpl','abnormal_plateau_absolute_mae','abnormal_plateau_drift_mae')


def summary(records,expected):
    result=dict(episodes=len(records),expected_episodes=expected,complete=len(records)==expected,
        invalid_frames=sum(r['invalid_predictions'] for r in records),metrics={})
    for name in METRICS:
        valid=[r[name] for r in records if r.get(name) is not None]
        result['metrics'][name]=dict(mean=math.fsum(valid)/len(valid) if valid and len(records)==expected else None,
            defined_episodes=len(valid),undefined_episodes=len(records)-len(valid),
            reasons=dict(Counter(r.get('metric_reasons',{}).get(name,r.get('correlation_undefined_reason') or 'not_eligible')
                for r in records if r.get(name) is None)))
    return result


def main():
    frozen=read(ROOT/'inputs.json');digest=sha(ROOT/'inputs.json');original=read(ROOT/'selection.json')
    assert sha(ROOT/'selection.json')==frozen['selection_sha256']
    for path,expected in original['sources'].items():assert sha(path)==expected
    for row in frozen['records']:
        assert sha(row['progress_path'])==row['progress_sha256']
        truth=progress_values(row);assert row['target_pt']==[truth[i] for i in row['frame_indices']]
    reports={}
    for name in MODELS:
        info=frozen['model_checkpoints'][name];assert sha(info['proof'])==info['proof_sha256']
        records=[];missing=[]
        for row in frozen['records']:
            path=result_path(name,row['key'])
            if not path.exists():missing.append(row['key']);continue
            r=read(path);assert validate_prefix(r,row,digest,info['checkpoint_sha256'])==len(row['frame_indices'])
            assert r['source_video_sha256']==row['video_hashes'] and r['progress_sha256']==row['progress_sha256']
            assert r['timestamps']==row['timestamps']
            audit=read(r['initialization_audit']);assert audit['strict_final_state']
            assert audit['checkpoint_sha256']==info['checkpoint_sha256'] and audit['inputs_sha256']==digest
            assert audit['uses_future_endpoint_reference']==(name in ('robobrain25-4b','dopamine-3b'))
            metrics=trajectory_metrics(r['predictions'],row['target_pt'],row['timestamps'])
            ep=ROOT/'events'/(row['key'].replace('/','__')+'.json');events=read(ep) if ep.exists() else None
            if events:validate_events(events,row,progress_values(row))
            extra=score_recover(r['predictions'],row['target_pt'],row['frame_indices'],row['timestamps'],row['frames'],events)
            # User requested last 2% of sampled frames, with ceil and at least one.
            if events and events.get('success') is True:
                n=max(1,math.ceil(.02*len(r['predictions'])));tail=r['predictions'][-n:]
                extra['successful_endpoint_score']=math.fsum(tail)/n if all(v is not None and math.isfinite(v) for v in tail) else None
                extra['endpoint_scored_indices']=list(range(len(r['predictions'])-n,len(r['predictions'])))
            records.append(dict(key=row['key'],task_id=row['task_id'],**metrics,**extra,result_sha256=sha(path),
                event_annotation_sha256=sha(ep) if ep.exists() else None))
        reports[name]=dict(summary=summary(records,10),by_task={t:summary([r for r in records if r['task_id']==t],2)
            for t in frozen['selection']},episodes=records,missing=missing,checkpoint=info['checkpoint'],
            uses_future_endpoint_reference=name in ('robobrain25-4b','dopamine-3b'))
    ready=read(ROOT/'event-readiness.json') if (ROOT/'event-readiness.json').exists() else {}
    out=dict(status='completed' if all(not r['missing'] for r in reports.values()) and not ready.get('physical_failure_onsets_missing',True)
        else 'predictions_or_annotations_pending',models=reports,expected_models=5,expected_episodes_per_model=10,
        inputs_sha256=digest,metric_aggregation='equal trajectory weight; eligible denominators explicit',event_readiness=ready)
    save(OUT/'report.json',out)
    lines=['# Recovery: five fine-tuned methods, two held-out trajectories per task',
        '', '| Method | Coverage | Pearson | Spearman | Endpoint | FPL | Platform MAE | Drift MAE |',
        '|---|---|---|---|---|---|---|---|']
    for name,r in reports.items():
        values=[]
        for metric in METRICS:
            value=r['summary']['metrics'][metric];values.append('N/A' if value['mean'] is None else f"{value['mean']:.6f} (n={value['defined_episodes']})")
        lines.append('| '+' | '.join([name,f"{r['summary']['episodes']}/10",*values])+' |')
    lines+=['','RoboBrain and Dopamine use this trajectory\'s true future endpoint reference.',
        'Endpoint is not classification accuracy. Undefined/invalid/not-applicable denominators are retained in report.json.',
        'Missing physical failure boundaries: '+str(ready.get('physical_failure_onsets_missing',[]))]
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(out['status'])


if __name__=='__main__':main()
