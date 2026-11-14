"""Audit all 252 curves; report episode-equal VOC, official VS, and user-defined SA."""
from collections import Counter
import csv
import math
import time
from .common import *
from .protocol import metrics,validate_record

METRICS=('pearson','spearman','VS','ER','C','SA')

def summary(rows,expected):
    out=dict(episodes=len(rows),expected_episodes=expected,complete=len(rows)==expected,metrics={},
        invalid_frames=sum(r['invalid_predictions'] for r in rows))
    for m in METRICS:
        values=[r[m] for r in rows if r[m] is not None]
        out['metrics'][m]=dict(mean=math.fsum(values)/len(values) if values and len(rows)==expected else None,
            defined_episodes=len(values),undefined_episodes=len(rows)-len(values),
            reasons=dict(Counter(r.get('SA_reason') if m=='SA' else r.get('vs_undefined_reason') if m in ('VS','ER','C') else r.get('correlation_undefined_reason')
                for r in rows if r[m] is None)))
    return out

def main():
    source=read(ROOT/'inputs.json');inputs_sha=sha(ROOT/'inputs.json');policy=read(ROOT/'success-policy.json')
    assert policy['all_selected_episodes_successful'] is True
    OUT.mkdir(parents=True,exist_ok=True);reports={};flat=[];points=[]
    for model in MODELS:
        records=[]
        for row in source['episodes']:
            info=source['registry'][job_name(model,row['suite'])];path=result_path(model,row['key']);r=read(path)
            identity=dict(model=model,suite=row['suite'],checkpoint_sha256=info['checkpoint_sha256'],inputs_sha256=inputs_sha,
                key=row['key'],condition='no_memory',backend_sha256=sha(Path(__file__).with_name('backend.py')))
            validate_record(r,row,identity,True)
            audit=read(r['initialization_audit']);assert audit['strict_final_state'] and audit['checkpoint_sha256']==info['checkpoint_sha256']
            assert audit['suite']==row['suite'] and audit['memory_enabled'] is False
            score=metrics(r['predictions'],row,True);assert r['metrics']==score
            entry=dict(model=model,key=row['key'],suite=row['suite'],task_id=row['task_id'],**score,result_sha256=sha(path))
            records.append(entry);flat.append({k:entry.get(k) for k in ['model','suite','task_id','key',*METRICS,'invalid_predictions','correlation_undefined_reason','SA_reason']})
            tail=set(score['endpoint_indices'])
            for i,(frame,t,pt,pred,raw) in enumerate(zip(row['frame_indices'],row['timestamps'],row['target_pt'],r['predictions'],r['raw_responses'])):
                points.append(dict(model=model,suite=row['suite'],task_id=row['task_id'],key=row['key'],
                    frame_index=frame,timestamp=t,target_pt=pt,prediction=pred,SA_tail=i in tail,raw_response=raw))
        reports[model]=dict(overall=summary(records,84),by_suite={s:summary([r for r in records if r['suite']==s],
            sum(e['suite']==s for e in source['episodes'])) for s in SUITES},
            by_task={t:summary([r for r in records if r['task_id']==t],2) for t in sorted({r['task_id'] for r in records})},episodes=records)
    assert len(points)==45288 and len(flat)==252
    result=dict(status='completed',models=reports,protocol=source['protocol'],success_policy=policy,
        inputs_sha256=inputs_sha,checkpoint_registry=source['registry'],metric_provenance=source['metric_provenance'],
        coverage=dict(models=3,episodes_per_model=84,curves=252,predictions=45288),finished=time.time())
    save(OUT/'metrics.json',result)
    for name,rows in [('episode_metrics.csv',flat),('predictions.csv',points)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparison=[]
    for model in MODELS:
        for suite in SUITES:
            x=reports[model]['by_suite'][suite]
            comparison.append(dict(model=model,suite=suite,episodes=x['episodes'],**{m:x['metrics'][m]['mean'] for m in METRICS}))
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    print('Completed: 9 suite/model runs, 252 curves, 45288 predictions; no-memory only.')

if __name__=='__main__':main()
