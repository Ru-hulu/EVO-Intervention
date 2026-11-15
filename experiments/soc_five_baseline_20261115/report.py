"""Coverage-first audit; never substitute a completed subset for a full suite."""
import csv
from .common import *
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics


def summarize(records,expected):
    out=dict(completed=len(records),expected=expected,complete=len(records)==expected,
        observations=sum(r['samples'] for r in records),invalid_frames=sum(r['invalid_predictions'] for r in records),metrics={})
    for k in ('pearson','spearman','VS','ER','C'):
        values=[r[k] for r in records if r[k] is not None]
        out['metrics'][k]=dict(mean=math.fsum(values)/len(values) if values and out['complete'] else None,
            defined=len(values),undefined=len(records)-len(values))
    return out


def report(required_task=None):
    inputs=read(ROOT/'inputs.json');scores=[];missing=[];tasks={}
    forest_inputs=read(FOREST/'inputs.json');assert sha(FOREST/'inputs.json')==inputs['external_forest_inputs_sha256']
    for model in MODELS:
        for suite in SUITES:
            task=f'{model}-{suite}';rr=[];expected=[r for r in inputs['episodes'] if r['suite']==suite]
            for row in expected:
                path=(FOREST_OUT/'episodes'/(row['key'].replace('/','__')+'.json')) if model=='forest' else result_path(model,row['key'])
                if not path.exists():missing.append(dict(model=model,key=row['key']));continue
                r=read(path)
                if model=='forest':
                    from experiments.forest_soc84_baseline_20261115.run import validate as forest_validate,identity as forest_identity
                    forest_validate(r,row,forest_identity(forest_inputs,row),True)
                else:validate(r,row,identity(inputs,row,model),True)
                assert r['initialization']['checkpoint_sha256']==inputs['registry'][task]['checkpoint_sha256']
                metrics=trajectory_metrics(r['predictions'],row['target_pt'],row['timestamps']);assert metrics==r['metrics']
                entry=dict(model=model,suite=suite,task_id=row['task_id'],key=row['key'],**metrics,
                    result_path=str(path),result_sha256=sha(path))
                scores.append(entry);rr.append(entry)
            tasks[task]=summarize(rr,len(expected))
    if required_task:assert tasks[required_task]['complete'],required_task
    value=dict(status='completed' if not missing else 'in_progress',tasks=tasks,episodes=scores,missing=missing,
        inputs_sha256=sha(ROOT/'inputs.json'),protocol=inputs['protocol'],
        completed_tasks=sum(t['complete'] for t in tasks.values()),expected_tasks=15,
        metric_provenance=read(Path(__file__).parents[1]/'prm_five_groups_20261114/metric_reference/provenance.json'))
    save(OUT/'report.json',value)
    cols=['model','suite','task_id','key','pearson','spearman','VS','ER','C','samples','invalid_predictions',
          'correlation_undefined_reason','vs_undefined_reason','result_path','result_sha256']
    with (OUT/'trajectory-metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r.get(k) for k in cols} for r in scores)
    with (OUT/'comparison.csv').open('w',newline='') as f:
        columns=['model','suite','completed','expected','complete','pearson','spearman','VS','ER','C','invalid_frames']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for model in MODELS:
            for suite in SUITES:
                t=tasks[f'{model}-{suite}'];w.writerow(dict(model=model,suite=suite,
                    **{k:t[k] for k in ('completed','expected','complete','invalid_frames')},
                    **{k:v['mean'] for k,v in t['metrics'].items()}))
    return value


if __name__=='__main__':print(report()['tasks'])
