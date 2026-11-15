"""Audit all frozen mappings and every prediction; never fill invalid outputs."""
import csv
import time
from .common import *
from experiments.group5_memoryfps_20261115.protocol import validate_record, scores
from experiments.group5_memoryfps_20261115.common import identity as old_identity
from experiments.group5_memoryfps_20261115.report import summary, METRICS


def audited_curve(data,execution,entries,model,row,level):
    entry=entries[level][row['key']];info=data['registry'][f'{model}-{row["suite"]}']
    cache=verify_cache(data,model,row,entry)
    if level==1:
        ref=cache['baseline_result'];path=Path(ref['path']);assert sha(path)==ref['sha256']
        old=read(SOURCE);ref6=old['streams']['6'];assert sha(ref6['path'])==ref6['sha256']
        old_entry=next(e for e in read(ref6['path'])['episodes'] if e['key']==row['key'])
        expected=old_identity(data['source_inputs_sha256'],info,row,old_entry)
    else:path=result_path(model,level,row['key']);expected=identity(execution['inputs_sha256'],info,row,entry)
    if not path.exists():return None
    r=read(path);assert r['status']=='completed'
    validate_record(r,row,entry,expected,True)
    assert digest([a['input_playback_slots'] for a in r['input_audits']])==digest(cache['input_playback_slots'])
    computed=scores(r['predictions'],entry);assert r['metrics']==computed
    if level>1:
        task=ROOT/'tasks'/f'{model}-{row["suite"]}'
        choice=read(task/'choice.json');assert choice['status']=='passed'
        assert r['batch_size']==choice['batch_size']
        assert r['initialization']==read(task/'batch.json')['initialization']
        for segment in r['execution_segments']:
            assert segment['execution']['tuning_sha256']==sha(task/'choice.json')
            assert segment['execution']['execution_sha256']==sha(ROOT/'execution.json')
    return r,computed,path,sha(path)


def task_report(task):
    data,execution=source();entries=streams(data);info=data['registry'][task]
    curves=[];missing=[];points=0
    for row in data['episodes']:
        if row['suite']!=info['suite']:continue
        for level in range(1,6):
            result=audited_curve(data,execution,entries,info['model'],row,level)
            if result is None:missing.append(dict(key=row['key'],level=level));continue
            r,metrics,path,h=result;points+=len(r['predictions'])
            curves.append(dict(key=row['key'],level=level,path=str(path),sha256=h,metrics=metrics))
    return dict(task=task,complete=not missing,curves=len(curves),observations=points,missing=missing,
        inputs_sha256=execution['inputs_sha256'],checkpoint_sha256=info['checkpoint_sha256'],
        trajectory_results=curves,checked=time.time())


def report():
    data,execution=source();verify_code();entries=streams(data);flat=[];comparisons=[];missing=[];count=0;points=0;paths={}
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=OUT/'predictions.csv.tmp'
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','suite','task_id','key','level','frame_index','timestamp','target_pt','prediction','common_6fps','raw_response'])
        writer.writeheader()
        for model in MODELS:
            for row in data['episodes']:
                for level in range(1,6):
                    result=audited_curve(data,execution,entries,model,row,level)
                    if result is None:missing.append(dict(model=model,key=row['key'],level=level));continue
                    r,metrics,path,h=result;paths[str(path)]=h;entry=entries[level][row['key']]
                    for scope,v in metrics.items():
                        flat.append(dict(model=model,suite=row['suite'],task_id=row['task_id'],key=row['key'],level=level,
                            scoring=scope,**{k:v[k] for k in (*METRICS,'samples','invalid_predictions','correlation_undefined_reason','vs_undefined_reason')}))
                    common=set(entry['common_scoring_indices'])
                    for i,(frame,t,pt,pred,raw) in enumerate(zip(r['source_frame_indices'],r['timestamps'],r['target_pt'],r['predictions'],r['raw_responses'])):
                        writer.writerow(dict(model=model,suite=row['suite'],task_id=row['task_id'],key=row['key'],level=level,
                            frame_index=frame,timestamp=t,target_pt=pt,prediction=pred,common_6fps=i in common,
                            raw_response=__import__('json').dumps(raw,ensure_ascii=False)))
                        points+=1
                    count+=1
    for model in MODELS:
        for suite in SUITES:
            expected=sum(r['suite']==suite for r in data['episodes'])
            for level in range(1,6):
                for scope in ('full_rate','common_6fps'):
                    rows=[r for r in flat if (r['model'],r['suite'],r['level'],r['scoring'])==(model,suite,level,scope)]
                    comparisons.append(dict(model=model,suite=suite,level=level,scoring=scope,expected=expected,
                        complete=len(rows)==expected,**(summary(rows) if len(rows)==expected else dict(episodes=len(rows)))))
    complete=not missing
    if complete:assert count==900 and points==351990
    tmp.replace(OUT/('predictions.csv' if complete else 'partial-predictions.csv'))
    for name,rows in (('episode_metrics.csv',flat),('comparison.csv',comparisons)):
        if not rows:continue
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    result=dict(status='completed' if complete else 'incomplete',curves=count,predictions=points,
        reused_baseline_curves=180,expected_curves=900,expected_predictions=351990,missing=missing,
        inputs_sha256=execution['inputs_sha256'],protocol=data['protocol'],execution=execution,
        metric_provenance=read(SOURCE)['metric_provenance'],comparison=comparisons,prediction_file_sha256=paths,checked=time.time())
    save(OUT/'metrics.json',result);return result


if __name__=='__main__':report()
