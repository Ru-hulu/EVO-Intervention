"""Read and audit formal results; no inference, no writes to predictions."""
import argparse
import csv
import io
import math
import statistics
import time
from collections import Counter
from .common import *
from .protocol import validate_record
from .metrics import score

METRICS = ('pearson','spearman','circle_consistency','static_mae')


def summarize(records, expected, applicable):
    complete = len(records) == expected
    output = dict(expected_trajectories=expected, present_trajectories=len(records), complete=complete,
        invalid_frames=sum(r['invalid_frames'] for r in records),
        trajectories_with_invalid_frames=sum(r['invalid_frames']>0 for r in records),
        correlation_undefined_reasons=dict(Counter(r['correlation_undefined_reason'] for r in records
            if r['correlation_undefined_reason'])))
    for key in METRICS:
        values=[r[key] for r in records if r[key] is not None]
        output[key]=dict(defined=len(values), undefined=len(records)-len(values),
            applicable=key in applicable, defined_trajectory_mean=(math.fsum(values)/len(values)
                if complete and values and key in applicable else None))
    return output


def seed_summary(rows, applicable):
    complete = len(rows)==3 and all(r['complete'] for r in rows)
    result=dict(seeds=list(SEEDS),all_seed_coverage_complete=complete)
    for key in METRICS:
        values=[r[key]['defined_trajectory_mean'] for r in rows if r[key]['defined_trajectory_mean'] is not None]
        result[key]=dict(applicable=key in applicable, defined_seeds=len(values),
            mean=statistics.mean(values) if complete and len(values)==3 else None,
            sample_std=statistics.stdev(values) if complete and len(values)==3 else None,
            per_seed_defined_counts=[r[key]['defined'] for r in rows],
            per_seed_undefined_counts=[r[key]['undefined'] for r in rows])
    return result


def csv_text(rows):
    if not rows:return ''
    fields=list(dict.fromkeys(k for row in rows for k in row))
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return output.getvalue()


def report():
    source=read(ROOT/'inputs.json');verify_contract(source);input_sha=sha(ROOT/'inputs.json')
    episode_metrics=[];summaries=[];missing=[];result_hashes={};audits={}
    for condition in CONDITIONS:
        manifest=read(source['conditions'][condition]['path'])
        entries={e['key']:e for e in manifest['episodes']}
        family,seed=condition.split('-seed');seed=int(seed)
        applicable={'pearson','spearman', 'circle_consistency' if family=='rewind_only' else 'static_mae'}
        for model in MODELS:
            for suite in SUITES:
                rows=[r for r in source['episodes'] if r['suite']==suite];scores=[];tasks={}
                info=source['registry'][f'{model}-{suite}']
                for row in rows:
                    path=result_path(model,condition,row['key']);entry=entries[row['key']]
                    if not path.exists():
                        missing.append(dict(model=model,suite=suite,condition=condition,key=row['key']));continue
                    result=read(path)
                    validate_record(result,row,entry,identity(input_sha,info,row,entry),complete=True)
                    assert result['status']=='completed'
                    audit=result['initialization']; name=f'{model}-{suite}'
                    if name in audits:assert audits[name]==audit
                    audits[name]=audit
                    canary=read(ROOT/'canary'/f'{name}.json')
                    assert canary['status']=='passed' and canary['gpu_forward_tested'] and canary['inputs_sha256']==input_sha
                    assert canary['initialization']==audit
                    metrics=score(result['predictions'],row,entry)
                    assert metrics==result['metrics'], 'Saved metrics do not match raw predictions'
                    scores.append(metrics);tasks.setdefault(row['task_id'],[]).append(metrics)
                    result_hashes[str(path)]=sha(path)
                    episode_metrics.append(dict(model=model,suite=suite,task_id=row['task_id'],key=row['key'],
                        condition=family,seed=seed,**metrics))
                summary=dict(model=model,suite=suite,condition=family,seed=seed,**summarize(scores,len(rows),applicable))
                summary['tasks']={task:summarize(tasks.get(task,[]),sum(r['task_id']==task for r in rows),applicable)
                                  for task in sorted({r['task_id'] for r in rows})}
                summaries.append(summary)
    across=[]
    for model in MODELS:
        for suite in SUITES:
            for family in ('static_only','rewind_only'):
                group=[s for s in summaries if (s['model'],s['suite'],s['condition'])==(model,suite,family)]
                applicable={'pearson','spearman','circle_consistency' if family=='rewind_only' else 'static_mae'}
                across.append(dict(model=model,suite=suite,condition=family,**seed_summary(group,applicable)))
    expected=source['expected_curves']
    assert len(episode_metrics)+len(missing)==expected
    observations=sum(m['observations'] for m in episode_metrics)
    if not missing:assert observations==source['expected_observations']
    output=dict(status='completed' if not missing else 'incomplete',generated=time.time(),inputs_sha256=input_sha,
        expected_curves=expected,present_curves=len(episode_metrics),missing=missing,observations=observations,
        expected_observations=source['expected_observations'],protocol=source['protocol'],
        initialization=audits,per_seed=summaries,across_seeds=across,episodes=episode_metrics,
        prediction_file_sha256=result_hashes,full_coverage=not missing)
    OUT.mkdir(parents=True,exist_ok=True)
    flat=[{k:v for k,v in e.items() if k!='events'} for e in episode_metrics]
    # Event-level details include every loop triplet, its original timestamp and
    # three raw predictions; CSV is a compact trajectory-level companion.
    (OUT/'trajectory-metrics.csv').write_text(csv_text(flat))
    comparison=[]
    for s in summaries:
        row={k:s[k] for k in ('model','suite','condition','seed','expected_trajectories','present_trajectories','complete','invalid_frames')}
        for key in METRICS:
            for field in ('defined_trajectory_mean','defined','undefined'):
                row[f'{key}_{field}']=s[key][field]
        comparison.append(row)
    (OUT/'comparison.csv').write_text(csv_text(comparison))
    save(OUT/'report.json',output)
    return output


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();r=report()
    print(json.dumps({k:r[k] for k in ('status','expected_curves','present_curves','observations')}))
