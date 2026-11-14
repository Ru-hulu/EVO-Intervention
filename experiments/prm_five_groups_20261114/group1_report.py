"""CPU-only, coverage-gated first-group results with per-episode provenance checks."""
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
import time

from .eval_common import EVAL, OUT, PLAN, result_path, trajectory_metrics
from .prepare_cache import BASE, save, sha

OLD=Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')
METRICS=('pearson','spearman','VS','ER','C')


def verify_record(record, episode, model, manifest_sha, plan_sha, uses_future):
    assert record['model']==model and record['key']==episode['key']
    assert record['suite']==episode['suite'] and record['task_id']==episode['task_id']
    assert record['group']==1 and record['condition']=='baseline' and record['input_fps']==5
    assert record['manifest_sha256']==manifest_sha and record['plan_sha256']==plan_sha
    for field in ['frame_indices','target_pt','timestamps','progress_sha256']:
        assert record[field]==episode[field], field
    assert record['source_video_sha256']==episode['video_hashes']
    assert record['uses_future_endpoint_reference'] is uses_future
    assert len(record['predictions'])==len(record['raw_responses'])==len(episode['frame_indices'])
    return trajectory_metrics(record['predictions'],episode['target_pt'],episode['timestamps'])


def summarize(rows, expected, expected_samples):
    complete=len(rows)==expected
    result={'coverage_complete':complete,'completed_episodes':len(rows),'expected_episodes':expected,
            'samples':sum(r['samples'] for r in rows),'expected_samples':expected_samples,
            'invalid_frames':sum(r['invalid_predictions'] for r in rows),
            'invalid_episodes':sum(r['invalid_predictions']>0 for r in rows),
            'correlation_undefined_reasons':dict(Counter(r['correlation_undefined_reason'] for r in rows if r['correlation_undefined_reason']))}
    if complete:
        assert result['samples']==expected_samples
    result['defined_episode_counts']={m:sum(r[m] is not None for r in rows) for m in METRICS}
    result['undefined_episode_counts']={m:len(rows)-result['defined_episode_counts'][m] for m in METRICS}
    # Incomplete suites have coverage counts only, not a publishable mean over
    # whichever episodes happened to finish first. Undefined denominators stay explicit.
    result['defined_trajectory_means']={m:(math.fsum(r[m] for r in rows if r[m] is not None)/result['defined_episode_counts'][m]
        if complete and result['defined_episode_counts'][m] else None) for m in METRICS}
    return result


def build():
    manifest=json.loads((EVAL/'group1-manifest.json').read_text())
    digest=sha(EVAL/'group1-manifest.json');plan=json.loads(PLAN.read_text());plan_sha=sha(PLAN)
    assert manifest['plan_sha256']==plan_sha and manifest['expected_episodes']==385 and manifest['expected_samples']==64827
    catalog=json.loads((OLD/'model-download-catalog.json').read_text())
    source_records={};result={};ready=[]

    @lru_cache(None)
    def audit(path, model, suite):
        p=Path(path);assert p.parent==EVAL/'jobs'
        a=json.loads(p.read_text());spec=next(m for m in plan['models'] if m['id']==model)
        if model=='forest':
            proof_path=BASE/suite/'checkpoint-verification.json'
            proof=json.loads(proof_path.read_text())
            assert proof['status']=='passed' and a['checkpoint_sha256']==proof['sha256']
            assert a['checkpoint_verification_sha256']==sha(proof_path)
            assert a['suite']==suite and a['final_epoch']==1 and a['trial_checkpoint_used'] is False
            assert a['strict_trainable_state_verified'] is True and a['memory_enabled'] is True
        else:
            ref=spec['prior_provenance_reference'];cat=catalog[model]
            assert a['official_revision']==cat['revision']==ref['revision']
            assert cat['local_dir']==ref['path'] and a['adapter_loaded'] is False
            expected=dict(ref['small_file_sha256'])
            expected.update({f['path']:f['lfs']['sha256'] for f in cat['files'] if f['path'].endswith(('.safetensors','.bin'))})
            assert all(a['official_file_sha256'].get(k)==v for k,v in expected.items())
            assert a['manifest_sha256']==digest and a['plan_sha256']==plan_sha
        return a,sha(p)

    for spec in plan['models']:
        model=spec['id'];rows=[];missing=[];audit_paths={}
        for episode in manifest['episodes']:
            path=result_path(model,episode['key'])
            if not path.exists():
                missing.append(episode['key']);continue
            r=json.loads(path.read_text())
            values=verify_record(r,episode,model,digest,plan_sha,spec['uses_test_trajectory_future_endpoint'])
            a,ah=audit(r['initialization_audit'],model,episode['suite'])
            if model=='forest':assert r['checkpoint_sha256']==a['checkpoint_sha256'] and r['checkpoint']==a['checkpoint']
            else:assert r['official_revision']==a['official_revision'] and r['checkpoint']==spec['prior_provenance_reference']['path']
            audit_paths[r['initialization_audit']]=ah
            source_records[str(path)]=sha(path)
            rows.append(dict(key=episode['key'],suite=episode['suite'],task_id=episode['task_id'],**values))
        suites={}
        for suite,n in [('State',89),('Order',119),('Count',177)]:
            subset=[r for r in rows if r['suite']==suite]
            suites[suite]=summarize(subset,n,manifest['by_suite'][suite]['samples'])
            if suites[suite]['coverage_complete']:ready.append({'model':model,'suite':suite})
        result[model]={'regime':spec['regime'],'uses_future_endpoint_reference':spec['uses_test_trajectory_future_endpoint'],
                       'by_suite':suites,'by_episode':rows,'missing_episodes':missing,'initialization_audits':audit_paths,
                       'overall':summarize(rows,385,64827)}
    report={'status':'verified_snapshot','updated':time.time(),'manifest_sha256':digest,'plan_sha256':plan_sha,
            'all_seven_models_complete':all(m['overall']['coverage_complete'] for m in result.values()),
            'aggregation':'per-trajectory equal mean over defined trajectories; explicit denominators; incomplete suite means withheld',
            'completed_model_suites':ready,'models':result,'source_result_sha256':source_records,
            'reporter_sha256':sha(__file__)}
    save(OUT/'group1/progress-report.json',report)
    return report


if __name__=='__main__':
    r=build()
    print(json.dumps({'all_seven_models_complete':r['all_seven_models_complete'],
        'completed_model_suites':r['completed_model_suites'],
        'coverage':{m:{s:v['completed_episodes'] for s,v in d['by_suite'].items()} for m,d in r['models'].items()}}))
