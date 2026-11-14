"""User-authorized R2VLM evaluation: ten frozen validation episodes per suite.

This separate queue does not re-enable cancelled full-dataset jobs, touch State
training, replace original predictions, or create a desktop automation.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from .prepare_cache import progress_values, save, sha
from .eval_common import EVAL, PLAN, OUT, ensure_images, trajectory_metrics, result_path
from .group1_report import summarize, verify_record, METRICS
from .overlap_guard import resources, pause_reasons, alive

HERE = Path('/cache/cvpzhr/setup/r2vlm-ten-per-suite-20261114')
RESULTS = Path('/cache/cvpzhr/evaluations/r2vlm-ten-per-suite-20261114')
CODE = Path(__file__).resolve().parents[2]
MODEL = 'r2vlm-alfred-rl'
SUITES = ('State', 'Order', 'Count')
EXCLUDED = 'Order/O03/swap-blocks_episode_001342'
PYTHON = '/cache/cvpzhr/envs/forest-labels/bin/python'
MODEL_PYTHON = '/cache/robot-baselines/envs/procvlm/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Frozen input differs: {path}'
    else:
        save(path, value)


def draw(population):
    assert len(population) == len(set(population)) and len(population) >= 10
    return sorted(random.Random(42).sample(sorted(population), 10))


def selection():
    baseline = read(EVAL/'group1-manifest.json')
    populations = {suite: sorted(r['key'] for r in baseline['episodes']
                            if r['suite'] == suite and r['key'] != EXCLUDED) for suite in SUITES}
    assert {suite: len(p) for suite,p in populations.items()} == dict(State=89, Order=118, Count=177)
    value = dict(schema='r2-ten-per-suite-selection-v1', user_authorized=True,
        policy='uniform sampling without replacement from sorted keys; independent Random(42) per suite; before checking predictions',
        seed=42, count_per_suite=10, input_fps=5, excluded_order_key=EXCLUDED,
        populations=populations, selected={s: draw(p) for s,p in populations.items()},
        original_manifest_sha256=sha(EVAL/'group1-manifest.json'), recover_excluded_by_user=True,
        plan_sha256=sha(PLAN), full_dataset_cancellation_remains=True,
        original_predictions_preserved=True, permitted_gpus=[2,3])
    freeze(HERE/'selection.json', value)
    return value


def prepare_inputs():
    chosen = read(HERE/'selection.json')
    baseline = read(EVAL/'group1-manifest.json')
    rows = [dict(r) for r in baseline['episodes'] if r['key'] in sum([chosen['selected'][s] for s in SUITES[:3]], [])]
    rows.sort(key=lambda r:(SUITES.index(r['suite']),r['key']))
    assert Counter(r['suite'] for r in rows) == {s:10 for s in SUITES}
    for row in rows:
        truth = progress_values(row)
        assert row['target_pt'] == [truth[i] for i in row['frame_indices']]
        assert sha(row['videos'][0]) == row['video_hashes'][0]
    value = dict(schema='r2-ten-per-suite-inputs-v1', selection_sha256=sha(HERE/'selection.json'),
                 plan_sha256=sha(PLAN), rows=rows, expected_episodes=30,
                 expected_samples=sum(len(r['target_pt']) for r in rows))
    freeze(HERE/'inputs.json', value)
    return value


def validate_saved(record, row, selection_sha):
    assert record['selection_sha256'] == selection_sha and record['model'] == MODEL
    for key in ['key','suite','task_id','target_pt','frame_indices','timestamps','progress_sha256']:
        assert record[key] == row[key], key
    assert record['source_video_sha256'] == row['video_hashes']
    assert len(record['predictions']) == len(record['raw_responses']) == len(row['target_pt'])
    assert record['metrics'] == trajectory_metrics(record['predictions'],row['target_pt'],row['timestamps'])


def worker(shard):
    from .experiment_backends import load_backend
    from .experiment_inputs import regular_entry
    from .official_baseline import OLD
    inputs = read(HERE/'inputs.json'); selected_sha = sha(HERE/'selection.json')
    assert inputs['selection_sha256'] == selected_sha and inputs['plan_sha256'] == sha(PLAN)
    rows = inputs['rows'][shard::2]
    lock = (HERE/f'worker-{shard}.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(pid=os.getpid(), shard=shard, gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
                  expected_episodes=len(rows), completed_episodes=0, reused_episodes=0, new_samples=0)
    def update(**kw):
        status.update(kw,updated=time.time());save(HERE/f'worker-{shard}.json',status)
    initial_oom = resources()['oom_kill']
    last_check = 0.
    resource_pause = False
    def pause():
        nonlocal last_check, resource_pause
        if (HERE/f'pause-worker-{shard}').exists():
            return True
        if not alive(int((HERE/'controller.pid').read_text())):
            return True
        if time.monotonic()-last_check >= 10:
            resource_pause = bool(pause_reasons(resources(), status['gpu'], initial_oom))
            last_check = time.monotonic()
        return resource_pause
    try:
        update(status='loading_model')
        backend = load_backend(MODEL, None, True, .30)
        audit_path = HERE/f'initialization-{shard}-{os.getpid()}.json'
        save(audit_path, dict(backend.audit, selection_sha256=selected_sha, input_sha256=sha(HERE/'inputs.json')))
        for row in rows:
            dest = RESULTS/'episodes'/(row['key'].replace('/','__')+'.json')
            partial = HERE/'partial'/(row['key'].replace('/','__')+'.json')
            if dest.exists():
                validate_saved(read(dest),row,selected_sha)
                update(completed_episodes=status['completed_episodes']+1)
                continue
            if pause():
                update(status='paused_by_resource_guard'); return 75
            previous = result_path(MODEL,row['key'])
            reused = previous.exists()
            provenance = dict(initialization_audit=str(audit_path), initialization_sha256=sha(audit_path))
            if reused:
                r = read(previous)
                verify_record(r,row,MODEL,sha(EVAL/'group1-manifest.json'),sha(PLAN),False)
                audit = read(r['initialization_audit'])
                for field in ['official_revision','official_file_sha256','native_worker_sha256','prompt_sha256']:
                    assert audit[field] == backend.audit[field], field
                assert audit['config'] == backend.audit['native_config'] and audit['adapter_loaded'] is False
                pred, raw = r['predictions'], r['raw_responses']
                provenance.update(reused_prediction_path=str(previous), reused_prediction_sha256=sha(previous),
                                  original_initialization_path=r['initialization_audit'],
                                  original_initialization_sha256=sha(r['initialization_audit']))
            else:
                update(status='preparing_images', episode=row['key'])
                assert sha(row['progress_path']) == row['progress_sha256']
                root = ensure_images(row, main_only=True)
                entry = regular_entry(row,'baseline')
                prior = read(partial) if partial.exists() else dict(selection_sha256=selected_sha,
                    input_sha256=sha(HERE/'inputs.json'), key=row['key'], predictions=[], raw_responses=[])
                assert prior['selection_sha256'] == selected_sha and prior['key'] == row['key']
                assert prior['input_sha256'] == sha(HERE/'inputs.json')
                pred,raw = prior['predictions'],prior['raw_responses']
                assert len(pred) == len(raw) <= len(row['target_pt'])
                backend.begin(row,entry,root,prior,pause)
                while len(pred) < len(row['target_pt']):
                    if pause():
                        update(status='paused_by_resource_guard');return 75
                    value,answer,_ = backend.predict(len(pred),pred,raw)
                    assert value is None or math.isfinite(value)
                    pred.append(value);raw.append(answer);save(partial,prior)
                    update(status='running', episode=row['key'], samples_in_episode=len(pred),
                           episode_samples=len(row['target_pt']),new_samples=status['new_samples']+1)
            record = dict(schema='r2-ten-per-suite-prediction-v1', model=MODEL,
                selection_sha256=selected_sha,input_sha256=sha(HERE/'inputs.json'),
                predictions=pred, raw_responses=raw, source_video_sha256=row['video_hashes'],
                official_revision=backend.audit['official_revision'],checkpoint=backend.audit['checkpoint'],
                uses_future_endpoint_reference=False, metrics=trajectory_metrics(pred,row['target_pt'],row['timestamps']),
                **{k:row[k] for k in ['key','suite','task_id','target_pt','frame_indices','timestamps','progress_sha256']},
                **provenance)
            validate_saved(record,row,selected_sha)
            assert not dest.exists(); save(dest,record); partial.unlink(missing_ok=True)
            update(completed_episodes=status['completed_episodes']+1,
                   reused_episodes=status['reused_episodes']+int(reused))
        update(status='completed'); return 0
    except BaseException:
        update(status='failed', error=traceback.format_exc());raise


def report():
    inputs = read(HERE/'inputs.json'); rows=[];missing=[]
    for row in inputs['rows']:
        p=RESULTS/'episodes'/(row['key'].replace('/','__')+'.json')
        if not p.exists():missing.append(row['key']);continue
        r=read(p);validate_saved(r,row,sha(HERE/'selection.json'))
        assert sha(row['progress_path']) == row['progress_sha256']
        assert sha(r['initialization_audit']) == r['initialization_sha256']
        if 'reused_prediction_path' in r:
            assert sha(r['reused_prediction_path']) == r['reused_prediction_sha256']
            assert sha(r['original_initialization_path']) == r['original_initialization_sha256']
        rows.append(dict(key=row['key'],suite=row['suite'],**r['metrics']))
    suites={s:summarize([r for r in rows if r['suite']==s],10,
        sum(len(r['target_pt']) for r in inputs['rows'] if r['suite']==s)) for s in SUITES}
    prior=read(OUT/'group1/comparison-118-order.json')
    models={m:v for m,v in prior['models'].items() if m not in ['forest',MODEL]}
    assert all(v['overall']['coverage_complete'] for v in models.values())
    result=dict(status='completed' if not missing else 'incomplete', selection=read(HERE/'selection.json'),
        selection_sha256=sha(HERE/'selection.json'),input_sha256=sha(HERE/'inputs.json'),
        by_suite=suites,by_episode=rows,missing=missing,original_comparison_sha256=sha(OUT/'group1/comparison-118-order.json'),
        original_models=models, original_results_unchanged=True,
        caution='R2VLM uses 10 random episodes per suite; existing other models use full original cohorts.')
    save(RESULTS/'report.json',result)
    lines=['# Group 1 comparison with R2VLM random-10 evaluation','',
        'Cells are Pearson / Spearman / VS, averaged equally over defined trajectory metrics. R2VLM uses 10 sampled trajectories per suite; other methods retain full-cohort results.', '',
        '| Model | State | Order | Count |','|---|---|---|---|']
    def cell(x):
        return ' / '.join('N/A' if x['defined_trajectory_means'][m] is None else f'{x["defined_trajectory_means"][m]:.4f}' for m in METRICS[:3])
    for m,v in models.items():lines.append('| '+m+' | '+' | '.join(cell(v['by_suite'][s]) for s in SUITES[:3])+' |')
    lines.append('| R2VLM (10 per suite) | '+' | '.join(cell(suites[s]) for s in SUITES[:3])+' |')
    lines += ['', '| R2VLM suite | Coverage | Pearson / Spearman / VS | P/S defined | Undefined | Invalid frames |',
        '|---|---:|---|---:|---:|---:|']
    for s,x in suites.items():lines.append(f'| {s} | {x["completed_episodes"]}/10 | {cell(x)} | {x["defined_episode_counts"]["pearson"]} | {x["undefined_episode_counts"]["pearson"]} | {x["invalid_frames"]} |')
    lines += ['', 'RoboBrain2.5 and Robo-Dopamine use the true future endpoint reference. Other methods do not.',
              'Recover is excluded by user request. Only State, Order and Count are evaluated.']
    (RESULTS/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def control():
    from .eval_dispatch import gpu_processes
    HERE.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in HERE.glob('*launch-*.json'):
        assert not alive(read(p)['pid']), 'Prior subset child still alive; inspect before restarting'
    selection();(HERE/'controller.pid').write_text(str(os.getpid()))
    initial_oom=resources()['oom_kill'];stage='prepare'
    active={};done=set();safe_since={};last_launch=0
    def launch(name,gpu,args,python):
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
            PYTHONUNBUFFERED='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTORCH_ALLOC_CONF='expandable_segments:True')
        environment.pop('PYTHONPATH',None)
        (HERE/f'pause-{name}').unlink(missing_ok=True)
        with (HERE/f'{name}.log').open('a') as log:
            child=subprocess.Popen([python,'-m','experiments.prm_five_groups_20261114.r2_subset',*args],
                cwd=CODE,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(HERE/f'{name}-launch-{child.pid}.json',dict(pid=child.pid,gpu=gpu,args=args,started=time.time(),resource_admission_passed=True))
        active[name]=(child,gpu)
    while True:
        sample=resources();now=time.time()
        for name,(child,gpu) in list(active.items()):
            if pause_reasons(sample,gpu,initial_oom):(HERE/f'pause-{name}').touch()
            code=child.poll()
            if code is None:continue
            del active[name];safe_since.pop(gpu,None)
            if code==0:done.add(name)
            elif code!=75:
                save(HERE/'status.json',dict(status='failed',failed_child=name,exit_code=code,pid=os.getpid(),updated=now))
                return 1
        if sample['oom_kill']>initial_oom:
            save(HERE/'status.json',dict(status='blocked_new_oom',pid=os.getpid(),updated=now));return 1
        if stage=='prepare':prepare_inputs();stage='inference'
        if stage=='inference' and all(f'worker-{i}' in done for i in range(2)):
            result=report();assert result['status']=='completed'
            save(HERE/'status.json',dict(status='completed',pid=os.getpid(),updated=now,report=str(RESULTS/'report.json')));return 0
        processes=gpu_processes()
        jobs=[
            (f'worker-{i}',2+i,['worker','--shard',str(i)],MODEL_PYTHON) for i in range(2)]
        for name,gpu,args,python in jobs:
            if name in done or name in active:continue
            if processes[gpu] or pause_reasons(sample,gpu,initial_oom):safe_since.pop(gpu,None);continue
            safe_since.setdefault(gpu,now)
            if now-safe_since[gpu]>=60 and now-last_launch>=60:
                launch(name,gpu,args,python);last_launch=now;break
        save(HERE/'status.json',dict(status=stage,pid=os.getpid(),updated=now,done=sorted(done),
            active={n:dict(pid=p.pid,gpu=g) for n,(p,g) in active.items()},safe_since=safe_since,
            resources=sample,selection_sha256=sha(HERE/'selection.json')))
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['select','prepare','worker','report','control'])
    parser.add_argument('--shard',type=int,choices=[0,1]);a=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    if a.command=='select':print(json.dumps(selection(),ensure_ascii=False))
    elif a.command=='prepare':print(json.dumps(prepare_inputs(),ensure_ascii=False))
    elif a.command=='worker':sys.exit(worker(a.shard))
    elif a.command=='report':print(json.dumps(report(),ensure_ascii=False))
    else:sys.exit(control())
