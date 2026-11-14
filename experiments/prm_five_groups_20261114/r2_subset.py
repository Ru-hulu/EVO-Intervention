"""User-authorized R2VLM evaluation: ten frozen validation episodes per suite.

This separate queue does not re-enable cancelled full-dataset jobs, touch State
training, replace original predictions, or create a desktop automation.
"""
import argparse
from collections import Counter
import csv
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

from .prepare_cache import BASE, DATA, RECOVER_PROMPTS, progress_values, save, sha
from .eval_common import EVAL, PLAN, OUT, ensure_images, sampling_indices, trajectory_metrics, result_path
from .group1_report import summarize, verify_record, METRICS
from .overlap_guard import resources, pause_reasons, alive

HERE = Path('/cache/cvpzhr/setup/r2vlm-ten-per-suite-20261114')
RESULTS = Path('/cache/cvpzhr/evaluations/r2vlm-ten-per-suite-20261114')
CODE = Path(__file__).resolve().parents[2]
MODEL = 'r2vlm-alfred-rl'
SUITES = ('State', 'Order', 'Count', 'Recover')
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
    recover_path = DATA/'Recover/manifest.json'
    recover = read(recover_path)
    assert sha(recover_path) == read(PLAN)['datasets']['suites']['Recover']['manifest_sha256']
    populations = {s: sorted(r['key'] for r in baseline['episodes']
                            if r['suite'] == s and r['key'] != EXCLUDED) for s in SUITES[:3]}
    populations['Recover'] = sorted(f"Recover/{t['task_id']}/{k}" for t in recover['tasks'] for k in t['val'])
    assert {s: len(p) for s,p in populations.items()} == dict(State=89, Order=118, Count=177, Recover=34)
    value = dict(schema='r2-ten-per-suite-selection-v1', user_authorized=True,
        policy='uniform sampling without replacement from sorted keys; independent Random(42) per suite; before checking predictions',
        seed=42, count_per_suite=10, input_fps=5, excluded_order_key=EXCLUDED,
        populations=populations, selected={s: draw(p) for s,p in populations.items()},
        original_manifest_sha256=sha(EVAL/'group1-manifest.json'), recover_manifest_sha256=sha(recover_path),
        plan_sha256=sha(PLAN), full_dataset_cancellation_remains=True,
        original_predictions_preserved=True, permitted_gpus=[2,3])
    freeze(HERE/'selection.json', value)
    return value


def missing_pour_records():
    from .recover_pour_labels import read_records, OUTPUT, verify_episode
    chosen = {k.split('/')[-1] for k in read(HERE/'selection.json')['selected']['Recover']}
    rows = [r for r in read_records() if r['sample_id'] in chosen]
    missing = []
    for row in rows:
        if (OUTPUT/f"{row['episode']}.progress.json").exists():
            verify_episode(row)
        else:
            missing.append(row)
    return missing


def labels():
    import numpy as np
    import torch
    from mem_lite.datasets.prepare_progress import extract_deltas
    from . import recover_pour_labels as native
    lock = (native.HERE/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = missing_pour_records()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.15)
    model, processor, digest = native.load_encoder('cuda')
    audit = dict(revision=native.REVISION, model=str(native.MODEL), vision_sha256=digest,
        strict_load=True, model_class=type(model).__name__, dtype=str(next(model.parameters()).dtype),
        batch_size=16, min_pixels=4096, max_pixels=262144, all_native_frames=True,
        cuda_memory_fraction=.15, plan_sha256=sha(PLAN), code_sha256=sha(__file__),
        native_algorithm_sha256=sha(native.__file__), selection_sha256=sha(HERE/'selection.json'))
    save(HERE/'label-initialization.json', audit)
    native.OUTPUT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if (HERE/'pause-labels').exists():
            return 75
        save(HERE/'label-status.json', dict(status='running', pid=os.getpid(), episode=row['episode'], updated=time.time()))
        assert sha(row['video']) == row['video_sha256'] and sha(row['annotation']) == row['annotation_sha256']
        rawpath = native.OUTPUT/f"{row['episode']}.deltas.npz"
        assert not rawpath.exists(), 'Inspect existing raw delta provenance before recovery'
        raw, info = extract_deltas(row['video'], model, processor, 16, 262144)
        assert info['frame_count'] == row['frames'] and info['fps'] == 30
        effective, pt, events = native.progress_from_deltas(raw, read(row['annotation']))
        tmp = rawpath.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez(f, delta=raw, timestamps=info['timestamps_seconds'])
        tmp.replace(rawpath)
        save(rawpath.with_suffix('.json'), dict(sha256=sha(rawpath), source=row, audit=audit))
        meta = native.OUTPUT/f"{row['episode']}.progress.json"
        csvpath = meta.with_suffix('.csv')
        tmp = csvpath.with_suffix('.csv.tmp')
        with tmp.open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['frame_index','delta','effective_delta','pt'])
            w.writerows(zip(range(row['frames']), raw, effective, pt, strict=True))
        tmp.replace(csvpath)
        save(meta, dict(audit, model_revision=native.REVISION, video_sha256=row['video_sha256'],
            annotation_sha256=row['annotation_sha256'], csv_sha256=sha(csvpath), events=events,
            raw_delta_sha256=sha(rawpath), total_frames=row['frames'], original_recover_files_modified=False,
            rule='first effective pour per cup and associated actual put-down; repeats/idle flat; final normalization'))
        native.verify_episode(row)
    selected_ids = {k.split('/')[-1] for k in read(HERE/'selection.json')['selected']['Recover']}
    verified = [native.verify_episode(r) for r in native.read_records() if r['sample_id'] in selected_ids]
    # Verify the complete inventory of original Recover files without changing it.
    inventory = read(native.HERE/'source-files-before.json')
    assert all(sha(DATA/'Recover'/p) == h for p,h in inventory.items())
    save(HERE/'selected-pour-verification.json', dict(status='passed', episodes=verified,
        original_recover_files_unchanged=True, all_35_pour_episodes_completed=False))
    save(HERE/'label-status.json', dict(status='completed', pid=os.getpid(), updated=time.time()))
    return 0


def prepare_inputs():
    import av
    chosen = read(HERE/'selection.json')
    baseline = read(EVAL/'group1-manifest.json')
    rows = [dict(r) for r in baseline['episodes'] if r['key'] in sum([chosen['selected'][s] for s in SUITES[:3]], [])]
    root = DATA/'Recover'; manifest = read(root/'manifest.json')
    for task in manifest['tasks']:
        meta = root/task['trajectory_index']; assert sha(meta) == task['trajectory_index_sha256']
        records = {r['sample_id']:r for r in map(json.loads, meta.read_text().splitlines())}
        for sample in task['val']:
            key = f"Recover/{task['task_id']}/{sample}"
            if key not in chosen['selected']['Recover']:
                continue
            r = records[sample]; stem = f"episode_{r['episode_index']:06d}"
            progress = meta.parent/(r.get('progress_path') or f'annotation/{stem}.progress.json')
            if task['task_id'] == 'R05' and not progress.exists():
                progress = BASE/'labels/Recover/R05'/f'{stem}.progress.csv'
            assert progress.exists(), f'Missing selected PT: {key}'
            views = r['video_views']; order = ['main_view','wrist_view',None] if 'wrist_view' in views else ['main_view','left_view','right_view']
            videos = [str(meta.parent/views[v]['video_path']) if v in views else None for v in order]
            hashes = [views[v]['video_sha256'] if v in views else None for v in order]
            assert sha(videos[0]) == hashes[0]
            with av.open(videos[0]) as video:
                rate = video.streams.video[0].average_rate
            indices = sampling_indices(r['total_frames'], rate)
            row = dict(key=key, suite='Recover', task_id=task['task_id'], sample_id=sample, split='val',
                frames=r['total_frames'], fps=r['fps'], source_rate=str(rate), input_fps=5, videos=videos,
                video_hashes=hashes, progress_path=str(progress), progress_sha256=sha(progress),
                task_prompt=RECOVER_PROMPTS[task['task_id']], frame_indices=indices,
                timestamps=[i/float(rate) for i in indices], input_issues=[], metadata_sha256=sha(meta))
            truth = progress_values(row); row['target_pt'] = [truth[i] for i in indices]
            rows.append(row)
    rows.sort(key=lambda r:(SUITES.index(r['suite']),r['key']))
    assert Counter(r['suite'] for r in rows) == {s:10 for s in SUITES}
    for row in rows:
        truth = progress_values(row)
        assert row['target_pt'] == [truth[i] for i in row['frame_indices']]
        assert sha(row['videos'][0]) == row['video_hashes'][0]
    value = dict(schema='r2-ten-per-suite-inputs-v1', selection_sha256=sha(HERE/'selection.json'),
                 plan_sha256=sha(PLAN), rows=rows, expected_episodes=40,
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
            reused = previous.exists() and row['suite'] != 'Recover'
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
              'The separate Recover result is not mixed into the original SOC overall metric.']
    (RESULTS/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def control():
    from .eval_dispatch import gpu_processes
    HERE.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in HERE.glob('*launch-*.json'):
        assert not alive(read(p)['pid']), 'Prior subset child still alive; inspect before restarting'
    selection();(HERE/'controller.pid').write_text(str(os.getpid()))
    initial_oom=resources()['oom_kill'];stage='labels' if missing_pour_records() else 'prepare'
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
        if stage=='labels' and ('labels' in done or not missing_pour_records()):stage='prepare'
        if stage=='prepare':prepare_inputs();stage='inference'
        if stage=='inference' and all(f'worker-{i}' in done for i in range(2)):
            result=report();assert result['status']=='completed'
            save(HERE/'status.json',dict(status='completed',pid=os.getpid(),updated=now,report=str(RESULTS/'report.json')));return 0
        processes=gpu_processes()
        jobs=[('labels',3,['labels'],PYTHON)] if stage=='labels' else [
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
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['select','labels','prepare','worker','report','control'])
    parser.add_argument('--shard',type=int,choices=[0,1]);a=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    if a.command=='select':print(json.dumps(selection(),ensure_ascii=False))
    elif a.command=='labels':sys.exit(labels())
    elif a.command=='prepare':print(json.dumps(prepare_inputs(),ensure_ascii=False))
    elif a.command=='worker':sys.exit(worker(a.shard))
    elif a.command=='report':print(json.dumps(report(),ensure_ascii=False))
    else:sys.exit(control())
