"""Prepare 20 shared R03 training trajectories for five models without training.

The existing Recovery processors, camera/history layouts, PT reader, FoREST
encoder/memory, and progress-only ProcVLM supervision are reused unchanged.
All paths below belong to a new run; original data and earlier caches are read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

SETUP = Path('/cache/cvpzhr/setup/recover-spray20-five-caches-20261116')
OUTPUT = Path('/cache/cvpzhr/datasets/recover-spray20-five-caches-5fps-20261116')
os.environ['RECOVER_CACHE_SETUP'] = str(SETUP)
os.environ['RECOVER_CACHE_OUTPUT'] = str(OUTPUT)
from experiments.recover_six_caches_20261114.common import (
    ROOT, OUT, VIEWS, read, rows, write_rows, sha, save, epdir, slots, source_checks)
from experiments.prm_five_groups_20261114.prepare_cache import progress_values

MODELS = ['robometer-4b', 'procvlm-2b', 'robobrain25-4b', 'dopamine-3b', 'forest']
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
CPU_PYTHON = '/cache/cvpzhr/envs/forest-labels/bin/python'
PROMPT = 'Remove all three spray bottles (red, green, and blue) from the rack, then reinsert them into the rack one by one.'


def choose(task, count=20, seed=42):
    train, val = task['train'], task['val']
    if task['task_id'] != 'R03' or len(train) != len(set(train)) or set(train) & set(val):
        raise ValueError('Invalid task or overlapping/duplicate splits')
    if len(train) < count:
        raise ValueError('Insufficient training trajectories; validation must not be used')
    return sorted(random.Random(seed).sample(sorted(train), count))


def freeze():
    assert not (ROOT/'selection.json').exists(), 'Existing frozen selection must not be replaced'
    data = Path('/cache/cvpzhr/datasets/ProPriMem/Recover').resolve()
    manifest_path = data/'manifest.json'; manifest = read(manifest_path)
    task = next(t for t in manifest['tasks'] if t['task_id'] == 'R03')
    chosen = choose(task); meta = data/task['trajectory_index']
    assert sha(meta) == task['trajectory_index_sha256']
    indexed = {r['sample_id']:r for r in rows(meta)}
    sources = {str(manifest_path):sha(manifest_path), str(meta):sha(meta)}
    records = []
    for key in chosen:
        r = indexed[key]; episode = f"episode_{r['episode_index']:06d}"
        assert r['fps'] == 30
        views = [r['video_views'][v] for v in ['main_view','left_view','right_view']]
        videos = [str(meta.parent/v['video_path']) for v in views]
        for p,v in zip(videos,views):
            sources[p] = sha(p); assert sources[p] == v['video_sha256'], p
        annotation = meta.parent/(r.get('annotation_path') or f'annotation/{episode}.json')
        progress = meta.parent/(r.get('progress_path') or f'annotation/{episode}.progress.json')
        sources[str(annotation)] = sha(annotation); sources[str(progress)] = sha(progress)
        label = read(progress)
        assert label['stage_annotation_sha256'] == sources[str(annotation)]
        # Retain the published goal and remove only its separate scripted-failure sentence.
        assert r['task_instructions'][0].startswith(PROMPT)
        row = dict(key=f'Recover/R03/{key}',suite='Recover',task_id='R03',
            task_name=task['task_name'],sample_id=key,split='train',frames=r['total_frames'],fps=r['fps'],
            videos=videos,video_hashes=[sources[p] for p in videos],annotation_path=str(annotation),
            annotation_sha256=sources[str(annotation)],task_prompt=PROMPT,
            progress_path=str(progress),progress_sha256=sources[str(progress)],generate_missing_progress=False)
        values = progress_values(row)
        assert values[0] == 0 and values[row['frames']-1] == 1
        row['rollback_steps_original'] = sum(values[i] < values[i-1] for i in range(1,row['frames']))
        records.append(row)
    frozen = dict(seed=42,episodes=20,tasks=1,per_task=20,sample_fps=5,
        include_episode_last_frame=True,selection={'R03':chosen},records=records,source_sha256=sources,
        source_dataset=str(data),available_train=len(task['train']),excluded_validation=task['val'],
        models=MODELS,training_started=False,desktop_automation_created=False,created=time.time())
    source_checks(frozen); save(ROOT/'selection.json',frozen)
    save(ROOT/'cache-contract.json',dict(sample_fps=5,models=MODELS,episodes=20,
        identical_selection_for_all_models=True,augmentation='none',original_dataset_modified=False,
        original_pt_rollback_and_plateaus_preserved=True,original_split_unchanged=True,
        task_prompt=PROMPT,prompt_source='Published task goal; scripted-failure sentence excluded',
        model_inputs='Existing Recovery native processors and history/view policy',
        proc_supervision='Progress answer and EOS only; no CoT supervision or previous-CoT input',
        grm_supervision='Native integer-percent score from initial BEFORE; original continuous PT retained',
        grm_future_endpoint=True,forest='Frozen Qwen3.5 vision plus CLIP/max_depth native memory',
        training_started=False))
    print(json.dumps(dict(episodes=20,selection=chosen,frames=sum(r['frames'] for r in records)),indent=2),flush=True)


def decode():
    from experiments.recover_six_caches_20261114.prepare import decode as native_decode
    selected = read(ROOT/'selection.json'); source_checks(selected)
    with ProcessPoolExecutor(2) as pool:
        receipts = list(pool.map(native_decode, selected['records']))
    for row in selected['records']:
        values = progress_values(row); observed = rows(epdir(row)/'observations.jsonl')
        expected = sorted(set(range(0,row['frames'],6)) | {row['frames']-1})
        assert [r['frame_index'] for r in observed] == expected
        for r in observed: r['pt'] = values[r['frame_index']]
        write_rows(epdir(row)/'samples.jsonl',observed)
    save(ROOT/'resolved-selection.json',selected)
    save(ROOT/'decode-verification.json',dict(status='passed',episodes=20,
        samples=sum(r['samples'] for r in receipts),records=receipts,
        selection_sha256=sha(ROOT/'selection.json'),labels_recomputed=False))


def verify():
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset
    from experiments.recover_six_caches_20261114.loader import load_input
    from experiments.recover_finetune_20261114.data import grm_answer
    selected = read(ROOT/'resolved-selection.json'); source_checks(selected)
    assert len(selected['records']) == 20
    assert set(selected['selection']['R03']).isdisjoint(selected['excluded_validation'])
    expected = {}; model_reports = {}; source_audit = []
    for row in selected['records']:
        observed = rows(epdir(row)/'samples.jsonl'); values = progress_values(row)
        indices = sorted(set(range(0,row['frames'],6)) | {row['frames']-1})
        assert [r['frame_index'] for r in observed] == indices
        assert [r['pt'] for r in observed] == [values[i] for i in indices]
        audit = read(epdir(row)/'decode-verification.json')
        assert sha(epdir(row)/'observations.jsonl') == audit['observations_sha256']
        for p,h in audit['png_sha256'].items(): assert sha(epdir(row)/p) == h
        expected[row['key']] = observed
        source_audit.append(dict(key=row['key'],samples=len(observed),
            rollback_steps_5fps=sum(observed[i]['pt'] < observed[i-1]['pt'] for i in range(1,len(observed)))))
    total = sum(map(len,expected.values()))
    for model in MODELS[:-1]:
        receipt = read(OUT/model/'cache-ready.json'); train = rows(OUT/model/'train.jsonl')
        assert receipt['episodes'] == 20 and receipt['samples'] == len(train) == total
        assert sha(OUT/model/'train.jsonl') == receipt['train_sha256']
        provenance = ROOT/f'{model}-native-provenance.json'
        assert sha(provenance) == receipt['native_provenance_sha256']
        for p,h in read(provenance)['source_sha256'].items(): assert sha(p) == h
        assert set(r['episode_id'] for r in train) == set(expected)
        for r in train:
            observed = expected[r['episode_id']]; source = observed[r['sample_index']]
            assert r['split'] == 'train' and r['task_id'] == 'R03'
            assert r['pt'] == source['pt'] and r['frame_index'] == source['frame_index']
            assert r['slots'] == [list(p) for p in slots(model,r['sample_index'],len(observed))]
        for r in receipt['episodes_verified']:
            path = Path(r['samples_file']); audit = read(path.with_suffix('.verification.json'))
            assert sha(path) == r['samples_sha256']
            for p,h in audit['pixel_sha256'].items(): assert sha(p) == h
            entries = rows(path)
            assert sha(entries[0]['prompt_tensor_file']) == r['prompt_sha256']
            for x in [entries[0],entries[len(entries)//2],entries[-1]]:
                batch, truth = load_input(x)
                assert batch['input_ids'].ndim == 2 and batch['pixel_values'].numel() > 0
                assert torch.isfinite(batch['pixel_values']).all() and truth['pt'].numel() == 1
        if model in ['robobrain25-4b','dopamine-3b']:
            targets = [dict(id=r['id'],original_pt=r['pt'],answer=grm_answer(r)) for r in train]
            write_rows(OUT/model/'text-targets.jsonl',targets)
        model_reports[model] = {k:receipt[k] for k in ['status','episodes','samples','pixel_bytes']}
    files = [OUT/'forest'/'train'/r['task_id']/r['sample_id'] for r in selected['records']]
    for row,path in zip(selected['records'],files):
        audit = read(path/'verification.json'); actual = rows(path/'samples.jsonl')
        assert sha(path/'samples.jsonl') == audit['samples_sha256']
        observed = expected[row['key']]
        assert len(actual) == len(observed)
        for i,(a,b) in enumerate(zip(actual,observed)):
            assert a['pt'] == b['pt'] and a['frame_index'] == b['frame_index']
            for v,p in enumerate(b['images']): assert sha(path/a['images'][VIEWS[v]]) == sha(p)
            if i == 0: assert a['history'] is None
            elif a['history']:
                h = torch.load(path/a['history'],map_location='cpu',weights_only=True)
                assert h['through_frame'] == observed[i-1]['frame_index']
                assert h['tokens'].ndim == 2 and torch.isfinite(h['tokens']).all()
    dataset = M3Dataset([str(p) for p in files]); assert len(dataset) == total
    forest = dict(status='native_training_cache_ready',episodes=20,samples=total,
        sample_files=[str(p) for p in files],data_signature=dataset.signature,
        history_source=dataset.history_source,selection_sha256=sha(ROOT/'selection.json'))
    save(OUT/'forest'/'cache-ready.json',forest)
    model_reports['forest'] = {k:forest[k] for k in ['status','episodes','samples']}
    source_checks(selected)
    result = dict(status='passed',models=model_reports,episodes_per_model=20,samples_per_model=total,
        task='R03',task_name='Inspect spray bottles',sample_fps=5,
        source_splits_videos_annotations_original_pt_unchanged=True,
        training_started=False,selection_sha256=sha(ROOT/'selection.json'),
        resolved_selection_sha256=sha(ROOT/'resolved-selection.json'),episodes=source_audit,
        source_checksums_sha256=sha(ROOT/'source-checksums.json'),verified=time.time())
    save(ROOT/'final-verification.json',result)


def launch(module,args,name,gpus='',python=PYTHON):
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    env.pop('PYTHONPATH',None)
    with (ROOT/(name+'.log')).open('a') as log:
        p = subprocess.Popen([python,'-m',module,*args],cwd=ROOT/'code',env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(ROOT/(name+'-launch.json'),dict(pid=p.pid,module=module,args=args,gpus=gpus,created=time.time()))
    return p


def controller():
    from experiments.prm_five_groups_20261114.overlap_guard import resources, pause_reasons
    lock = (ROOT/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'controller.pid').exists(), 'Inspect previous controller before restarting'
    (ROOT/'controller.pid').write_text(str(os.getpid()))
    initial = resources(); initial_oom = initial['oom_kill']; save(ROOT/'resource-baseline.json',initial)
    def status(phase,**extra):
        save(ROOT/'status.json',dict(status=phase,pid=os.getpid(),updated=time.time(),training_started=False,**extra))
    def wait(child,phase):
        while child.poll() is None:
            sample = resources(); status(phase,child=child.pid,resources=sample)
            time.sleep(5)
        assert child.returncode == 0, f'{phase} failed; retain outputs and inspect logs'
    package = 'experiments.recover_spray20_caches_20261116.run'
    native = 'experiments.recover_six_caches_20261114.'
    wait(launch(package,['decode'],'decode'),'decoding')
    for model in MODELS[:-1]:
        py = '/cache/cvpzhr/envs/robometer/bin/python' if model == 'robometer-4b' else CPU_PYTHON
        wait(launch(native+'model_cache',[model,'--smoke'],model+'-smoke',python=py),'native_input_smoke')
    since = None
    while True:
        sample = resources()
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                            '--format=csv,noheader'],text=True).strip()
        reasons = {g:pause_reasons(sample,g,initial_oom) for g in [0,1]}
        safe = not occupied and not any(reasons.values())
        since = (since or time.time()) if safe else None
        status('waiting_for_idle_GPUs',reasons=reasons,occupied=occupied)
        if since and time.time()-since >= 30: break
        time.sleep(5)
    forest = [launch(native+'gpu_cache',['forest','--worker',str(i)],f'forest-{i}',str(i)) for i in [0,1]]
    active = {}; pending = MODELS[:-1].copy(); failures = []
    while active or pending or any(p.poll() is None for p in forest):
        sample = resources()
        for model,p in list(active.items()):
            if p.poll() is not None:
                if p.returncode: failures.append(model)
                del active[model]
        safe = (sample['oom_kill'] == initial_oom and not sample['under_oom'] and
                sample['resident_and_shmem'] < .60*sample['memory_limit'] and
                sample['memory_working_set'] < .82*sample['memory_limit'] and
                sample['disk_free'] > 100*2**30)
        if pending and len(active) < 2 and safe:
            m = pending.pop(0)
            py = '/cache/cvpzhr/envs/robometer/bin/python' if m == 'robometer-4b' else CPU_PYTHON
            active[m] = launch(native+'model_cache',[m],m,python=py)
        status('building_caches',active={m:p.pid for m,p in active.items()},pending=pending,failed=failures,
            forest=[dict(pid=p.pid,exit=p.poll()) for p in forest],resources=sample)
        if not active and not any(p.poll() is None for p in forest) and not safe:
            raise RuntimeError('Resource guard blocked further preparation; no automatic retry')
        time.sleep(5)
    assert not failures and all(p.returncode == 0 for p in forest),(failures,[p.returncode for p in forest])
    wait(launch(package,['verify'],'verify'),'verifying_all_caches')
    wait(launch(native+'proc_progress_only',[],'proc-progress-only',python=CPU_PYTHON),'adding_progress_only_targets')
    result = read(ROOT/'final-verification.json'); proc = read(ROOT/'proc-progress-only-verification.json')
    assert proc['status'] == 'passed' and proc['episodes'] == 20 and proc['samples'] == result['samples_per_model']
    result.update(status='completed',proc_progress_only_supervision_verified=True,
        proc_training_data=str(OUT/'procvlm-2b'/'progress-only/train.jsonl'),finished=time.time())
    save(ROOT/'verification.json',result);save(OUT/'verification.json',result);save(ROOT/'status.json',result)
    print(json.dumps({k:result[k] for k in ['status','models','samples_per_model']},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','decode','verify','controller'])
    action = parser.parse_args().action
    try: globals()[action]()
    except BaseException:
        if action == 'controller':
            save(ROOT/'controller-error.json',dict(error=traceback.format_exc(),time=time.time()))
            save(ROOT/'status.json',dict(status='failed',error=traceback.format_exc(),updated=time.time()))
        raise
