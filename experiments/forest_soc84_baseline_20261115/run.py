import argparse
import csv
import fcntl
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from collections import Counter
from experiments.soc_grm_validation_20261115.common import read, rows, sha, save, freeze, resolve, SOURCE, SELECTION, EXPECTED
from experiments.soc_disrupted_eval_20261115.guard import process_identity, alive, gpu_processes, host_sample, idle_safe
from experiments.recover_validation_20261114.common import safe
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics

ROOT = Path('/cache/cvpzhr/setup/forest-soc84-baseline-20261115')
OUT = Path('/cache/cvpzhr/evaluations/forest-soc84-baseline-20261115')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
PACKAGE = 'experiments.forest_soc84_baseline_20261115.run'
SUITES = ('State','Order','Count')
GPUS = (0,1)


def digest(value):
    import hashlib,json
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def shards(episodes):
    parts=[[],[]]; load=[0,0]
    for row in sorted(episodes,key=lambda r:(-len(r['frame_indices']),r['key'])):
        i=min(range(2),key=lambda i:load[i]);parts[i].append(row['key']);load[i]+=len(row['frame_indices'])
    return parts


def model_input(row):
    return dict(key=row['key'],task=row['task_prompt'],positions=row['frame_indices'],images=[
        [str(Path(row['image_root'])/f'{f:08d}-{v}.png') if p else None for v,p in enumerate(row['videos'])]
        for f in row['frame_indices']])


def result_path(key): return OUT/'episodes'/(key.replace('/','__')+'.json')


def identity(inputs, row):
    return dict(inputs_sha256=sha(ROOT/'inputs.json'),model='forest',suite=row['suite'],key=row['key'],
        checkpoint_sha256=inputs['registry'][row['suite']]['checkpoint_sha256'],condition='baseline_memory_5fps')


def validate(record,row,expected,complete=False):
    assert record['identity']==expected
    n=len(record['predictions']);assert n<=len(row['frame_indices'])
    if complete:assert record['status']=='completed' and n==len(row['frame_indices'])
    for k in ('frame_indices','timestamps','target_pt'):assert record[k]==row[k][:n]
    assert len(record['raw_responses'])==len(record['input_audits'])==n
    assert all(p is None or (type(p) in (float,int) and math.isfinite(p)) for p in record['predictions'])
    a=record['initialization'];assert digest(a)==record['initialization_sha256']
    assert a['strict_final_state'] and a['memory_enabled'] and not a['future_endpoint_reference']
    assert a['checkpoint_sha256']==expected['checkpoint_sha256'] and a['suite']==row['suite']
    for i,a in enumerate(record['input_audits']):
        assert a['playback_step']==i and a['playback_position']==row['frame_indices'][i]
        assert a['input_images']==3 and [list(s) for s in a['input_playback_slots']]==[[i,0],[i,1],[i,2]]
    return n


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ROOT.mkdir(parents=True,exist_ok=True)
    source=read(SOURCE);eps=source['episodes'];assert Counter(r['suite'] for r in eps)==EXPECTED
    assert len(eps)==len({r['key'] for r in eps})==84
    assert set(Counter((r['suite'],r['task_id']) for r in eps).values())=={2}
    hashes={str(SOURCE):sha(SOURCE),str(SELECTION):sha(SELECTION)};resolved={}
    def protect(path,expected=None):
        expected=expected or sha(path);actual=resolve(path,expected)
        hashes[str(actual)]=expected;resolved[str(path)]=str(actual);return actual
    for p,h in read(SELECTION)['source_hashes'].items():protect(p,h)
    for row in eps:
        assert row['split']=='val' and row['input_fps']==5 and not row['input_issues']
        pt=protect(row['progress_path'],row['progress_sha256'])
        with pt.open() as f:truth=list(csv.DictReader(f))
        assert [int(r['frame_index']) for r in truth]==list(range(row['frames']))
        assert row['target_pt']==[float(truth[i]['pt']) for i in row['frame_indices']]
        assert row['timestamps']==[i/row['fps'] for i in row['frame_indices']]
        cache=Path(row['image_root']);proof=read(cache/'verification.json');protect(cache/'verification.json')
        assert proof['input']==dict(video_hashes=row['video_hashes'],frame_indices=row['frame_indices'])
        assert proof['decoded_native_frames']==row['frames'] and proof['lossless']
        expected={f'{i:08d}-{v}.png' for i in row['frame_indices'] for v,p in enumerate(row['videos']) if p}
        assert set(proof['png_sha256'])==expected
        for n,h in proof['png_sha256'].items():assert sha(cache/n)==h
    registry={s:source['registry'][f'forest-{s}'] for s in SUITES}
    import torch
    torch.set_num_threads(2)
    for s,info in registry.items():
        protect(info['checkpoint'],info['checkpoint_sha256']);protect(info['proof'],info['proof_sha256'])
        proof=read(info['proof']);assert proof['status']=='completed' and proof['final_checkpoint_verified'] and proof['fresh_initialization_verified']
        ck=torch.load(info['checkpoint'],map_location='cpu',weights_only=True)
        assert ck['epoch']==1 and ck['next_batch']==0 and ck['config']==info['config'];del ck
        cfg=info['config'];assert cfg['run_protocol']['suite']==s and cfg['run_protocol']['train_episodes']==35
        assert cfg['event_split_method']=='max_depth'
        clip=Path(cfg['history_source']['memory_config']['clip_model_path'])
        for p in clip.rglob('*'):
            if p.is_file() and '.cache' not in p.parts and str(p) in source['frozen_files']:protect(p,source['frozen_files'][str(p)])
    receipt=read('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')
    for n,item in receipt['files'].items():protect(Path(receipt['destination'])/n,item['sha256'])
    training=read('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114/selection.json')
    assert not {r['episode_id'] for r in training['episodes']} & {r['key'] for r in eps}
    plan=dict(protocol='SOC84 unperturbed baseline with native FoREST memory',model='forest',
        input_fps=5,include_original_last_frame=True,memory_enabled=True,future_endpoint_reference=False,
        selected_trajectories=EXPECTED,weights='matching suite SOC35 final 1 epoch',
        suite_order=list(SUITES),gpus=list(GPUS),workers_per_gpu=1,batch=1,
        scheduling='Both GPUs evaluate trajectory shards of the same suite; next suite only after both exit and suite audit passes',
        metrics=['pearson','spearman','VS','ER','C'],aggregation='trajectory-equal; invalid/undefined denominators reported',
        perturbations=False,no_memory_ablation=False,multifps=False,
        original_sources_unchanged=True,protected_files=hashes,resolved_sources=resolved,
        registry=registry,episodes=eps,shards={s:shards([r for r in eps if r['suite']==s]) for s in SUITES},
        source_inputs_sha256=sha(SOURCE),selection_sha256=sha(SELECTION))
    freeze(ROOT/'inputs.json',plan)
    save(ROOT/'preparation.json',dict(status='passed',episodes=84,observations=15096,
        inputs_sha256=sha(ROOT/'inputs.json'),gpu_forward_tested=False))
    print('prepared 84 trajectories; State22 -> Order32 -> Count30; two GPUs per suite',flush=True)


def audit_report(inputs, required_suite=None):
    records=[];missing=[]
    for row in inputs['episodes']:
        if required_suite and row['suite']!=required_suite:continue
        path=result_path(row['key'])
        if not path.exists():missing.append(row['key']);continue
        r=read(path);validate(r,row,identity(inputs,row),True)
        metrics=trajectory_metrics(r['predictions'],row['target_pt'],row['timestamps'])
        assert r['metrics']==metrics
        records.append(dict(key=row['key'],suite=row['suite'],task_id=row['task_id'],**metrics,result_sha256=sha(path)))
    if required_suite:assert not missing,missing
    summaries={}
    for s in SUITES:
        rr=[r for r in records if r['suite']==s];full=len(rr)==EXPECTED[s]
        summaries[s]=dict(completed=len(rr),expected=EXPECTED[s],complete=full,
            invalid_frames=sum(r['invalid_predictions'] for r in rr),metrics={})
        for k in ('pearson','spearman','VS','ER','C'):
            vals=[r[k] for r in rr if r[k] is not None]
            summaries[s]['metrics'][k]=dict(mean=math.fsum(vals)/len(vals) if vals and full else None,
                defined=len(vals),undefined=len(rr)-len(vals))
    value=dict(status='completed' if not missing else 'in_progress',model='forest',by_suite=summaries,
        missing=missing,episodes=records,inputs_sha256=sha(ROOT/'inputs.json'),
        metric_provenance=read(Path(__file__).parents[1]/'prm_five_groups_20261114/metric_reference/provenance.json'))
    if required_suite:
        value['status']='suite_completed';save(ROOT/f'{required_suite}-verification.json',value)
    else:
        save(OUT/'report.json',value)
        if records:
            columns=['key','suite','task_id','pearson','spearman','VS','ER','C','samples','invalid_predictions','correlation_undefined_reason','vs_undefined_reason']
            OUT.mkdir(parents=True,exist_ok=True)
            with (OUT/'trajectory-metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r.get(k) for k in columns} for r in records)
    return value


def worker(ticket_path):
    ticket=read(ticket_path);s=ticket['suite'];part=ticket['shard'];gpu=ticket['gpu']
    folder=ROOT/'jobs'/f'{s}-{part}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(process=process_identity(os.getpid()),suite=s,shard=part,gpu=gpu,status='starting')
    def update(**kw):status.update(kw,updated=time.time());save(folder/'status.json',status)
    update()
    assert process_identity(os.getppid())==ticket['controller'] and alive(ticket['controller'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    assert idle_safe(host_sample(),gpu_processes(),gpu,ticket['initial_oom'])
    inputs=read(ROOT/'inputs.json');assert sha(ROOT/'inputs.json')==ticket['inputs_sha256']
    last=[0]
    def check():
        if time.time()-last[0]<5:return
        last[0]=time.time();state=read(ROOT/'status.json')
        if not alive(ticket['controller']) or state['controller']!=ticket['controller'] or time.time()-state['updated']>180:
            raise RuntimeError('Controller missing or stale; preserve prefix')
        if (ROOT/'pause-all').exists() or not safe(host_sample(),gpu,ticket['initial_oom']) or len(gpu_processes()[gpu])>1:
            raise SafetyPause('Resource guard or cooperative pause')
    from experiments.soc_disrupted_eval_20261115.engine import SafetyPause
    try:
        import random,numpy as np,torch
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.70);torch.cuda.manual_seed_all(42)
        from experiments.soc_disrupted_eval_20261115.backend import Backend
        update(status='loading_model');backend=Backend(inputs['registry'][s]);save(folder/'initialization.json',backend.audit)
        selected={r['key']:r for r in inputs['episodes']};keys=inputs['shards'][s][part]
        # Native sequential memory, reset, and prefix-rebuild checks on actual 5-FPS images.
        row=selected[keys[0]];episode=model_input(row);backend.begin(episode);ref=[]
        update(status='checking_native_memory')
        for i in range(min(16,len(row['frame_indices']))):check();ref.append(backend.predict(i))
        assert all(v['prediction'] is not None for v in ref)
        assert ref[-1]['input_audit']['history_tokens']>0
        backend.begin(episode)
        for i in range(8):check();backend.rebuild_step(i)
        replay=[]
        for i in range(8,len(ref)):check();replay.append(backend.predict(i))
        error=max(abs(a['prediction']-b['prediction']) for a,b in zip(ref[8:],replay));assert error<=.002
        backend.begin(episode);assert abs(backend.predict(0)['prediction']-ref[0]['prediction'])<=1e-6
        save(folder/'smoke.json',dict(status='passed',gpu_forward_tested=True,checkpoint_sha256=backend.audit['checkpoint_sha256'],
            resume_error=error,reset_verified=True,native_history_verified=True))
        completed=0
        for key in keys:
            row=selected[key];dest=result_path(key);partial=dest.with_suffix('.partial.json');expected=identity(inputs,row)
            if dest.exists():validate(read(dest),row,expected,True);completed+=1;continue
            backend.begin(model_input(row))
            record=read(partial) if partial.exists() else dict(identity=expected,predictions=[],raw_responses=[],input_audits=[],
                frame_indices=[],timestamps=[],target_pt=[],initialization=backend.audit,initialization_sha256=digest(backend.audit),status='partial')
            start=validate(record,row,expected);assert record['initialization']==backend.audit
            update(status='running',episode=key,completed_episodes=completed,expected_episodes=len(keys),observations_in_episode=start)
            for i in range(start):check();backend.rebuild_step(i)
            saved=0
            try:
                for i in range(start,len(row['frame_indices'])):
                    check();value=backend.predict(i)
                    record['predictions'].append(value['prediction']);record['raw_responses'].append(value['raw_response'])
                    record['input_audits'].append(value['input_audit'])
                    for k in ('frame_indices','timestamps','target_pt'):record[k].append(row[k][i])
                    if time.time()-saved>=2:
                        save(partial,record);saved=time.time();update(observations_in_episode=i+1)
            finally:save(partial,record)
            record.update(status='completed',metrics=trajectory_metrics(record['predictions'],row['target_pt'],row['timestamps']))
            validate(record,row,expected,True);save(dest,record);partial.unlink();completed+=1
        update(status='completed',completed_episodes=completed);return 0
    except SafetyPause as e:update(status='paused',reason=str(e));return 75
    except BaseException:update(status='failed',error=traceback.format_exc());raise


def controller():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'status.json').exists():assert not alive(read(ROOT/'status.json').get('controller'))
    for p in (ROOT/'jobs').glob('*/status.json'):assert not alive(read(p).get('process'))
    for p,h in read(ROOT/'source-checksums.json')['files'].items():assert sha(ROOT/'code'/p)==h,p
    inputs=read(ROOT/'inputs.json');assert read(ROOT/'preparation.json')['inputs_sha256']==sha(ROOT/'inputs.json')
    identity_=process_identity(os.getpid());(ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    initial=host_sample();status=dict(status='starting',controller=identity_,suite_order=list(SUITES),gpus=list(GPUS),started=time.time())
    def update(**kw):status.update(kw,updated=time.time());save(ROOT/'status.json',status)
    update()
    for s in SUITES:
        pending=[0,1];active={};done=[];stable=None
        while pending or active:
            now=time.time();processes=gpu_processes();sample=host_sample()
            if not active:
                if all(idle_safe(sample,processes,g,initial['oom_kill']) for g in GPUS):stable=stable or now
                else:stable=None
            if pending and not active and not status.get('failures') and stable is not None and now-stable>=60 and not (ROOT/'pause-all').exists():
                for part in pending:
                    folder=ROOT/'jobs'/f'{s}-{part}';folder.mkdir(parents=True,exist_ok=True)
                    tp=folder/f'ticket-{time.time_ns()}.json';save(tp,dict(suite=s,shard=part,gpu=GPUS[part],controller=identity_,
                        initial_oom=initial['oom_kill'],inputs_sha256=sha(ROOT/'inputs.json')))
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[part]),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                        TRITON_CACHE_DIR=str(ROOT/'triton'/f'{s}-{part}'))
                    for k in ('PYTHONPATH','FOREST_NATIVE_GATED_NORM','PRM_CUDA_MEMORY_FRACTION','PYTORCH_CUDA_ALLOC_CONF'):env.pop(k,None)
                    log=folder/f'worker-{time.time_ns()}.log'
                    with log.open('w') as f:p=subprocess.Popen([PYTHON,'-m',PACKAGE,'worker','--ticket',str(tp)],cwd=ROOT/'code',env=env,
                        stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    active[part]=dict(process=p,log=str(log))
                pending=[];stable=None
            for part,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                record=read(ROOT/'jobs'/f'{s}-{part}'/'status.json')
                if code==0 and record['status']=='completed':done.append(part)
                elif code==75:pending.append(part)
                else:
                    status.setdefault('failures',[]).append(dict(suite=s,shard=part,returncode=code,log=item['log']))
                del active[part]
            update(status='running' if active else 'waiting_for_safe_idle_gpus',suite=s,completed_shards=done,pending_shards=pending,
                active={str(k):dict(pid=v['process'].pid,gpu=GPUS[k],log=v['log']) for k,v in active.items()})
            if status.get('failures') and not active:raise RuntimeError(status['failures'])
            if (ROOT/'pause-all').exists() and not active:update(status='paused');return
            if pending or active:time.sleep(5)
        audit_report(inputs,s);audit_report(inputs)
        update(status='suite_completed',suite=s)
    result=audit_report(inputs);assert result['status']=='completed'
    update(status='completed',report=str(OUT/'report.json'),finished=time.time(),active={})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker','report']);p.add_argument('--ticket');a=p.parse_args()
    try:
        if a.action=='worker':raise SystemExit(worker(a.ticket))
        elif a.action=='report':print(audit_report(read(ROOT/'inputs.json'))['by_suite'])
        else:globals()[a.action]()
    except Exception:
        if a.action=='controller':
            status=read(ROOT/'status.json');status.update(status='failed',error=traceback.format_exc(),updated=time.time());save(ROOT/'status.json',status)
        raise
