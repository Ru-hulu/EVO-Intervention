"""Run disposable checks only; never launch the formal one-epoch training."""
from pathlib import Path
import os,json,time,subprocess,signal,statistics,math,traceback,hashlib
SETUP=Path(__file__).resolve().parent
CODE=Path('/cache/cvpzhr/code/robometer-current3-w8-e1-20261104')
PY='/cache/cvpzhr/envs/robometer/bin/python'
ENV={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'),
     'CUDA_VISIBLE_DEVICES':'0,1,2,3','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
     'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'PYTHONNOUSERSITE':'1','NCCL_DEBUG':'WARN'}
state={'pid':os.getpid(),'started_at':time.time(),'formal_training_started':False}
def save(phase,**kw):
    state.update(phase=phase,updated_at=time.time(),**kw)
    p=SETUP/'preflight-status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(SETUP/'preflight-status.json')
def run(phase,args):
    save(phase)
    with (SETUP/(phase+'.log')).open('w') as log:
        p=subprocess.Popen([PY,*args],cwd=SETUP,env=ENV,stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
        save(phase,child_pid=p.pid)
        try:code=p.wait(timeout=1200)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if code:raise RuntimeError(f'{phase} exited with {code}; see {phase}.log')
def dist(args):
    return ['-m','torch.distributed.run','--standalone','--nproc_per_node=4',*args]
try:
    save('checking')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not apps,'GPUs became occupied; do not stop any task automatically'
    assert json.loads((SETUP/'cpu-tests.passed.json').read_text())['passed']
    run('benchmark',dist([str(SETUP/'benchmark.py'),'--config',str(SETUP/'benchmark.config.json')]))
    output=SETUP/'benchmark-output'
    result=json.loads((output/'smoke-result.json').read_text())
    assert result['step']==40 and result['scheduler_last_epoch']==40 and result['optimizer_state_entries']>0
    events=[json.loads(line) for line in (output/'lifecycle.jsonl').read_text().splitlines()]
    for step in [20,40]:
        part=[e for e in events if e['step']==step]
        names=[e['event'] for e in part]
        assert names==['checkpoint_save_started','checkpoint_saved','validation_started','validation_finished'],names
        receipt=json.loads((output/f'checkpoint-{step}'/'complete.json').read_text())
        assert receipt['time'] <= next(e['time'] for e in part if e['event']=='validation_started')
        val=json.loads((output/f'checkpoint-{step}'/'validation.json').read_text())
        assert val['eval_samples']==752
        assert sum(val['eval_'+suite+'_samples'] for suite in ['Count','Order','State'])==752
    run('resume',dist([str(SETUP/'benchmark.py'),'--config',str(SETUP/'benchmark.config.json'),
        '--resume',str(output/'checkpoint-20'),'--output',str(SETUP/'resume-output'),'--stop-after','22']))
    resumed=json.loads((SETUP/'resume-output'/'smoke-result.json').read_text())
    assert resumed['step']==22 and resumed['scheduler_last_epoch']==22 and resumed['optimizer_state_entries']>0
    for rank in range(4):
        timing=json.loads((SETUP/'resume-output'/f'timing-rank{rank}.json').read_text())
        assert [r['step'] for r in timing['records']]==[21,22]
    timings=[json.loads((output/f'timing-rank{r}.json').read_text()) for r in range(4)]
    measured_steps=list(range(6,21))+list(range(26,41))
    values=[]
    for step in measured_steps:
        values.append(max(next(v['seconds'] for v in t['records'] if v['step']==step) for t in timings))
    eval_values=[]
    for step in ['20','40']:
        n=min(len(t['evaluation'][step]) for t in timings)
        for i in range(3,n-1):
            eval_values.append(max(t['evaluation'][step][i]['seconds'] for t in timings))
    train_step=statistics.mean(values);eval_batch=statistics.mean(eval_values)
    nsteps=9967;nbatches=math.ceil(66891/32)
    save_seconds=[e['seconds'] for e in events if e['event']=='checkpoint_saved']
    train_seconds=train_step*nsteps
    validation_seconds=eval_batch*nbatches*2
    startup_allowance=120
    total=train_seconds+validation_seconds+sum(save_seconds)+startup_allowance
    estimate=dict(status='passed',measured_train_steps=len(values),mean_train_step_seconds=train_step,
        median_train_step_seconds=statistics.median(values),train_step_min=min(values),train_step_max=max(values),
        measured_validation_batches=len(eval_values),mean_validation_batch_seconds=eval_batch,
        samples_per_training_second=32/train_step,samples_per_validation_second=32/eval_batch,
        peak_allocated_gib=max(t['peak_allocated_gib'] for t in timings),
        peak_reserved_gib=max(t['peak_reserved_gib'] for t in timings),
        epoch_steps=nsteps,validation_steps=[4984,9967],validation_samples=66891,
        predicted_training_hours=train_seconds/3600,
        predicted_two_validations_minutes=validation_seconds/60,
        checkpoint_seconds=save_seconds,startup_allowance_seconds=startup_allowance,
        predicted_total_hours=total/3600,rough_total_hours_range=[total*.85/3600,total*1.15/3600],
        checkpoint_before_validation_verified=True,resume_verified=True,
        official_weights_unchanged=True,formal_training_started=False,
        note='Short four-GPU benchmark with actual AdamW updates in disposable adapters; forecast assumes GPUs remain dedicated.',
        config=str(SETUP/'config.json'),code=str(CODE))
    (SETUP/'time-estimate.json').write_text(json.dumps(estimate,indent=2))
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CODE/'robometer_training').glob('*.py')}
    (SETUP/'code-sha256.json').write_text(json.dumps(files,indent=2))
    save('completed',child_pid=None,report=str(SETUP/'time-estimate.json'))
    print(json.dumps(estimate,indent=2),flush=True)
except BaseException as exc:
    save('failed',error=repr(exc),traceback=traceback.format_exc())
    raise
