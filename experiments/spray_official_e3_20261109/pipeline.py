from pathlib import Path
import os,sys,json,time,subprocess,fcntl,traceback,hashlib,math,shutil
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def records(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    busy=set(x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
    return {int(c[0]):{'memory_mib':int(c[2]),'utilization':int(c[3]),'occupied':c[1].strip() in busy} for line in raw.splitlines() if (c:=line.split(','))}
def lane(task):
    policy=read(ROOT/'launch-policy.json');t=policy['tasks'][task];s=Path(t['setup_dir']);out=Path(t['output_dir']);ev=Path(t['evaluation_dir']);py=t['python'];gpus=t['gpus']
    lock=(s/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'task':task,'pid':os.getpid(),'started_at':time.time(),'initialization':t['initialization'],'output_dir':str(out),'evaluation_dir':str(ev),'gpu_ids':gpus,'epochs':3}
    def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(s/'status.json',state)
    pythonpath=str(s/'code')+(':'+policy['robometer_upstream'] if task=='robometer' else '')
    env=dict(os.environ,PYTHONPATH=pythonpath,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='2' if task=='forest' else '1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             PYTHONNOUSERSITE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def tr(target,args,module=False):return [py,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2']+(['-m'] if module else [])+[target]+args
    def wait_idle(phase):
        while True:
            info=gpu_state()
            if all(not info[g]['occupied'] and info[g]['memory_mib']<1024 and info[g]['utilization']<10 for g in gpus):
                time.sleep(3);info=gpu_state()
                if all(not info[g]['occupied'] and info[g]['memory_mib']<1024 and info[g]['utilization']<10 for g in gpus):return
            update(phase,gpu_state=info);time.sleep(10)
    def run(cmd,log,phase,gpu=True):
        update(phase,command=cmd,log=str(log),child_pid=None)
        with log.open('a') as f:
            child=subprocess.Popen(cmd,cwd=s,env=env if gpu else dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            update(phase,child_pid=child.pid)
            while child.poll() is None:
                extra={};current=phase
                if phase=='training':
                    rr=records(out/('train.jsonl' if task=='forest' else 'metrics.jsonl'))
                    if rr:extra['last_training_record']=rr[-1]
                    extra['saved_epochs']=([i for i in range(1,4) if (out/f'epoch-{i:02d}.pt').exists()] if task=='forest' else [i for i in range(1,4) if (out/f'checkpoint-{412*i}/complete.json').exists()])
                    if task=='robometer':
                        lifecycle=records(out/'lifecycle.jsonl')
                        if lifecycle and lifecycle[-1]['event']=='validation_started':current='evaluating_epoch3'
                if phase=='evaluating_epoch3':extra['evaluated_samples']=sum(read(p).get('samples',0) for p in ev.glob('worker-*.json'))
                update(current,**extra);time.sleep(10)
        update(phase,child_pid=None,returncode=child.returncode)
        if child.returncode:raise RuntimeError(f'{phase} exited {child.returncode}; see {log}')
    def verify_init(folder):
        result=[]
        for rank in [0,1]:
            x=read(folder/f'initialization-rank{rank}.json');assert x['initial_checkpoint'] is None and x['fresh_lora_B_all_zero']
            assert x['optimizer_reset'] and x['scheduler_reset'] and x['training_counter_reset']
            if task=='robometer':assert x['strict_original_weight_load'] and x['official_progress_head_exactly_preserved']
            else:assert x['initialization']=='fresh Qwen3.5-4B base' and set(x['fresh_components'])=={'m3_lora','history_projection','progress_head'}
            result.append(x)
        key='initial_trainable_sha256' if task=='forest' else 'official_progress_head_sha256'
        assert result[0][key]==result[1][key]
    try:
        update('preflight');assert read(ROOT/'preflight.json')['status']=='passed'
        for rel,sha in read(ROOT/'run-files-sha256.json').items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
        assert not out.exists(),'Existing output requires explicit recovery audit'
        cfg=read(s/'training.config.json');smoke=s/'smoke';smoke.mkdir()
        wait_idle('waiting_for_two_gpus')
        if task=='forest':
            test=dict(cfg,max_steps=4,log_every_steps=1,output_dir=str(smoke/'output'),initialization_audit_dir=str(smoke))
            save(smoke/'config.json',test)
            run(tr('mem_lite.train_stage3',['--config',str(smoke/'config.json')],True),smoke/'run.log','gpu_smoke')
            rows=records(smoke/'output/train.jsonl');assert len(rows)==4 and rows[-1]['step']==4
            assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['samples']==18 for r in rows)
            peak=max(max(r['peak_reserved_gib_per_gpu']) for r in rows)
            assert peak<74
            save(s/'smoke-verification.json',{'status':'passed','steps':4,'peak_reserved_gib':peak,'weights_reused':False,'last_step_seconds':rows[-1]['step_seconds']})
        else:
            run(tr(str(s/'train_official.py'),['--config',str(s/'training.config.json'),'--smoke']),smoke/'run.log','gpu_smoke')
            rows=[read(smoke/f'smoke-verification-rank{r}.json') for r in [0,1]]
            assert all(r['status']=='passed' and r['steps']==4 and r['peak_reserved_gib']<78 for r in rows)
            save(s/'smoke-verification.json',{'status':'passed','ranks':rows,'weights_reused':False})
        verify_init(smoke)
        if (smoke/'output').exists():shutil.rmtree(smoke/'output')
        wait_idle('waiting_for_training_gpus');state['training_started_at']=time.time()
        command=tr('mem_lite.train_stage3',['--config',str(s/'training.config.json')],True) if task=='forest' else tr(str(s/'train_official.py'),['--config',str(s/'training.config.json')])
        run(command,s/'train.log','training');state['training_finished_at']=time.time();verify_init(s)
        if task=='forest':
            run([py,str(s/'verify_checkpoints.py')],s/'checkpoint-verification.log','verifying_checkpoints',False)
            wait_idle('waiting_for_evaluation_gpus');state['evaluation_started_at']=time.time()
            args=['--checkpoint',str(out/'epoch-03.pt'),'--output',str(ev),'--batch-size','4','--workers','4']
            run(tr(str(s/'evaluate_final.py'),args),s/'evaluation.log','evaluating_epoch3')
            run([py,str(s/'evaluate_final.py'),*args,'--aggregate'],s/'evaluation-aggregate.log','aggregating',False)
            run([py,str(s/'plot_validation.py')],s/'plotting.log','plotting',False)
            metrics=read(ev/'metrics.json');assert read(ev/'curves/manifest.json')['status']=='passed';checkpoint=str(out/'epoch-03.pt');overview=str(ev/'curves/overview.png')
        else:
            run([py,str(s/'verify_robometer.py')],s/'checkpoint-verification.log','verifying_checkpoints',False)
            code="import postprocess_validation as p; p.render(p.read_json(p.SETUP/'training.config.json'))"
            run([py,'-c',code],s/'plotting.log','plotting',False)
            metrics=read(ev/'verified-metrics.json');checkpoint=str(out/'checkpoint-1236');overview=str(ev/'curves/validation_overview.png')
            manifest={'status':'passed','episodes':7,'samples':1919,'checkpoint':checkpoint,'smoothing':False,
                      'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ev/'curves').iterdir() if p.is_file() and p.suffix in ['.png','.pdf','.json','.csv']}}
            save(ev/'curves/manifest.json',manifest)
        assert metrics['coverage_complete'] and metrics['overall']['episodes']==7 and metrics['overall']['frames']==1919
        update('completed',finished_at=time.time(),checkpoint=checkpoint,metrics=metrics['overall'],curve_path=overview,saved_epochs=[1,2,3])
    except BaseException as exc:
        update('failed',failed_phase=state.get('phase'),error=repr(exc),traceback=traceback.format_exc());raise
def main():
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children={};logs=[];started=time.time()
    for task in read(ROOT/'launch-policy.json')['tasks']:
        log=(ROOT/task/'pipeline.log').open('a');logs.append(log)
        children[task]=subprocess.Popen([sys.executable,'-u',str(__file__),'--task',task],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while True:
        states={task:read(ROOT/task/'status.json') if (ROOT/task/'status.json').exists() else {'phase':'starting'} for task in children}
        for task,child in children.items():
            if child.poll() is not None and child.returncode and states[task].get('phase')!='failed':states[task]={'phase':'failed','error':f'lane process exited {child.returncode}'}
        done=all(c.poll() is not None for c in children.values());failed=any(s['phase']=='failed' for s in states.values())
        phase=('failed' if failed else 'completed') if done else ('running_with_failure' if failed else 'running')
        save(ROOT/'status.json',{'pid':os.getpid(),'started_at':started,'updated_at':time.time(),'phase':phase,'tasks':states})
        if done:
            if not failed:save(ROOT/'comparison.json',{'forest':states['forest']['metrics'],'robometer':states['robometer']['metrics'],'source':'official/base initialization; final epoch 3; same clean validation set'})
            return
        time.sleep(10)
if __name__=='__main__':
    if '--task' in sys.argv:lane(sys.argv[sys.argv.index('--task')+1])
    else:main()
