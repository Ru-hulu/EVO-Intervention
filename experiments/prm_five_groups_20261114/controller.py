"""Persistent two-GPU training queue, with cooperative cache draining and eval handoff."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiments.prm_five_groups_20261114.prepare_cache import BASE, connect, save, sha
from .evaluation_priority import policy, AdoptedTraining, matches

CODE = Path(__file__).parents[2]
PYTHON = '/cache/cvpzhr/envs/forest-labels/bin/python'
SUITES = ['State', 'Count', 'Order', 'Recover']
COUNTS = {'State':352, 'Count':699, 'Order':468, 'Recover':130}
CACHE_WORKERS_PER_GPU = 2


def alive(pid):
    try:
        status = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        return status != 'Z'
    except FileNotFoundError:
        return False


def env(gpus):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), OMP_NUM_THREADS='4',
                MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                PRM_CUDA_MEMORY_FRACTION='0.70',
                PYTORCH_ALLOC_CONF='expandable_segments:True')


def spawn(command, log, environment):
    with Path(log).open('a') as f:
        return subprocess.Popen(command, cwd=CODE, env=environment, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)


def training_config(suite, gpus, *, smoke=False):
    from mem_lite.datasets.m3_dataset import M3Dataset
    db=connect()
    jobs=db.execute('SELECT payload,status FROM jobs WHERE suite=? AND split="train" ORDER BY key',(suite,)).fetchall()
    done=[json.loads(j['payload']) for j in jobs if j['status']=='completed']
    if smoke:
        done=done[:2]
    elif len(done)!=COUNTS[suite] or len(jobs)!=COUNTS[suite]:
        raise ValueError(f'Incomplete training cache for {suite}')
    if not done:
        raise ValueError('No completed training cache')
    for r in done:
        directory=Path(r['output']); proof=json.loads((directory/'cache-verification.json').read_text())
        if sha(directory/'samples.jsonl')!=proof['samples_sha256']:
            raise ValueError(f'Cache changed: {directory}')
        if sha(r['progress_path'])!=r['progress_sha256']:
            raise ValueError(f'Labels changed: {r["key"]}')
    files=[r['output'] for r in done]
    dataset=M3Dataset(files)
    config=json.loads((CODE/'mem_lite/configs/stage3_train.json').read_text())
    output=BASE/'training-smoke' if smoke else Path(f'/cache/cvpzhr/outputs/forest-{suite.lower()}-qwen35-clip-depth-e1-20261114')
    config.update(model_path='/cache/cvpzhr/models/Qwen3.5-4B',output_dir=str(output),sample_files=files,
        epochs=1,max_steps=2 if smoke else None,batch_size=1,gradient_accumulation_steps=9,
        num_workers=2,persistent_workers=True,fused_optimizer=True,learning_rate=1e-4,
        lr_schedule='cosine',warmup_ratio=.03,save_every_steps=100,save_every_epochs=1,
        log_every_steps=1 if smoke else 10,seed=42,history_capacity=dataset.history_capacity,
        data_signature=dataset.signature,history_source=dataset.history_source,event_split_method='max_depth',
        audit_initialization=True,base_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        run_protocol={'suite':suite,'world_size':2,'gpus':gpus,'effective_batch':18,'fps':5,
                      'initialization':'independent original base, fresh language LoRA, history projection and progress head',
                      'train_episodes':len(done),'train_samples':len(dataset),'epochs':1,
                      'trial_only':smoke,'trial_weights_reused':False,'artificial_augmentation':False})
    config['model'].update(history_dim=dataset.history_dim,gradient_checkpointing=True)
    config['cuda_memory_fraction']=0.70
    folder=BASE/('smoke' if smoke else suite);folder.mkdir(exist_ok=True)
    path=folder/'training.config.json';save(path,config)
    return path,config


def verify_training(suite, config, *, smoke=False):
    import torch
    path=Path(config['output_dir'])/('last.pt' if smoke else 'epoch-01.pt')
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    expected=2 if smoke else ckpt['config']['total_optimizer_steps']
    assert ckpt['step']==expected and ckpt['scheduler']['last_epoch']==expected
    assert len(ckpt['rng_by_rank'])==2 and ckpt['optimizer']['state']
    if not smoke:
        assert ckpt['epoch']==1 and ckpt['next_batch']==0
    audits=[json.loads((path.parent/f'initialization-rank{r}-fresh.json').read_text()) for r in range(2)]
    assert all(a['fresh_language_lora'] and not a['resume'] and a['world_size']==2 for a in audits)
    assert audits[0]['initial_trainable_sha256']==audits[1]['initial_trainable_sha256']
    assert all(a['base_revision']==config['base_revision'] for a in audits)
    assert ckpt['config']['data_signature']==config['data_signature']
    proof={'status':'passed','suite':suite,'checkpoint':str(path),'sha256':sha(path),
           'step':expected,'epoch':ckpt['epoch'],'optimizer_scheduler_rng_verified':True,
           'fresh_initialization_verified_both_ranks':True,'finished':time.time(),'trial_only':smoke}
    save(BASE/('smoke' if smoke else suite)/'checkpoint-verification.json',proof)
    return proof


def resume_configuration(suite, gpus):
    """Only the explicit, fully audited Count pause can resume automatically."""
    import torch
    from .followup_schedule import COUNT_CHECKPOINT, COUNT_SHA
    assert suite == 'Count' and gpus == [2,3]
    authorization = json.loads((BASE/suite/'resume-authorization.json').read_text())
    assert authorization['status'] == 'authorized_after_group1_official_audit'
    assert authorization['gpus'] == gpus and authorization['step'] == 1200
    assert authorization['checkpoint'] == str(COUNT_CHECKPOINT)
    assert authorization['checkpoint_sha256'] == COUNT_SHA == sha(COUNT_CHECKPOINT)
    assert authorization['handoff_sha256'] == sha(BASE/'evaluation/group1-official-handoff.json')
    path = BASE/suite/'training.config.json'
    assert authorization['config_sha256'] == sha(path)
    config = json.loads(path.read_text())
    checkpoint = torch.load(COUNT_CHECKPOINT, map_location='cpu', weights_only=True)
    assert checkpoint['format'] == 'mem-lite-streamforest-v1' and checkpoint['step'] == 1200
    assert checkpoint['optimizer']['state'] and len(checkpoint['rng_by_rank']) == 2
    assert checkpoint['scheduler']['last_epoch'] == 1200
    for key in ('data_signature','history_source','model','run_protocol','batch_size',
                'gradient_accumulation_steps','learning_rate','epochs','seed','output_dir'):
        assert checkpoint['config'][key] == config[key], key
    assert config['epochs'] == 1 and config['batch_size'] == 1 and config['gradient_accumulation_steps'] == 9
    for source in config['sample_files']:
        directory = Path(source); proof = json.loads((directory/'cache-verification.json').read_text())
        assert sha(directory/'samples.jsonl') == proof['samples_sha256']
    del checkpoint
    return path, config, str(COUNT_CHECKPOINT)


def run(adopt=None):
    lock=(BASE/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=BASE/'controller.pid'
    if prior.exists() and alive(int(prior.read_text())):
        raise RuntimeError('Existing live controller; do not launch twice')
    prior.write_text(str(os.getpid()))
    save(BASE/'controller-capabilities.json', {'pid':os.getpid(),'verified_count_resume':True,
         'state_acceleration_maintenance':True,
         'code_sha256':sha(__file__),'at':time.time()})
    children={}; workers=json.loads((BASE/'cache-worker-pids.json').read_text())
    groups=[{'gpus':[0,1],'suite':None,'phase':'cache'}, {'gpus':[2,3],'suite':None,'phase':'cache'}]
    completed={}; failed={}; started=time.time()
    paused={}
    if adopt:
        handoff=json.loads(Path(adopt).read_text())
        assert not matches(handoff['old_controller']), 'Old controller still alive'
        assert time.time()-handoff['created']<300, 'Stale controller handoff'
        groups=handoff['groups'];started=handoff['started']
        completed=handoff['completed'];failed=handoff['failed'];paused=handoff.get('paused',{})
        for group in groups:
            if group['phase']=='training':
                record=handoff['training'][group['suite']]
                assert sha(BASE/group['suite']/'training.config.json')==record['config_sha256']
                children[group['suite']]=AdoptedTraining(record)
                group['config']=record['config']
        save(BASE/'controller-adoption.json',{'status':'passed','pid':os.getpid(),
             'handoff':str(adopt),'adopted_training_pids':{s:p.pid for s,p in children.items()},'at':time.time()})
    for suite in SUITES:
        p=BASE/suite/'checkpoint-verification.json'
        if p.exists():
            proof=json.loads(p.read_text())
            assert proof['status']=='passed' and sha(proof['checkpoint'])==proof['sha256']
            completed[suite]=proof
    save(BASE/'launch-authorization.json', {'user_authorized':True,
        'scope':'four independent one-epoch FoREST suite training then all five confirmed evaluation groups',
        'recover_missing_pour_pt_authorized':True,'notify_before_evaluation':'user already notified of automatic train-to-evaluation handoff',
        'timestamp':started})
    while True:
        db=connect()
        stats=[dict(x) for x in db.execute('SELECT suite,split,status,count(*) AS episodes,sum(samples) AS observations FROM jobs GROUP BY suite,split,status')]
        ready={s for s in SUITES if db.execute('SELECT count(*) FROM jobs WHERE suite=? AND split="train" AND status="completed"',(s,)).fetchone()[0]==COUNTS[s]}
        pending_cache=db.execute('SELECT count(*) FROM jobs WHERE status="pending"').fetchone()[0]
        priority=policy(BASE)
        for group in groups:
            suite=group['suite'];gpus=group['gpus']
            if suite == 'State':
                from .state_acceleration import request as acceleration_request, validate_resume, mark_resumed
                maintenance = acceleration_request(BASE)
                if maintenance and group['phase'] == 'training' and children[suite].poll() is not None:
                    group.update(phase='training_maintenance')
                    paused[suite] = {'reason':'user_authorized_acceleration_validation',
                                     'checkpoint':maintenance['checkpoint'], 'at':time.time()}
                if group['phase'] == 'training_maintenance':
                    if maintenance and maintenance['status'] == 'resume_requested':
                        path,config,resume,python = validate_resume(BASE,maintenance)
                        resume_env=env(gpus)
                        if maintenance.get('native_gated_norm') and maintenance['backend']=='accelerated':
                            resume_env['FOREST_NATIVE_GATED_NORM']='1'
                        proc=spawn([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                                    '-m','mem_lite.train_stage3','--config',str(path),'--resume',resume],
                                   BASE/suite/'train.log',resume_env)
                        children[suite]=proc;group.update(phase='training',pid=proc.pid,config=config)
                        paused.pop(suite,None)
                        save(BASE/suite/'acceleration-resume-launch.json',{'pid':proc.pid,'gpus':gpus,
                             'config_sha256':sha(path),'started':time.time(),'resume':resume,'python':python})
                        mark_resumed(BASE,maintenance,proc.pid)
                    continue
            if group['phase']=='training':
                proc=children[suite]
                if proc.poll() is None:
                    continue
                if suite in priority.get('paused_suites',[]):
                    paused[suite]={'reason':'user_requested_group1_priority','checkpoint':str(Path(group['config']['output_dir'])/'last.pt'),
                                   'stopped_pid':group['pid'],'at':time.time()}
                elif proc.returncode==0:
                    try:
                        completed[suite]=verify_training(suite,group['config'])
                    except Exception:
                        failed[suite]=traceback.format_exc()
                else:
                    failed[suite]=f'training exit {proc.returncode}; inspect {BASE/suite/"train.log"}'
                group.update(phase='cache',suite=None)
                for gpu in gpus:(BASE/f'drain-gpu-{gpu}').unlink(missing_ok=True)
            if any(g in priority.get('reserved_gpus',[]) for g in gpus):
                group.update(phase='evaluation_reserved',suite=None)
                for gpu in gpus:(BASE/f'drain-gpu-{gpu}').touch()
                continue
            if group['phase']=='evaluation_reserved':group.update(phase='cache',suite=None)
            if group['phase']=='cache':
                busy={g['suite'] for g in groups}
                choice=next((s for s in SUITES if s in ready and s not in completed and s not in failed and s not in busy and s not in priority.get('paused_suites',[])
                    and not (s=='Count' and (BASE/s/'resume-authorization.json').exists() and gpus!=[2,3])),None)
                if choice and (BASE/'smoke/checkpoint-verification.json').exists():
                    group.update(phase='draining',suite=choice)
                    for gpu in gpus:(BASE/f'drain-gpu-{gpu}').touch()
            if group['phase']=='draining':
                if any(alive(w['pid']) for w in workers if w['gpu'] in gpus):
                    continue
                # Any unrelated compute process blocks allocation; never kill it.
                gpu_uuids={int(line.split(',')[0]):line.split(',')[1].strip() for line in subprocess.check_output(
                    ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
                used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
                if any(gpu_uuids[g] in used for g in gpus):
                    continue
                suite=group['suite']
                try:
                    resume=None
                    if suite=='Count' and (BASE/suite/'resume-authorization.json').exists():
                        path,config,resume=resume_configuration(suite,gpus)
                    else:
                        path,config=training_config(suite,gpus)
                        if Path(config['output_dir']).exists():
                            raise RuntimeError('Existing unverified training output requires checkpoint inspection; refusing fresh overwrite')
                    proc=spawn([PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                                '-m','mem_lite.train_stage3','--config',str(path)]+(['--resume',resume] if resume else []),BASE/suite/'train.log',env(gpus))
                    children[suite]=proc;group.update(phase='training',pid=proc.pid,config=config)
                    paused.pop(suite,None)
                    save(BASE/suite/('resume-launch.json' if resume else 'launch.json'),{'pid':proc.pid,'gpus':gpus,'config_sha256':sha(path),'started':time.time(),'resume':resume})
                except Exception:
                    failed[suite]=traceback.format_exc();group.update(phase='cache',suite=None)
                    for gpu in gpus:(BASE/f'drain-gpu-{gpu}').unlink(missing_ok=True)
            if group['phase']=='cache' and pending_cache:
                for gpu in gpus:
                    if (BASE/f'drain-gpu-{gpu}').exists():
                        continue
                    living=sum(alive(w['pid']) for w in workers if w['gpu']==gpu)
                    for n in range(living,CACHE_WORKERS_PER_GPU):
                        # Failed jobs stay failed for audit; replacement workers claim only pending jobs.
                        wid=f'g{gpu}-r{time.time_ns()}'
                        proc=spawn([PYTHON,'-m','experiments.prm_five_groups_20261114.prepare_cache','worker',
                                    '--worker-id',wid,'--gpu',str(gpu)],BASE/f'cache-worker-{wid}.log',env([gpu]))
                        workers.append({'gpu':gpu,'worker':wid,'pid':proc.pid})
                    save(BASE/'cache-worker-pids.json',workers)
        state={'status':'training_and_preparation','controller_pid':os.getpid(),'started':started,'updated':time.time(),
               'resource_policy':{'cache_workers_per_gpu':CACHE_WORKERS_PER_GPU,'training_cuda_allocator_fraction':0.70,
                                  'training_gradient_checkpointing':True,'stability_over_maximum_throughput':True},
               'groups':[{k:v for k,v in g.items() if k!='config'} for g in groups],
               'cache':stats,'training_completed':completed,'training_failed':failed,'training_paused':paused,'evaluation_started':False}
        save(BASE/'status.json',state)
        if len(completed)==4:
            for gpu in range(4):(BASE/f'drain-gpu-{gpu}').touch()
            if any(alive(w['pid']) for w in workers):
                time.sleep(5);continue
            state.update(status='launching_evaluations',evaluation_started=True)
            save(BASE/'status.json',state)
            command=[PYTHON,'-m','experiments.prm_five_groups_20261114.eval_dispatch','run']
            proc=spawn(command,BASE/'evaluation-controller.log',env([0,1,2,3]))
            save(BASE/'evaluation-controller.json',{'pid':proc.pid,'started':time.time()})
            rc=proc.wait()
            state.update(status='completed' if rc==0 else 'evaluation_failed',evaluation_exit=rc,updated=time.time())
            save(BASE/'status.json',state)
            return
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--adopt');args=parser.parse_args()
    run(args.adopt)
