"""One authorized State checkpoint transition, preserving every evaluation worker."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from .prepare_cache import BASE, save, sha
from .prepare_state_acceleration import AUDIT, ENV, ORIGINAL
from .gpu_ownership import process_identity, training_identity
from .eval_dispatch import gpu_minors, gpu_processes
from .eval_handoff import verify_command, descendants
from .evaluation_priority import matches
from .overlap_guard import resources, pause_reasons
from .controller import CODE


def wait_for(predicate, timeout):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        result=predicate()
        if result:return result
        time.sleep(2)
    raise TimeoutError('State acceleration transition timed out; inspect its saved status')


def main():
    lock=(AUDIT/'transition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (AUDIT/'completion.json').exists():raise RuntimeError('Transition already completed')
    (AUDIT/'supervisor.pid').write_text(str(os.getpid()))
    environment=json.loads((AUDIT/'environment.json').read_text());assert environment['status']=='passed'
    training=json.loads((BASE/'status.json').read_text());assert time.time()-training['updated']<30
    capability=json.loads((BASE/'controller-capabilities.json').read_text())
    assert capability.get('state_acceleration_maintenance') is True
    assert capability['pid']==training['controller_pid']
    assert capability['code_sha256']==sha(CODE/'experiments/prm_five_groups_20261114/controller.py')
    group,=[g for g in training['groups'] if g['suite']=='State' and g['phase']=='training']
    assert group['gpus']==[0,1]
    identity=training_identity(group,BASE,gpu_minors())
    original_tree=descendants(group['pid'])
    dispatch=json.loads((BASE/'evaluation/dispatcher-status.json').read_text())
    assert dispatch['boot_code_sha256']==sha(CODE/'experiments/prm_five_groups_20261114/eval_dispatch.py')
    evals=[dict(receipt=r,identity=verify_command(r)) for r in dispatch['active'].values()]
    assert {0,1}<={r['receipt']['gpu'] for r in evals}<={0,1,2,3}
    sample=resources();initial_oom=sample['oom_kill']
    assert not any(pause_reasons(sample,g,initial_oom) for g in (0,1))
    config_path=BASE/'State/training.config.json'
    config=json.loads(config_path.read_text()); output=Path(config['output_dir'])
    checkpoint=AUDIT/f'state-resume-{time.time_ns()}.pt'
    assert not checkpoint.exists(), 'Do not overwrite an existing resume checkpoint'
    shutil.copy2(output/'last.pt',checkpoint)
    import torch
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert ckpt['format']=='mem-lite-streamforest-v1' and ckpt['epoch']==0
    assert ckpt['optimizer']['state'] and len(ckpt['rng_by_rank'])==2
    assert ckpt['scheduler']['last_epoch']==ckpt['step']
    assert ckpt['next_batch']==ckpt['step']*9
    for k,v in config.items():assert ckpt['config'][k]==v,k
    checkpoint_step=ckpt['step'];del ckpt
    value=dict(status='testing',user_authorized=True,suite='State',gpus=[0,1],
        supervisor=process_identity(os.getpid()),original_training=identity,
        config_sha256=sha(config_path),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        step=checkpoint_step,created=time.time())
    def progress(phase,**extra):
        save(AUDIT/'status.json',dict(phase=phase,updated=time.time(),supervisor_pid=os.getpid(),**extra))
        print(phase,flush=True)
    save(AUDIT/'before.json',dict(training=training,evaluations=evals,resources=sample,
                                 gpu_processes=gpu_processes(),original_tree=original_tree))
    save(AUDIT/'request.json',value)
    progress('stopping_only_State_after_checkpoint_backup',step=checkpoint_step)
    assert all(matches(r) for r in [identity['parent'],*identity['ranks']])
    os.kill(identity['parent']['pid'],signal.SIGTERM)
    save(AUDIT/'signals.json',dict(signals=[{'pid':identity['parent']['pid'],'signal':'SIGTERM'}],
                                  evaluation_workers_signaled=False))
    wait_for(lambda:not any(matches(r) for r in original_tree),60)
    wait_for(lambda:any(g['suite']=='State' and g['phase']=='training_maintenance'
                        for g in json.loads((BASE/'status.json').read_text())['groups']),40)
    assert all(matches(r['identity']) for r in evals)
    assert all(len(gpu_processes()[g])==1 for g in (0,1)), 'Unexpected process after State exit'
    progress('validating_in_original_training_slots')
    validation=AUDIT/'probe';validation.mkdir(exist_ok=True)

    def evals_healthy():
        for r in evals:
            if not matches(r['identity']):
                job=json.loads((BASE/'evaluation/jobs'/f"{r['receipt']['name']}.json").read_text())
                assert job['status'] in ('completed','completed_with_input_blockers'), 'An evaluation stopped unexpectedly'

    def probe(python,backend,ddp=False):
        command=[str(python)]
        if ddp:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
        command+=['-m','experiments.prm_five_groups_20261114.acceleration_probe',
                  '--backend',backend,'--checkpoint',str(checkpoint),'--output',str(validation)]
        if ddp:command+=['--ddp-smoke']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1' if ddp else '0',OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
            TRITON_CACHE_DIR=str(AUDIT/'triton-cache'),MAX_JOBS='2')
        env.pop('PYTHONPATH',None)
        if backend=='accelerated':env['FOREST_NATIVE_GATED_NORM']='1'
        name=backend+('-ddp' if ddp else '')
        progress('gpu_probe_'+name)
        with (validation/f'{name}.log').open('a') as log:
            p=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.monotonic()+900
        tree={}
        try:
            while p.poll() is None:
                for r in descendants(p.pid):tree[(r['pid'],r['start_ticks'])]=r
                evals_healthy()
                current=resources()
                assert not any(pause_reasons(current,g,initial_oom) for g in (0,1)), 'Resource margin reduced'
                if time.monotonic()>until:raise TimeoutError('Bounded acceleration probe timed out')
                time.sleep(3)
            assert p.returncode==0, f'{name} probe exited {p.returncode}; see its log'
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
            wait_for(lambda:not any(matches(r) for r in tree.values()),30)

    backend='original'
    try:
        reference=AUDIT/'history/full-fusion/probe/original.pt'
        reference_receipt=AUDIT/'history/full-fusion/request.json'
        if reference.exists() and json.loads(reference_receipt.read_text())['checkpoint_sha256']==value['checkpoint_sha256']:
            shutil.copy2(reference,validation/'original.pt')
        else:
            probe(ORIGINAL/'bin/python','original')
        probe(ENV/'bin/python','accelerated')
        from .verify_acceleration import compare, LIMITS
        proof=compare(validation)
        assert proof['status']=='passed', 'Numerical or memory comparison failed; tolerances are not relaxed'
        probe(ENV/'bin/python','accelerated',ddp=True)
        ranks=[json.loads((validation/f'accelerated-ddp-rank{i}.json').read_text()) for i in (0,1)]
        assert all(r['status']=='passed' and r['world_size']==2 and r['checkpoint_step']==checkpoint_step for r in ranks)
        assert all(r['peak_allocated_gib']<=LIMITS['allocated_gib'] and r['peak_reserved_gib']<=LIMITS['reserved_gib'] for r in ranks)
        assert ranks[0]['initial_trainable_sha256']==ranks[1]['initial_trainable_sha256']
        evals_healthy()
        proof.update(two_rank_smoke_passed=True,checkpoint_sha256=value['checkpoint_sha256'],
                     config_sha256=value['config_sha256'],two_rank_smoke=ranks,native_gated_norm=True,at=time.time())
        save(AUDIT/'verification.json',proof);backend='accelerated'
    except Exception:
        save(AUDIT/'verification.json',dict(status='failed',error=traceback.format_exc(),
            checkpoint_sha256=value['checkpoint_sha256'],config_sha256=value['config_sha256'],at=time.time()))
        progress('validation_failed_restoring_original_backend')
    # Neither successful nor failed diagnostic weights are ever resumed.
    assert not any(matches(r) for r in original_tree)
    save(AUDIT/'request.json',dict(value,status='resume_requested',backend=backend,native_gated_norm=backend=='accelerated',
         reason='validation_passed' if backend=='accelerated' else
                'acceleration_validation_failed_restore_original_training'))
    progress('waiting_for_checkpoint_resume',backend=backend,step=checkpoint_step)
    def resumed():
        r=json.loads((AUDIT/'request.json').read_text())
        return r if r['status']=='resumed' else None
    resumed_value=wait_for(resumed,120)
    new_pid=resumed_value['resumed_pid']
    def advanced():
        assert Path(f'/proc/{new_pid}').exists(), 'Resumed training parent exited'
        lines=(output/'train.jsonl').read_text().splitlines()
        r=json.loads(lines[-1])
        return r if r['timestamp']>resumed_value['resumed_at'] and r['step']>checkpoint_step else None
    record=wait_for(advanced,1800)
    evals_healthy()
    state=json.loads((BASE/'status.json').read_text())
    new_group,=[g for g in state['groups'] if g['suite']=='State']
    new_identity=training_identity(new_group,BASE,gpu_minors())
    audits=[json.loads((output/f'initialization-rank{r}-resume.json').read_text()) for r in (0,1)]
    assert all(a['resume'] and a['world_size']==2 for a in audits)
    assert audits[0]['initial_trainable_sha256']==audits[1]['initial_trainable_sha256']
    if backend=='accelerated':
        assert audits[0]['initial_trainable_sha256']==ranks[0]['initial_trainable_sha256']
    assert sha(config_path)==value['config_sha256'] and resources()['oom_kill']==initial_oom
    save(AUDIT/'completion.json',dict(status='passed' if backend=='accelerated' else 'original_backend_restored',
        backend=backend,resumed_from_step=checkpoint_step,latest_training=record,new_training=new_identity,
        config_sha256=value['config_sha256'],checkpoint_sha256=value['checkpoint_sha256'],
        evaluation_workers_signaled=False,original_environment_unchanged=True,finished=time.time()))
    progress('completed',backend=backend,latest_step=record['step'])


if __name__=='__main__':
    try:main()
    except BaseException:
        save(AUDIT/'supervisor-error.json',dict(error=traceback.format_exc(),at=time.time()))
        raise
