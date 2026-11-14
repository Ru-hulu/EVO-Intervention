"""One-time CPU-controller handoff; never send a signal to a GPU worker."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .prepare_cache import BASE, save, sha
from .eval_common import EVAL
from .eval_dispatch import gpu_minors, gpu_processes
from .gpu_ownership import process_identity, training_identity
from .evaluation_priority import matches
from .eval_handoff import descendants, verify_command

CODE=Path(__file__).parents[2]
PYTHON='/cache/cvpzhr/envs/forest-labels/bin/python'


def cpu_identity(pid,module,proc=Path('/proc')):
    identity=process_identity(pid,proc); root=proc/str(pid)
    command=[s.decode() for s in (root/'cmdline').read_bytes().split(b'\0') if s]
    assert command[command.index('-m')+1]==module
    assert not any(re.search(r'/dev/nvidia\d+$',line) for line in (root/'maps').read_text().splitlines()), 'Refuse to signal a GPU process'
    return identity


def wait_until(predicate,seconds=25):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.2)
    raise RuntimeError('Controller handoff condition timed out; inspect saved handoff before resuming')


def upgrade():
    lock=(EVAL/'controller-upgrade.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checks=json.loads((BASE/'source-checksums.json').read_text())
    assert checks['status']=='passed'
    for name in ('controller.py','eval_dispatch.py','eval_handoff.py','upgrade_controllers.py'):
        relative='experiments/prm_five_groups_20261114/'+name
        assert sha(CODE/relative)==checks['files'][relative]
    assert not (EVAL/'dispatcher-reload-when-idle.json').exists()
    controller_pid=int((BASE/'controller.pid').read_text()); dispatcher_pid=int((EVAL/'dispatcher.pid').read_text())
    controllers=[cpu_identity(controller_pid,'experiments.prm_five_groups_20261114.controller'),
                 cpu_identity(dispatcher_pid,'experiments.prm_five_groups_20261114.eval_dispatch')]
    assert controller_pid!=dispatcher_pid
    folder=EVAL/'controller-upgrade-history'/str(time.time_ns());folder.mkdir(parents=True)
    stopped=[];signals=[]
    def send(record,sig):
        assert matches(record)
        assert record in controllers
        os.kill(record['pid'],sig);signals.append(dict(pid=record['pid'],signal=signal.Signals(sig).name))
    try:
        for record in controllers:
            send(record,signal.SIGSTOP);stopped.append(record)
        training=json.loads((BASE/'status.json').read_text())
        dispatch=json.loads((EVAL/'dispatcher-status.json').read_text())
        assert time.time()-training['updated']<30 and time.time()-dispatch['updated']<30
        assert not dispatch.get('cpu_action'), 'Wait for the existing CPU preparation/report to finish'
        assert training['controller_pid']==controller_pid and dispatch['pid']==dispatcher_pid
        assert all(g['phase'] in ('training','evaluation_reserved') for g in training['groups'])
        workers=[]
        for receipt in dispatch['active'].values():
            ident=verify_command(receipt)
            workers.append(dict(receipt=receipt,identity=ident,process_tree=descendants(receipt['pid'])))
        children={int(x) for x in Path(f'/proc/{dispatcher_pid}/task/{dispatcher_pid}/children').read_text().split()}
        assert children=={r['identity']['pid'] for r in workers}, 'Dispatcher launch/exit race; no handoff performed'
        minors=gpu_minors(); records={}
        for group in training['groups']:
            if group['phase']=='training':
                record=training_identity(group,BASE,minors)
                path=BASE/group['suite']/'training.config.json'
                record.update(config=json.loads(path.read_text()),config_sha256=sha(path));records[group['suite']]=record
        gpu_before=gpu_processes()
        train_path=folder/'training-handoff.json';dispatch_path=folder/'evaluation-handoff.json'
        save(train_path,dict(old_controller=controllers[0],created=time.time(),groups=training['groups'],
            started=training['started'],completed=training['training_completed'],failed=training['training_failed'],
            paused=training.get('training_paused',{}),training=records))
        save(dispatch_path,dict(old_dispatcher=controllers[1],created=time.time(),workers=workers,
            initial_oom=dispatch['resource_sample']['oom_kill']))
        save(folder/'before.json',dict(training=training,dispatcher=dispatch,gpu_processes=gpu_before,controllers=controllers))
        for record in controllers:
            send(record,signal.SIGTERM);send(record,signal.SIGCONT);stopped.remove(record)
        wait_until(lambda:not any(matches(r) for r in controllers),10)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        environment.pop('PYTHONPATH',None)
        for module,args,log in (
            ('controller',['--adopt',str(train_path)],BASE/'controller.log'),
            ('eval_dispatch',['run','--adopt',str(dispatch_path)],EVAL/'dispatcher.log')):
            with log.open('a') as f:
                process=subprocess.Popen([PYTHON,'-m','experiments.prm_five_groups_20261114.'+module,*args],
                    cwd=CODE,env=environment,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            expected=process.pid
            statepath=BASE/'status.json' if module=='controller' else EVAL/'dispatcher-status.json'
            field='controller_pid' if module=='controller' else 'pid'
            wait_until(lambda:json.loads(statepath.read_text()).get(field)==expected)
        live=[r['identity'] for r in workers]+[p for r in records.values() for p in [r['parent'],*r['ranks']]]
        assert all(matches(identity) for identity in live), 'A worker changed during handoff; inspect terminal state'
        save(folder/'verification.json',dict(status='passed',at=time.time(),gpu_processes_before=gpu_before,
            gpu_processes_after=gpu_processes(),preserved_worker_identities=live,signals=signals,
            gpu_workers_signaled=False,new_controller_pid=int((BASE/'controller.pid').read_text()),
            new_dispatcher_pid=int((EVAL/'dispatcher.pid').read_text())))
        print(folder)
    finally:
        for record in stopped:
            if matches(record):os.kill(record['pid'],signal.SIGCONT)
        save(folder/'signal-audit.json',dict(signals=signals,gpu_workers_signaled=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();upgrade()
