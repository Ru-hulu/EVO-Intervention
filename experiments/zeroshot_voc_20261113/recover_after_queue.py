"""One-shot recovery of the restored Robometer dependency after healthy jobs exit."""
import fcntl
import subprocess
import time
from common import *


def live_command(pid, script):
    try:
        proc=Path('/proc')/str(pid)
        if (proc/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return False
        return script in (proc/'cmdline').read_bytes().decode(errors='replace')
    except FileNotFoundError:
        return False


def main():
    lock=(ROOT/'recovery.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=ROOT/'recovery.json'
    if state_path.exists() and read(state_path).get('status') in ['relaunched','already_completed']:
        return
    old_pid=int((ROOT/'controller.pid').read_text())
    write(state_path,dict(status='waiting_for_healthy_queue',controller_pid=old_pid,
                         repair='restored original runtime/robometer_training/uniform.py',updated_at=now()))
    while live_command(old_pid,str(ROOT/'pipeline.py')):
        time.sleep(15)
    # A different controller must never be superseded or duplicated.
    current_pid=int((ROOT/'controller.pid').read_text())
    if current_pid!=old_pid and live_command(current_pid,str(ROOT/'pipeline.py')):
        write(state_path,dict(status='another_controller_active',controller_pid=current_pid,updated_at=now()))
        return
    for path in (ROOT/'jobs').glob('*.json'):
        if path.name.endswith('-audit.json'):continue
        job=read(path)
        if job.get('pid') and live_command(job['pid'],str(ROOT/'worker.py')):
            raise RuntimeError('Old worker is still alive; recovery refused: '+str(path))
    failed=ROOT/'jobs/robometer-4b-1.json'
    if failed.exists() and read(failed)['status']=='failed':
        job=read(failed)
        assert "No module named 'robometer_training.uniform'" in job['error']
        saved=ROOT/'recovery-history/robometer-4b-1-missing-uniform.json'
        assert not saved.exists()
        saved.parent.mkdir(exist_ok=True)
        failed.replace(saved)
    elif all((ROOT/'jobs'/f'robometer-4b-{s}.json').exists() and
             read(ROOT/'jobs'/f'robometer-4b-{s}.json')['status']=='completed' for s in range(4)):
        write(state_path,dict(status='already_completed',updated_at=now()));return
    else:
        raise RuntimeError('Unexpected Robometer state; recovery refused')
    with (ROOT/'controller.log').open('ab') as log:
        proc=subprocess.Popen(['/cache/cvpzhr/envs/forest/bin/python','-u',str(ROOT/'pipeline.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'controller.pid').write_text(str(proc.pid)+'\n')
    write(state_path,dict(status='relaunched',controller_pid=proc.pid,previous_controller_pid=old_pid,
                         preserved_all_predictions=True,updated_at=now()))


if __name__=='__main__':main()
