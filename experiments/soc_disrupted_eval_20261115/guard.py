"""Idle-GPU admission only; never signals another task or guesses host PIDs."""
import os
import subprocess
import time
from pathlib import Path
from .common import ROOT, read
from .engine import SafetyPause
from experiments.recover_validation_20261114.common import host_sample, safe


def process_identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else dict(pid=int(pid), start_ticks=int(fields[19]))
    except (OSError, ValueError):
        return None


def alive(identity):
    return bool(identity) and process_identity(identity['pid']) == identity


def gpu_processes():
    def query(fields, kind):
        return subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
            '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    mapping = {u.strip():int(i) for i,u in (line.split(',') for line in query('index,uuid', 'gpu'))}
    result = {i:[] for i in mapping.values()}
    for line in query('gpu_uuid,pid', 'compute-apps'):
        uuid,pid = map(str.strip, line.split(','))
        result[mapping[uuid]].append(int(pid))
    return result


def idle_safe(sample, processes, gpu, initial_oom):
    return not processes[gpu] and safe(sample, gpu, initial_oom)


def validate_ticket(ticket, job_id, token, parent, now):
    assert ticket['token'] == token and ticket['job_id'] == job_id
    assert ticket['controller'] == parent and now-ticket['issued'] < 180
    assert ticket['inputs_sha256']


class WorkerGuard:
    def __init__(self, job_id, token):
        self.job_id = job_id
        self.ticket = read(ROOT/'tickets'/f'{job_id}.json')
        validate_ticket(self.ticket, job_id, token, process_identity(os.getppid()), time.time())
        assert alive(self.ticket['controller'])
        self.gpu = self.ticket['gpu']; self.last = 0
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(self.gpu)
        sample = host_sample()
        assert idle_safe(sample, gpu_processes(), self.gpu, self.ticket['initial_oom'])

    def check(self):
        now = time.time()
        if now-self.last < 5:
            return
        self.last = now
        if not alive(self.ticket['controller']):
            raise SafetyPause('Controller exited; preserve prefix and release this GPU')
        status = read(ROOT/'status.json')
        if status.get('controller') != self.ticket['controller'] or now-status.get('updated', 0)>180:
            raise SafetyPause('Controller heartbeat stale')
        if (ROOT/'pause-all').exists():
            raise SafetyPause('Cooperative pause requested')
        if not safe(host_sample(), self.gpu, self.ticket['initial_oom']):
            raise SafetyPause('Resource guard')
        if len(gpu_processes()[self.gpu])>1:
            raise SafetyPause('An additional GPU context appeared; no process ownership assumed')
