"""Explicit user GPU reservations and identity-checked CPU controller handoff."""
import json
from pathlib import Path
import re

from .gpu_ownership import process_identity


def policy(base):
    path = base / 'evaluation-priority.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get('status') != 'active':
        return {}
    assert value.get('user_authorized') is True
    assert value.get('reserved_gpus') == [2, 3]
    assert value.get('paused_suites') == ['Count']
    return value


def reserved(base, gpu):
    return gpu in policy(base).get('reserved_gpus', [])


def matches(record, proc=Path('/proc')):
    try:
        current = process_identity(record['pid'], proc)
        return current['start_ticks'] == record['start_ticks']
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


class AdoptedTraining:
    """Never launch a replacement for an adopted live torchrun or its ranks."""
    def __init__(self, record):
        self.record = record
        self.pid = record['parent']['pid']
        self.returncode = None
        assert all(matches(p) for p in [record['parent'], *record['ranks']])

    def poll(self):
        if any(matches(p) for p in [self.record['parent'], *self.record['ranks']]):
            return None
        # Exit code is unavailable after reparenting. A putative success still
        # must pass the controller's full final-checkpoint verification.
        final = Path(self.record['config']['output_dir']) / 'epoch-01.pt'
        self.returncode = 0 if final.is_file() else -1
        return self.returncode


def robometer_identity(job, gpu, minor, proc=Path('/proc')):
    ident = process_identity(job['pid'], proc)
    root = proc / str(job['pid'])
    cmd = (root / 'cmdline').read_bytes().split(b'\0')
    assert b'experiments.prm_five_groups_20261114.robometer_eval' in cmd
    assert cmd[cmd.index(b'--shard') + 1] == str(gpu).encode()
    env = dict(x.split(b'=', 1) for x in (root / 'environ').read_bytes().split(b'\0') if b'=' in x)
    assert env[b'CUDA_VISIBLE_DEVICES'] == str(gpu).encode()
    mapped = {int(m.group(1)) for line in (root / 'maps').read_text().splitlines()
              if (m := re.search(r'/dev/nvidia(\d+)$', line))}
    assert mapped == {minor}
    return {'pid': ident['pid'], 'start_ticks': ident['start_ticks'], 'gpu': gpu, 'minor': minor}


class ReservedEvaluationOwnership:
    """Prove a surviving Robometer CUDA context before adding one baseline.

    Requires a singleton NVML context, the configured live local worker/device,
    and newly completed predictions over 60 continuous seconds. Unknown extra
    processes or a changed worker invalidate admission; host PIDs stay opaque.
    """
    def __init__(self):
        self.samples = {}
        self.status = {}

    def observe(self, gpu, identity, job, host_pids, now):
        if len(host_pids) != 1 or job.get('status') != 'running' or now-job.get('updated', 0) > 30:
            self.samples.pop(gpu, None)
            self.status[gpu] = {'status': 'blocked_ambiguous_or_stale_worker'}
            return False
        previous = self.samples.get(gpu)
        if (not previous or previous['identity'] != identity or previous['host_pid'] != host_pids[0]
                or not 0 <= now-previous['last_seen'] <= 30):
            previous = dict(identity=identity, host_pid=host_pids[0], started=now,
                            initial_samples=job.get('completed_samples', 0), last_seen=now)
            self.samples[gpu] = previous
        previous['last_seen'] = now
        ready = now-previous['started'] >= 60 and job.get('completed_samples', 0) > previous['initial_samples']
        self.status[gpu] = dict(previous, status='verified_robometer_singleton' if ready else 'waiting_for_fresh_predictions_and_60_seconds')
        return ready

    def permits(self, base, gpu, job, host_pids, minor, now):
        if not reserved(base, gpu):
            return False
        if not host_pids:
            self.samples.pop(gpu, None)
            return job.get('status') in {'completed', 'completed_with_input_blockers'} and not matches_job(job)
        try:
            identity = robometer_identity(job, gpu, minor)
            return self.observe(gpu, identity, job, host_pids, now)
        except (AssertionError, FileNotFoundError, KeyError, ValueError, ProcessLookupError) as exc:
            self.samples.pop(gpu, None)
            self.status[gpu] = {'status': 'unverified', 'reason': str(exc)}
            return False


def matches_job(job):
    try:
        process_identity(job['pid'])
        return True
    except (FileNotFoundError, KeyError, ValueError, ProcessLookupError):
        return False
