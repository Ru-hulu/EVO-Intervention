"""Identity-checked adoption of existing evaluation processes; no GPU signals."""
import json
from pathlib import Path
import time

from .evaluation_priority import matches
from .gpu_ownership import process_identity


def descendants(pid, proc=Path('/proc')):
    result = []
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            ident = process_identity(current, proc)
            result.append(ident)
            pending += [int(p) for p in (proc/str(current)/'task'/str(current)/'children').read_text().split()]
        except (FileNotFoundError, ValueError, ProcessLookupError):
            continue
    return result


def verify_command(receipt, proc=Path('/proc')):
    pid = receipt['pid']; root = proc/str(pid)
    command = [s.decode() for s in (root/'cmdline').read_bytes().split(b'\0') if s]
    index = command.index('-m')
    assert command[index+1:] == [receipt['module'], *receipt['args'], '--memory-fraction', str(receipt['memory_fraction'])]
    gpu = [part.split(b'=',1)[1].decode() for part in (root/'environ').read_bytes().split(b'\0')
           if part.startswith(b'CUDA_VISIBLE_DEVICES=')]
    assert gpu == [str(receipt['gpu'])]
    return process_identity(pid, proc)


class AdoptedEvaluation:
    def __init__(self, record, jobs, proc=Path('/proc')):
        self.record = record; self.pid = record['receipt']['pid']; self.returncode = None
        self.jobs = Path(jobs); self.proc = proc
        if matches(record['identity'], proc):
            assert verify_command(record['receipt'], proc)['start_ticks'] == record['identity']['start_ticks']
        else:
            state = json.loads((self.jobs/f"{record['receipt']['name']}.json").read_text())
            assert state['status'] in ('completed','completed_with_input_blockers','paused_by_resource_guard','failed')

    def poll(self):
        if matches(self.record['identity'], self.proc):
            known={(r['pid'],r['start_ticks']) for r in self.record['process_tree']}
            self.record['process_tree'] += [r for r in descendants(self.pid,self.proc)
                                           if (r['pid'],r['start_ticks']) not in known]
        if any(matches(identity, self.proc) for identity in self.record['process_tree']):
            return None
        path = self.jobs/f"{self.record['receipt']['name']}.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        self.returncode = 0 if state.get('status') in ('completed','completed_with_input_blockers','paused_by_resource_guard') else -1
        return self.returncode


def adopt(path, jobs, planned):
    handoff = json.loads(Path(path).read_text())
    assert not matches(handoff['old_dispatcher']) and time.time()-handoff['created'] < 300
    known = {job['name']: job for job in planned}; active = {}
    for record in handoff['workers']:
        receipt = record['receipt']; job = known[receipt['name']]
        assert all(receipt[k] == job[k] for k in job)
        assert receipt['gpu'] not in active
        worker = AdoptedEvaluation(record, jobs)
        if worker.poll() is None:
            active[receipt['gpu']] = (job, worker, receipt)
    return active, handoff
