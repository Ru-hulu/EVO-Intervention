import fcntl
import os
import time
from .common import *
from experiments.soc_disrupted_eval_20261115.engine import SafetyPause
from experiments.soc_disrupted_eval_20261115.guard import process_identity, alive, gpu_processes, host_sample
from experiments.recover_validation_20261114.common import safe


def contexts(gpu):
    result=set()
    for p in ROOT.glob('tasks/*/workers/*/context.json'):
        d=read(p)
        if d['gpu']==gpu and alive(d['process']) and alive(d['manager']):result.add(d['host_pid'])
    return result


class Guard:
    def __init__(self,ticket,folder):
        self.ticket=ticket;self.folder=folder;self.gpu=ticket['gpu'];self.last=0
        assert alive(ticket['manager']) and ticket['manager']==process_identity(os.getppid())
        assert os.environ['CUDA_VISIBLE_DEVICES']==str(self.gpu)
        self.lock=(ROOT/f'admission-gpu{self.gpu}.lock').open('a')

    def admit(self,torch):
        fcntl.flock(self.lock,fcntl.LOCK_EX)
        try:
            before=set(gpu_processes()[self.gpu])
            if not before<=contexts(self.gpu):raise SafetyPause('Unrelated GPU context')
            if not safe(host_sample(),self.gpu,self.ticket['initial_oom']):raise SafetyPause('Resource pressure')
            torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.19)
            marker=torch.zeros(1,device='cuda');torch.cuda.synchronize()
            added=set(gpu_processes()[self.gpu])-before
            if len(added)!=1:raise SafetyPause('CUDA context identity is ambiguous')
            save(self.folder/'context.json',dict(gpu=self.gpu,host_pid=added.pop(),
                process=process_identity(os.getpid()),manager=self.ticket['manager'],registered=time.time()))
            del marker
        finally:fcntl.flock(self.lock,fcntl.LOCK_UN)

    def check(self):
        now=time.time()
        if now-self.last<5:return
        self.last=now
        manager=read(self.folder.parent.parent/'status.json')
        if not alive(self.ticket['manager']) or manager.get('manager')!=self.ticket['manager'] or now-manager['updated']>180:
            raise SafetyPause('Manager stopped; preserve prefix')
        if (ROOT/'pause-all').exists() or (self.folder.parent.parent/'abort').exists():
            raise SafetyPause('Cooperative stop requested')
        if not safe(host_sample(),self.gpu,self.ticket['initial_oom']):raise SafetyPause('Resource protection')
        try:fcntl.flock(self.lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:return
        try:
            if not set(gpu_processes()[self.gpu])<=contexts(self.gpu):raise SafetyPause('Unrelated GPU context appeared')
        finally:fcntl.flock(self.lock,fcntl.LOCK_UN)


def wait_file(path,guard,timeout=1800):
    start=time.time()
    while not path.exists():
        guard.check()
        assert time.time()-start<timeout, f'Timed out waiting for {path}'
        time.sleep(.25)
    return read(path)
