"""Independent trajectory replicas; bounded resources; durable raw prefixes."""
import argparse
import fcntl
import os
import time
import traceback
from .common import *
from experiments.soc_disrupted_eval_20261115.guard import process_identity,alive,gpu_processes,host_sample
from experiments.recover_validation_20261114.common import safe
from experiments.soc_disrupted_eval_20261115.engine import SafetyPause
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics


def contexts(gpu,controller):
    result=set()
    for p in ROOT.glob('jobs/*/*/context.json'):
        d=read(p)
        if d['gpu']==gpu and d['controller']==controller and alive(d['process']):result.add(d['host_pid'])
    return result


class Guard:
    def __init__(self,ticket,folder):
        self.t=ticket;self.folder=folder;self.last=0;self.gpu=ticket['gpu']
        assert process_identity(os.getppid())==ticket['controller'] and alive(ticket['controller'])
        assert os.environ['CUDA_VISIBLE_DEVICES']==str(self.gpu)
        self.lock=(ROOT/f'admission-gpu{self.gpu}.lock').open('a')

    def admit(self,torch):
        fcntl.flock(self.lock,fcntl.LOCK_EX)
        try:
            before=set(gpu_processes()[self.gpu])
            if not before<=contexts(self.gpu,self.t['controller']):raise SafetyPause('Unknown GPU process')
            if not safe(host_sample(),self.gpu,self.t['initial_oom']):raise SafetyPause('Resource pressure')
            torch.cuda.init();torch.cuda.set_per_process_memory_fraction(self.t['allocator_fraction'])
            marker=torch.zeros(1,device='cuda');torch.cuda.synchronize()
            new=set(gpu_processes()[self.gpu])-before
            if len(new)!=1:raise SafetyPause('Ambiguous GPU context admission')
            save(self.folder/'context.json',dict(gpu=self.gpu,host_pid=new.pop(),process=process_identity(os.getpid()),controller=self.t['controller']))
            del marker
        finally:fcntl.flock(self.lock,fcntl.LOCK_UN)

    def check(self):
        if time.time()-self.last<5:return
        self.last=time.time();s=read(ROOT/'status.json')
        if not alive(self.t['controller']) or s['controller']!=self.t['controller'] or time.time()-s['updated']>180:
            raise SafetyPause('Controller stopped')
        if (ROOT/'pause-all').exists() or not safe(host_sample(),self.gpu,self.t['initial_oom']):raise SafetyPause('Resource protection')
        try:fcntl.flock(self.lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:return
        try:
            if not set(gpu_processes()[self.gpu])<=contexts(self.gpu,self.t['controller']):raise SafetyPause('Unknown GPU process appeared')
        finally:fcntl.flock(self.lock,fcntl.LOCK_UN)


def main(ticket_path):
    ticket=read(ticket_path);task=ticket['task'];shard=ticket['shard'];folder=ROOT/'jobs'/task/f'shard{shard}'
    folder.mkdir(parents=True,exist_ok=True);lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='starting',task=task,shard=shard,gpu=ticket['gpu'],process=process_identity(os.getpid()))
    def update(**kw):state.update(kw,updated=time.time());save(folder/'status.json',state)
    update()
    try:
        inputs=read(ROOT/'inputs.json');assert sha(ROOT/'inputs.json')==ticket['inputs_sha256'];job=inputs['tasks'][task]
        for p,h in read(ROOT/'source-checksums.json')['files'].items():assert sha(ROOT/'code'/p)==h,p
        guard=Guard(ticket,folder)
        import torch,random,numpy as np
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2);guard.admit(torch);torch.cuda.manual_seed_all(42)
        from .backend import Backend
        update(status='loading_model');backend=Backend(inputs['registry'][task]);save(folder/'initialization.json',backend.audit)
        selected={r['key']:r for r in inputs['episodes']};keys=job['shards'][shard]
        update(status='smoke_testing');check=backend.smoke(model_input(selected[keys[0]]));guard.check();save(folder/'smoke.json',check)
        completed=0;completed_obs=0;started=time.time()
        for key in keys:
            row=selected[key];path=result_path(job['model'],key);partial=path.with_suffix('.partial.json');expected=identity(inputs,row,job['model'])
            if path.exists():
                validate(read(path),row,expected,True);completed+=1;completed_obs+=len(row['frame_indices']);continue
            backend.begin(model_input(row))
            record=read(partial) if partial.exists() else dict(identity=expected,predictions=[],raw_responses=[],input_audits=[],
                frame_indices=[],timestamps=[],target_pt=[],source_video_sha256=row['video_hashes'],progress_sha256=row['progress_sha256'],
                initialization=backend.audit,initialization_sha256=digest(backend.audit),status='partial')
            start=validate(record,row,expected);assert record['initialization']==backend.audit
            update(status='running',episode=key,completed_episodes=completed,total_episodes=len(keys),observations_in_episode=start)
            saved=0
            try:
                for i in range(start,len(row['frame_indices'])):
                    guard.check();v=backend.predict(i)
                    record['predictions'].append(v['prediction']);record['raw_responses'].append(v['raw_response']);record['input_audits'].append(v['input_audit'])
                    for k in ('frame_indices','timestamps','target_pt'):record[k].append(row[k][i])
                    if time.time()-saved>=2:
                        save(partial,record);saved=time.time();update(observations_in_episode=i+1,completed_observations=completed_obs+i+1,
                            observations_per_second=(completed_obs+i+1)/(time.time()-started))
            finally:save(partial,record)
            record.update(status='completed',metrics=trajectory_metrics(record['predictions'],row['target_pt'],row['timestamps']))
            validate(record,row,expected,True);save(path,record);partial.unlink()
            completed+=1;completed_obs+=len(row['frame_indices'])
            if completed==1:save(folder/'first-real-episode.json',dict(status='passed',result=str(path),sha256=sha(path),invalid=record['metrics']['invalid_predictions']))
        backend.drain();update(status='completed',completed_episodes=completed,completed_observations=completed_obs);return 0
    except SafetyPause as e:update(status='paused',reason=str(e));return 75
    except BaseException:update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args();raise SystemExit(main(a.ticket))
