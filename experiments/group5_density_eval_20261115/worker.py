"""One persistent replica: equivalence, four concurrency trials, then real work."""
import argparse
import os
import time
import traceback
from .common import *
from .guard import Guard, wait_file, SafetyPause, process_identity
from experiments.group5_memoryfps_20261115.protocol import model_episode, validate_record
from experiments.group5_throughput_20261115.worker import run_episode


def equivalent(reference, actual, tolerance):
    assert len(reference)==len(actual)
    errors=[]
    for x,y in zip(reference,actual):
        assert (x['prediction'] is None)==(y['prediction'] is None), 'Invalid-output mask changed'
        if x['prediction'] is not None:errors.append(abs(x['prediction']-y['prediction']))
        assert digest(x['input_audit'])==digest(y['input_audit']), 'Native input/history changed'
    assert errors and max(errors)<=tolerance, f'Prediction difference {max(errors,default=None)}'
    return max(errors)


def check_equivalence(b,rows,all_entries,guard):
    import torch
    trials=[]
    if b.name=='forest':
        row=max(rows,key=lambda r:r['frames']);e=model_episode(row,all_entries[5][row['key']])
        # Test a mature memory after the first sampling-density transition.
        boundary=next(i for i in range(1,len(e['images']))
                      if all_entries[5][row['key']]['segment_ids'][i]!=all_entries[5][row['key']]['segment_ids'][i-1])
        start=min(max(128,boundary-4),len(e['images'])-32)
        b.native.begin(e)
        for i in range(start):guard.check();b.native.rebuild_step(i)
        ref=[]
        for i in range(start,start+24):guard.check();ref.extend(b.native.predict_many([i]))
        b.begin(e)
        for i in range(start):guard.check();b.rebuild_step(i)
        got=[]
        for i in range(start,start+24):guard.check();got.extend(b.predict_many([i]))
        error=equivalent(ref,got,1e-6)
        b.begin(e)
        for i in range(start+12):guard.check();b.rebuild_step(i)
        replay=[b.predict_many([i])[0] for i in range(start+12,start+24)]
        resume=equivalent(got[12:],replay,1e-6)
        assert got[-1]['input_audit']['history_tokens']>0
        return dict(status='passed',eligible_batches=[1],choice_batch=1,
                    max_difference=error,resume_difference=resume,history_prefix=start,observations=24)
    cases=[]
    selected={r['key']:r for r in (min(rows,key=lambda r:len(r['task_prompt'])),max(rows,key=lambda r:len(r['task_prompt'])))}
    for row in selected.values():
        for level in (2,5):
            entry=all_entries[level][row['key']];e=model_episode(row,entry)
            boundary=next(i for i in range(1,len(e['images'])) if entry['segment_ids'][i]!=entry['segment_ids'][i-1])
            start=min(max(0,boundary-4),len(e['images'])-16)
            steps=sorted(set([0,len(e['images'])-1]+list(range(start,start+16))))
            b.native.begin(e);ref={}
            for i in steps:guard.check();ref[i]=b.native.predict_many([i])[0]
            cases.append((e,start,ref))
    # Larger RoboMeter batches already failed the fixed tolerance. Do not retry
    # those unsafe choices; optimize exact singleton concurrency instead.
    for batch in ((1,4,8,16) if b.name=='procvlm-2b' else (1,)):
        elapsed=0;count=0;error=0;ok=True;reason=None
        try:
            for e,start,ref in cases:
                b.begin(e)
                windows=[list(range(i,min(i+batch,start+16))) for i in range(start,start+16,batch)]
                windows=[[0]]+windows+[[len(e['images'])-1]]
                for steps in windows:
                    # Short batches have their own prefetch grid, just as at an episode tail.
                    b.begin(e);guard.check();torch.cuda.synchronize();t=time.perf_counter()
                    got=b.predict_many(steps);torch.cuda.synchronize();elapsed+=time.perf_counter()-t
                    count+=len(steps);error=max(error,equivalent([ref[i] for i in steps],got,.002))
                    b.drain()
        except AssertionError as ex:ok=False;reason=str(ex)
        trials.append(dict(batch_size=batch,passed=ok,max_difference=error,
                           observations_per_second=count/elapsed if elapsed else 0,reason=reason))
    passed=[r for r in trials if r['passed']];assert passed
    # Actual batch selection is shared by both GPUs, then concurrency is measured.
    choice=max(passed,key=lambda r:r['observations_per_second'])['batch_size']
    return dict(status='passed',eligible_batches=[r['batch_size'] for r in passed],choice_batch=choice,trials=trials)


def run():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args()
    ticket=read(a.ticket);task=ticket['task'];folder=ROOT/'tasks'/task/'workers'/str(ticket['rank'])
    taskdir=folder.parent.parent;folder.mkdir(parents=True,exist_ok=True)
    state=dict(process=process_identity(os.getpid()),task=task,gpu=ticket['gpu'],rank=ticket['rank'])
    def update(**kw):state.update(kw,updated=time.time());save(folder/'status.json',state)
    update(status='starting');claimed=None;owner=str(os.getpid())
    try:
        data,execution=source();verify_code();info=data['registry'][task]
        assert sha(ROOT/'execution.json')==ticket['execution_sha256']
        guard=Guard(ticket,folder)
        import random
        import numpy as np
        import torch
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
        guard.admit(torch);torch.cuda.manual_seed_all(42)
        from experiments.group5_throughput_20261115.backend import Backend
        update(status='loading_model');b=Backend(info)
        save(folder/'initialization.json',b.audit);guard.check()
        rows=[r for r in data['episodes'] if r['suite']==info['suite']];entries=streams(data)
        if ticket['replica']==0:
            update(status='checking_numerical_equivalence')
            proof=check_equivalence(b,rows,entries,guard)
            proof.update(initialization=b.audit,checkpoint_sha256=info['checkpoint_sha256'],finished=time.time())
            save(folder/'equivalence.json',proof)
        save(folder/'loaded.json',dict(status='passed',initialization=b.audit))
        selected=wait_file(taskdir/'batch.json',guard);batch=selected['batch_size']
        row=max(rows,key=lambda r:r['frames']);entry=entries[5][row['key']];e=model_episode(row,entry)
        for count in (1,2,3,4):
            stage=taskdir/'tuning'/str(count)
            wait_file(stage/'prepare.json',guard)
            if ticket['replica']<count:
                update(status='warming_up',replicas_per_gpu=count,batch_size=batch)
                b.begin(e);start=min(128,max(0,len(e['images'])-max(96,batch*8)))
                for i in range(start):guard.check();b.rebuild_step(i)
                i=start
                for _ in range(24 if b.name=='forest' else 4):
                    guard.check()
                    if i+batch>len(e['images']):b.begin(e);i=0
                    b.predict_many(list(range(i,min(i+batch,len(e['images'])))));i+=batch
                torch.cuda.synchronize()
                save(stage/f'ready-{ticket["rank"]}.json',dict(ready=time.time()))
                go=wait_file(stage/'go.json',guard)
                update(status='benchmarking',replicas_per_gpu=count)
                t=time.perf_counter();observations=0;torch.cuda.reset_peak_memory_stats()
                while time.perf_counter()-t<go['seconds']:
                    guard.check()
                    if i+batch>len(e['images']):
                        b.begin(e)
                        for j in range(start):guard.check();b.rebuild_step(j)
                        i=start
                    steps=list(range(i,min(i+batch,len(e['images']))))
                    b.predict_many(steps);i+=len(steps);observations+=len(steps)
                torch.cuda.synchronize();seconds=time.perf_counter()-t;b.drain()
                save(stage/f'result-{ticket["rank"]}.json',dict(status='passed',observations=observations,
                    seconds=seconds,observations_per_second=observations/seconds,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,finished=time.time()))
            wait_file(stage/'done.json',guard)
        choice=wait_file(taskdir/'choice.json',guard)
        assert choice['status']=='passed' and choice['checkpoint_sha256']==info['checkpoint_sha256']
        if ticket['replica']>=choice['replicas_per_gpu']:
            b.drain();update(status='tuning_only_completed');return 0
        wait_file(taskdir/'formal-go.json',guard)
        runtime=dict(execution_sha256=ticket['execution_sha256'],backend=b.execution_audit,
                     tuning_sha256=sha(taskdir/'choice.json'),batch_size=batch)
        rows={r['key']:r for r in rows};done=0
        while True:
            guard.check();claimed=claim(taskdir/'queue.sqlite',owner)
            if claimed is None:break
            key,level=claimed['key'],claimed['level'];row=rows[key];entry=entries[level][key]
            cache=verify_cache(data,info['model'],row,entry)
            update(status='evaluating',level=level,episode=key,completed_curves=done,observations_in_episode=0)
            r=run_episode(b,row,entry,identity(execution['inputs_sha256'],info,row,entry),
                result_path(info['model'],level,key),batch,runtime,guard.check,
                lambda n:update(observations_in_episode=n))
            assert r['source_frame_indices']==cache['inference']['positions']
            assert digest([a['input_playback_slots'] for a in r['input_audits']])==digest(cache['input_playback_slots'])
            finish(taskdir/'queue.sqlite',owner,key,level);claimed=None;done+=1
        b.drain();update(status='completed',completed_curves=done);return 0
    except SafetyPause as ex:
        if claimed:finish(taskdir/'queue.sqlite',owner,claimed['key'],claimed['level'],'pending')
        update(status='paused',reason=str(ex));return 75
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':raise SystemExit(run())
