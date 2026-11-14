"""GPU smoke tests and independent batched current-frame predictions."""
import argparse
import fcntl
import math
import os
import statistics
import time
import traceback
from .common import *
from .protocol import current_sample,validate_record,metrics

def valid(x):return x is not None and math.isfinite(x) and 0<=x<=1

def smoke(backend,rows,folder):
    import torch
    # Mixed prompts, start/middle/end, and missing camera slots when present.
    selected=[max(rows,key=lambda r:len(r['task_prompt'])),min(rows,key=lambda r:len(r['task_prompt']))]
    missing=next((r for r in rows if None in r['videos']),None)
    if missing:selected.append(missing)
    samples=[current_sample(r,i,backend.name) for r in selected for i in (0,len(r['frame_indices'])//2,len(r['frame_indices'])-1)]
    reference=[backend.predict([s])[0] for s in samples]
    assert all(valid(x['prediction']) for x in reference),reference
    save(folder/'status.json',dict(status='testing_batches',reference_samples=len(reference),updated=time.time()))
    repeat=backend.predict([samples[0]])[0]
    assert repeat['input_fingerprint']==reference[0]['input_fingerprint']
    assert abs(repeat['prediction']-reference[0]['prediction'])<=1e-6,'History/order-dependent singleton prediction'
    candidates=[32,16] if backend.name=='procvlm-2b' else [64,32]
    trials=[]
    for batch_size in candidates:
        assert not (ROOT/'resource-stop.json').exists(),'Resource guard requested this evaluation stop'
        save(folder/'status.json',dict(status='testing_batches',batch_size=batch_size,trials=trials,updated=time.time()))
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        items=[samples[i%len(samples)] for i in range(batch_size)]
        try:
            elapsed=[];maximum=0.
            for _ in range(2):
                torch.cuda.synchronize();start=time.perf_counter();result=backend.predict(items);torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-start)
                assert all(valid(x['prediction']) for x in result),result
                assert all(x['input_fingerprint']==reference[i%len(samples)]['input_fingerprint'] for i,x in enumerate(result))
                maximum=max(maximum,max(abs(x['prediction']-reference[i%len(samples)]['prediction']) for i,x in enumerate(result)))
            free,_=torch.cuda.mem_get_info()
            acceptable=maximum<=.002 and free>=8*1024**3
            trials.append(dict(batch_size=batch_size,status='passed' if acceptable else 'numerical_or_memory_limit',
                max_singleton_difference=maximum,samples_per_second=batch_size/elapsed[-1],seconds=elapsed,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch_size=batch_size,status='oom'));torch.cuda.empty_cache()
    passed=[x for x in trials if x['status']=='passed']
    if passed:choice=max(passed,key=lambda x:x['samples_per_second'])
    else:
        start=time.perf_counter();backend.predict([samples[0]]);torch.cuda.synchronize()
        choice=dict(batch_size=1,status='passed',samples_per_second=1/(time.perf_counter()-start))
    report=dict(status='passed',audit=backend.audit,history_independence_checked=True,
        all_input_fingerprints_match=True,test_samples=len(samples),reference_predictions=reference,
        trials=trials,choice=choice,finished=time.time())
    save(folder/'verification.json',report)
    return report

def evaluate(backend,rows,info,folder,inputs_sha):
    policy=read(ROOT/'success-policy.json');assert policy['all_selected_episodes_successful'] is True
    batch_size=read(ROOT/'smoke'/job_name(info['model'],info['suite'])/'verification.json')['choice']['batch_size']
    started=time.time();completed=0;new_samples=0
    for row in rows:
        identity=dict(model=info['model'],suite=info['suite'],checkpoint_sha256=info['checkpoint_sha256'],
            inputs_sha256=inputs_sha,key=row['key'],condition='no_memory',backend_sha256=sha(Path(__file__).with_name('backend.py')))
        destination=result_path(info['model'],row['key']);partial=destination.with_suffix('.partial.json')
        if destination.exists():
            validate_record(read(destination),row,identity,True);completed+=1;continue
        if partial.exists():record=read(partial);validate_record(record,row,identity)
        else:record=dict(identity=identity,predictions=[],raw_responses=[],input_token_counts=[],frame_indices=[],timestamps=[],target_pt=[],
            history_length=0,memory_enabled=False,input_images_per_frame=1 if info['model']=='procvlm-2b' else 3,
            initialization_audit=str(folder/'initialization.json'),source_video_sha256=row['video_hashes'],progress_sha256=row['progress_sha256'])
        for start in range(len(record['predictions']),len(row['frame_indices']),batch_size):
            assert not (ROOT/'resource-stop.json').exists(),'Resource guard requested this evaluation stop'
            indices=list(range(start,min(start+batch_size,len(row['frame_indices']))))
            outputs=backend.predict([current_sample(row,i,info['model']) for i in indices])
            for i,value in zip(indices,outputs):
                p=value['prediction'];record['predictions'].append(float(p) if valid(p) else None)
                record['raw_responses'].append(value['raw_response']);record['input_token_counts'].append(value['input_tokens'])
                for field in ('frame_indices','timestamps','target_pt'):record[field].append(row[field][i])
            save(partial,record);new_samples+=len(indices)
            save(folder/'status.json',dict(status='running',model=info['model'],suite=info['suite'],gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                completed_episodes=completed,total_episodes=len(rows),episode=row['key'],samples_in_episode=len(record['predictions']),
                new_samples=new_samples,batch_size=batch_size,samples_per_second=new_samples/max(time.time()-started,1),updated=time.time()))
        validate_record(record,row,identity,True)
        record.update(metrics=metrics(record['predictions'],row,True),finished=time.time())
        save(destination,record);partial.unlink();completed+=1
    report=dict(status='completed',model=info['model'],suite=info['suite'],episodes=completed,
        expected_episodes=len(rows),expected_predictions=sum(len(r['frame_indices']) for r in rows),
        checkpoint=info['checkpoint'],checkpoint_sha256=info['checkpoint_sha256'],batch_size=batch_size,
        seconds=time.time()-started,finished=time.time(),memory_enabled=False)
    save(folder/'verification.json',report);save(folder/'status.json',report)

def main():
    import torch
    from .backend import Backend
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--phase',choices=['smoke','formal'],required=True)
    p.add_argument('--base-job');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    folder=ROOT/a.phase/a.job;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.85)
        source=read(ROOT/'inputs.json');info=source['registry'][a.base_job or a.job]
        rows=[r for r in source['episodes'] if r['suite']==info['suite']]
        assert 0<=a.shard<a.shards
        if a.phase=='formal':rows=rows[a.shard::a.shards]
        else:assert a.shard==0 and a.shards==1
        save(folder/'status.json',dict(status='loading_model',job=a.job,phase=a.phase,pid=os.getpid(),updated=time.time()))
        backend=Backend(info);save(folder/'initialization.json',backend.audit)
        if a.phase=='smoke':
            report=smoke(backend,rows,folder);save(folder/'status.json',dict(status='completed',choice=report['choice'],updated=time.time()))
        else:evaluate(backend,rows,info,folder,sha(ROOT/'inputs.json'))
    except BaseException:
        save(folder/'status.json',dict(status='failed',error=traceback.format_exc(),updated=time.time()))
        raise

if __name__=='__main__':main()
