"""Real-input, GPU equivalence and throughput gate; never writes formal results."""
import argparse
import gc
import os
import time
import traceback
from .common import *
from .backend import Backend
from experiments.soc_disrupted_eval_20261115.backend import Backend as Native
from experiments.soc_disrupted_eval_20261115.protocol import model_episode
from experiments.soc_disrupted_eval_20261115.guard import host_sample, gpu_processes, idle_safe, alive, process_identity
from experiments.recover_validation_20261114.common import safe
from experiments.soc_disrupted_eval_20261115.engine import resume_canary


def compare(reference, outputs):
    assert len(reference) == len(outputs)
    errors=[]; invalid_equal=True; inputs_equal=True
    for a,b in zip(reference,outputs):
        av,bv=a['prediction'],b['prediction']
        invalid_equal &= (av is None)==(bv is None)
        inputs_equal &= digest(a['input_audit'])==digest(b['input_audit'])
        if av is not None and bv is not None: errors.append(abs(av-bv))
    return dict(max_abs_error=max(errors,default=0),invalid_positions_equal=invalid_equal,
                input_equivalence_verified=inputs_equal,finite_comparisons=len(errors))


def cases(source, job):
    result=[]
    for condition in ('static_only-seed42','rewind_only-seed42'):
        pairs=task_inputs(source,job,condition)
        shortest=min(pairs,key=lambda p:len(p[1]['target_pt']))
        if job['model']=='forest':
            result.append((shortest[0],shortest[1],list(range(len(shortest[1]['target_pt'])))))
            if condition.startswith('rewind'):
                # Exercise a mature memory after a long real prefix as well.
                pair=max(pairs,key=lambda p:len(p[1]['target_pt']))
                n=len(pair[1]['target_pt']);result.append((pair[0],pair[1],list(range(max(0,n-48),n))))
        else:
            chosen=[shortest,min(pairs,key=lambda p:len(p[0]['task_prompt'])),max(pairs,key=lambda p:len(p[0]['task_prompt']))]
            seen=set()
            for row,entry in chosen:
                if row['key'] in seen: continue
                seen.add(row['key']);n=len(entry['target_pt'])
                indices={round(i*(n-1)/47) for i in range(48)}
                for event in entry['events']:
                    first=event['first_inserted_step'];duration=event['duration_steps']
                    indices.update(i for i in range(first-2,first+2*duration+3) if 0<=i<n)
                result.append((row,entry,sorted(indices)))
    return result


def run(job_id, gpu, parent_pid):
    import random
    import numpy as np
    import torch
    path=OPT/'tuning'/f'{job_id}.json';progress=OPT/'tuning'/f'{job_id}-status.json'
    source=read(ROOT/'inputs.json');job=jobs(source)[job_id];info=source['registry'][job_id]
    assert source['source_code_sha256']==code_hashes()
    initial=host_sample();initial_oom=initial['oom_kill'];parent=process_identity(parent_pid)
    assert parent and idle_safe(initial,gpu_processes(),gpu,initial_oom)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    last=0
    def check(stage=None):
        nonlocal last
        now=time.time()
        if now-last<5 and stage is None:return
        last=now
        assert alive(parent),'Tuning controller exited'
        assert safe(host_sample(),gpu,initial_oom),'Resource guard'
        assert len(gpu_processes()[gpu])<=1,'Another CUDA context appeared'
        if stage is not None:save(progress,dict(status='running',stage=stage,updated=now,process=process_identity(os.getpid()),gpu=gpu))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70);torch.cuda.manual_seed_all(42)
    check('loading')
    backend=Backend(info)
    report=dict(status='running',job_id=job_id,gpu=gpu,inputs_sha256=sha(ROOT/'inputs.json'),
                initialization=backend.audit,backend_code_sha256=backend_hashes(),candidates=[],started=time.time())
    save(path,report)
    try:
        canary_path=ROOT/'canary'/f'{job_id}.json'
        if not canary_path.exists():
            check('native_checkpoint_and_resume_canary');checks=[]
            for condition in ('static_only-seed42','rewind_only-seed42'):
                row,entry=min(task_inputs(source,job,condition),key=lambda p:len(p[1]['target_pt']))
                checks.append(dict(condition=condition,key=row['key'],comparisons=resume_canary(
                    backend,row,entry,identity(report['inputs_sha256'],info,row,entry),
                    ROOT/'canary'/job_id/condition,check)))
            save(canary_path,dict(status='passed',inputs_sha256=report['inputs_sha256'],initialization=backend.audit,
                                 checks=checks,gpu_forward_tested=True,formal_predictions_used=False))
        validate_canary(read(canary_path),report['inputs_sha256'],backend.audit)
        prepared=[]
        for index,(row,entry,steps) in enumerate(cases(source,job)):
            check(f'native_reference_{index}')
            episode=model_episode(row,entry);backend.begin(episode)
            if backend.name=='forest':
                for i in range(steps[0]):check();backend.rebuild_step(i)
            reference=[];torch.cuda.synchronize();start=time.monotonic()
            for i in steps:check();reference.append(Native.predict(backend,i))
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if backend.name!='forest':
                for i in steps:backend.encoded(i,verify=True)
            prepared.append((episode,steps,reference))
            save(OPT/'tuning'/job_id/f'reference-{index}.json',dict(key=row['key'],steps=steps,outputs=reference,
                elapsed=elapsed,observations_per_second=len(steps)/elapsed))
        report['reference_observations']=sum(len(v[1]) for v in prepared)
        report['case_keys']=[v[0]['key'] for v in prepared]
        report['native_obs_per_second']=report['reference_observations']/sum(
            read(OPT/'tuning'/job_id/f'reference-{i}.json')['elapsed'] for i in range(len(prepared)))
        save(path,report)
        sizes=(1,2,4,8,16) if backend.name=='forest' else (1,2,4,8,16,32)
        family_path=OPT/'tuning'/f"{job['model']}-State.json"
        if job['suite']!='State' and family_path.exists():
            family=read(family_path);tested=family.get('candidates',[])
            singles=[c for c in tested if c['batch_size']==1 and c['accepted']]
            larger=[c for c in tested if c['batch_size']>1]
            if singles and larger and not any(c['accepted'] for c in larger) and (
                family.get('status')=='passed' or (backend.name=='forest' and any(
                    c.get('observations_per_second',0)<singles[0]['observations_per_second'] for c in larger))):
                sizes=(1,)
                report['search_pruning']=dict(reason='Batched runtime family failed numerical equivalence on the State checkpoint; validate native singleton plus caching independently on this suite',
                    evidence_job=f"{job['model']}-State",observed_candidates=tested)
                save(path,report)
        for batch_size in sizes:
            check(f'benchmark_batch_{batch_size}')
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            candidate=dict(batch_size=batch_size,cpu_image_cache=backend.name!='forest',cpu_threads=2,
                           cache_limit_bytes=backend.cache_limit,accepted=False)
            all_reference=[];all_outputs=[];elapsed=0
            try:
                for episode,steps,reference in prepared:
                    backend.begin(episode)
                    if backend.name=='forest':
                        for i in range(steps[0]):check();backend.rebuild_step(i)
                    torch.cuda.synchronize();start=time.monotonic()
                    outputs=[]
                    for offset in range(0,len(steps),batch_size):
                        check();outputs.extend(backend.predict_many(steps[offset:offset+batch_size]))
                    torch.cuda.synchronize();elapsed+=time.monotonic()-start
                    all_reference.extend(reference);all_outputs.extend(outputs)
                candidate.update(compare(all_reference,all_outputs),elapsed_seconds=elapsed,
                    observations_per_second=len(all_outputs)/elapsed,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                candidate['accepted']=(candidate['max_abs_error']<=.002 and candidate['invalid_positions_equal']
                    and candidate['input_equivalence_verified'] and candidate['finite_comparisons']>0
                    and candidate['peak_reserved_mib']<56*1024)
                save(OPT/'tuning'/job_id/f'batch-{batch_size}-outputs.json',all_outputs)
            except torch.cuda.OutOfMemoryError:
                candidate.update(error='allocator memory cap reached; candidate rejected')
                gc.collect();torch.cuda.empty_cache()
                report['candidates'].append(candidate);save(path,report);break
            report['candidates'].append(candidate);save(path,report)
            # Do not try the next doubled allocation if it would consume the headroom.
            if candidate.get('peak_reserved_mib',0)>28*1024:break
        valid=[v for v in report['candidates'] if v['accepted']]
        assert valid,'No input/numeric-equivalent candidate'
        chosen=max(valid,key=lambda v:v['observations_per_second'])
        # Repeat selected runtime with a different batch boundary, simulating resume.
        check('selected_runtime_resume_verification');errors=[]
        for episode,steps,reference in prepared:
            cut=max(1,len(steps)//3);remaining=steps[cut:]
            backend.begin(episode)
            if backend.name=='forest':
                for i in range(remaining[0]):check();backend.rebuild_step(i)
            outputs=[]
            for offset in range(0,len(remaining),chosen['batch_size']):
                check();outputs.extend(backend.predict_many(remaining[offset:offset+chosen['batch_size']]))
            proof=compare(reference[cut:],outputs)
            assert proof['max_abs_error']<=.002 and proof['invalid_positions_equal'] and proof['input_equivalence_verified'],proof
            errors.append(proof)
        report.update(status='passed',chosen=chosen,resume_validation=errors,finished=time.time(),
                      formal_predictions_written=False,resource_final=host_sample())
        save(path,report);check('completed')
    except Exception:
        report.update(status='failed',error=traceback.format_exc(),finished=time.time());save(path,report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--parent',type=int,required=True);a=p.parse_args();run(a.job,a.gpu,a.parent)
