"""Two GPUs evaluate while two validate; expand to four after all gates pass."""
import fcntl
import os
import signal
import subprocess
import time
import traceback
from .common import *
from experiments.soc_disrupted_eval_20261115.guard import alive,process_identity,gpu_processes


def publish_ready(source):
    configs={};reuse=read(OPT/'exact-input-reuse-counts.json')
    for key in jobs(source):
        path=OPT/'tuning'/f'{key}.json'
        if not path.exists() or read(path).get('status')!='passed':continue
        receipt=verify_choice(key);chosen=receipt['chosen'];effective=chosen['observations_per_second'];memo=False
        singleton=next(c for c in receipt['candidates'] if c['batch_size']==1)
        if key in reuse and singleton['accepted']:
            estimate=singleton['observations_per_second']*reuse[key]['total']/reuse[key]['unique_exact_visual_inputs']*.9
            if estimate>effective:chosen=singleton;effective=estimate;memo=True
        configs[key]=dict(batch_size=chosen['batch_size'],cpu_image_cache=chosen['cpu_image_cache'],cpu_threads=2,
            exact_input_memoization=memo,estimated_effective_obs_per_second=effective,
            cpu_cache_limit_bytes=chosen['cache_limit_bytes'],tuning_receipt_sha256=sha(path),
            runtime_manifest_sha256=sha(OPT/'runtime-manifest.json'),maximum_allowed_numeric_difference=.002,
            memory_updates='native singleton vision in presented order' if key.startswith('forest-') else 'native independent visual windows',
            durable_writes='atomic every 8 predictions or 2 seconds; flush on pause and episode end',
            initialization_audit_role='frozen native reference')
    old=read(OPT/'execution-configs.json') if (OPT/'execution-configs.json').exists() else {}
    for key,value in old.items():assert configs[key]==value,'Never mutate an active job configuration'
    save(OPT/'execution-configs.json',configs)
    return configs


def launch(module,arguments,log_name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=f'{OPT}/code:{ROOT}/code',PYTHONDONTWRITEBYTECODE='1')
    with (OPT/log_name).open('a') as log:
        p=subprocess.Popen([PYTHON,'-m',f'experiments.soc_disrupted_opt_20261115.{module}',*arguments],
            cwd=OPT/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return p


def main():
    lock=(OPT/'two-plus-two.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(OPT/'runtime-manifest.json')['files']==runtime_hashes()
    handoff=read(OPT/'two-plus-two-handoff.json');source=read(ROOT/'inputs.json')
    eval_gpus=handoff['evaluation_gpus'];tuning_gpus=handoff['tuning_gpus'];old=handoff['old_tuning_controller']
    assert not alive(read(ROOT/'status.json')['controller'])
    for item in read(ROOT/'launches.json'):assert not alive(item['process'])
    assert all(not gpu_processes()[g] for g in eval_gpus)
    assert (ROOT/'pause-all').exists() and (OPT/'previous-results-cleared.json').exists()
    assert not OUT.exists() or not list(OUT.rglob('*.json'))
    ready=publish_ready(source);assert 'procvlm-2b-State' in ready and 'procvlm-2b-Order' in ready
    save(OPT/'execution-gpus.json',eval_gpus);(ROOT/'pause-all').unlink()
    evaluation=launch('controller',['--start-evaluation','--resume'],'evaluation-controller.log')
    tuner=None;old_retired=False;initial=time.time()
    while True:
        assert read(OPT/'runtime-manifest.json')['files']==runtime_hashes()
        ready=publish_ready(source)
        if evaluation.poll() is not None:
            state=read(ROOT/'status.json')
            if state.get('status')=='completed':return
            raise RuntimeError(f'Evaluation controller stopped: {evaluation.returncode}, {state.get("status")}')
        if not old_retired:
            running=[v for v in handoff['old_status']['active'].values() if alive(v['process'])]
            if not running:
                if alive(old):os.kill(old['pid'],signal.SIGKILL)
                old_retired=True
                save(OPT/'benchmark-controller-retired.json',dict(controller=old,retired_at=time.time(),
                     all_benchmark_children_exited=True,healthy_gpu_processes_signaled=False))
        if old_retired and tuner is None:
            if all(not gpu_processes()[g] for g in tuning_gpus):
                tuner=launch('tune_queue',['--gpus',','.join(map(str,tuning_gpus))],'tuning-controller-resumed.log')
        if tuner is not None and tuner.poll() is not None:
            status=read(OPT/'tuning-status.json')
            assert tuner.returncode==0 and status['status']=='completed',status
            ready=publish_ready(source);assert len(ready)==9
            save(OPT/'execution-gpus.json',[0,1,2,3])
            save(OPT/'two-plus-two-status.json',dict(status='all_four_gpus_released_to_evaluation',
                updated=time.time(),evaluation_controller=process_identity(evaluation.pid),ready=list(ready),
                evaluation_gpus=[0,1,2,3],tuning_completed=True))
            return
        save(OPT/'two-plus-two-status.json',dict(status='two_evaluating_two_validating',updated=time.time(),
            coordinator=process_identity(os.getpid()),evaluation_controller=process_identity(evaluation.pid),
            evaluation_gpus=eval_gpus,tuning_gpus=tuning_gpus,ready=list(ready),
            old_benchmark_controller_retired=old_retired,new_benchmark_controller=process_identity(tuner.pid) if tuner else None))
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception:
        save(OPT/'two-plus-two-error.json',dict(error=traceback.format_exc(),updated=time.time()))
        raise
