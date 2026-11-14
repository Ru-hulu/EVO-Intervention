"""Start the authorized fresh evaluation only after all nine tuning gates pass."""
import fcntl
import os
import subprocess
import time
from .common import *
from experiments.soc_disrupted_eval_20261115.guard import alive, process_identity, gpu_processes


def main():
    lock=(OPT/'release.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OPT/'release.pid').write_text(str(os.getpid()))
    manifest_path=OPT/'runtime-manifest.json'
    assert read(manifest_path)['files']==runtime_hashes()
    while True:
        status=read(OPT/'tuning-status.json')
        if status['status']=='failed':
            save(OPT/'release-status.json',dict(status='blocked_by_tuning_failure',failures=status['failures'],updated=time.time()))
            return
        if status['status']=='completed' and not alive(status['controller']):break
        save(OPT/'release-status.json',dict(status='waiting_for_nine_validated_configurations',
            controller=process_identity(os.getpid()),updated=time.time(),completed=status['completed']))
        time.sleep(10)
    assert not any(gpu_processes().values()),'An unrelated task acquired a GPU; inspect before release'
    assert not alive(read(ROOT/'status.json')['controller'])
    for item in read(ROOT/'launches.json'):assert not alive(item['process'])
    assert read(manifest_path)['files']==runtime_hashes()
    assert (ROOT/'pause-all').exists() and (OPT/'previous-results-cleared.json').exists()
    assert not OUT.exists() or not list(OUT.rglob('*.json')),'Previous formal results were not cleared'
    source=read(ROOT/'inputs.json');configurations={}
    reuse=read(OPT/'exact-input-reuse-counts.json')
    for key in jobs(source):
        receipt=verify_choice(key)
        chosen=receipt['chosen'];effective=chosen['observations_per_second'];memo=False
        singleton=next(c for c in receipt['candidates'] if c['batch_size']==1)
        if key in reuse and singleton['accepted']:
            estimated=singleton['observations_per_second']*reuse[key]['total']/reuse[key]['unique_exact_visual_inputs']*.9
            if estimated>effective:
                chosen=singleton;memo=True;effective=estimated
        configurations[key]=dict(batch_size=chosen['batch_size'],
            cpu_image_cache=chosen['cpu_image_cache'],cpu_threads=2,exact_input_memoization=memo,
            estimated_effective_obs_per_second=effective,
            cpu_cache_limit_bytes=chosen['cache_limit_bytes'],
            tuning_receipt_sha256=sha(OPT/'tuning'/f'{key}.json'),runtime_manifest_sha256=sha(manifest_path),
            maximum_allowed_numeric_difference=.002,
            memory_updates='native singleton vision in presented order' if key.startswith('forest-') else 'native independent visual windows',
            durable_writes='atomic every 8 predictions or 2 seconds; flush on pause and episode end',initialization_audit_role='frozen native reference')
    save(OPT/'execution-configs.json',configurations)
    # This enables only the new queue, after all old GPU/controller identities
    # and benchmark processes have been verified to have exited.
    (ROOT/'pause-all').unlink()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=f'{OPT}/code:{ROOT}/code',PYTHONDONTWRITEBYTECODE='1')
    with (OPT/'evaluation-controller.log').open('a') as log:
        p=subprocess.Popen([PYTHON,'-m','experiments.soc_disrupted_opt_20261115.controller',
            '--start-evaluation','--resume'],cwd=OPT/'code',env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OPT/'release-status.json',dict(status='evaluation_controller_launched',updated=time.time(),
        controller=process_identity(p.pid),configuration_sha256=sha(OPT/'execution-configs.json'),
        runtime_manifest_sha256=sha(manifest_path)))


if __name__=='__main__':main()
