"""Two-rank disposable feasibility run; never reuse its weights for training."""
import json
from pathlib import Path
import subprocess
import time
import traceback
from experiments.prm_five_groups_20261114.controller import BASE, CODE, PYTHON, env, training_config, verify_training
from experiments.prm_five_groups_20261114.prepare_cache import save


def main():
    # Cache workers drain cooperatively; never overlap the full model with them.
    import subprocess
    from experiments.prm_five_groups_20261114.controller import alive
    while True:
        workers=json.loads((BASE/'cache-worker-pids.json').read_text())
        if not any(alive(w['pid']) for w in workers if w['gpu'] in [0,1]):
            break
        time.sleep(5)
    config_path, config = training_config('State', [0,1], smoke=True)
    if Path(config['output_dir']).exists():
        raise FileExistsError('Existing smoke output must be inspected, not overwritten')
    save(BASE/'smoke/status.json',{'status':'running','started':time.time(),'trial_weights_reused':False})
    with (BASE/'smoke/train.log').open('w') as f:
        result=subprocess.run([PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             '-m','mem_lite.train_stage3','--config',str(config_path)],cwd=CODE,env=env([0,1]),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Two-rank smoke exited {result.returncode}; see smoke/train.log')
    proof=verify_training('State',config,smoke=True)
    save(BASE/'smoke/status.json',{'status':'passed','checkpoint':proof,'trial_weights_reused':False})


if __name__=='__main__':
    try:
        main()
    except Exception:
        save(BASE/'smoke/status.json',{'status':'failed','error':traceback.format_exc(),'finished':time.time()})
        raise
    finally:
        for gpu in [0,1]:
            (BASE/f'drain-gpu-{gpu}').unlink(missing_ok=True)
