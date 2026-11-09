"""Single-GPU augmentation job, authorized to share GPU 3 with existing training."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import prepare_repeats as a

GPU = 3
child = None


def status(phase, **kwargs):
    a.write_json(a.SETUP/"status.json",dict(phase=phase,pid=os.getpid(),gpu=GPU,
                                         mode="shared_single_gpu",updated_at=a.now(),
                                         training_started=False,**kwargs))


def launch(script, arguments, label, gpu):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(a.CODE),PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(a.ROOT/"cache/huggingface"),
               TRITON_CACHE_DIR=str(a.ROOT/"cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(a.ROOT/"cache/torchinductor"))
    with (a.SETUP/f"{label}.log").open("a") as log:
        return subprocess.Popen([sys.executable,"-u",str(a.SETUP/script),*arguments],
                                cwd=a.SETUP,env=env,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT)


def stop(signum, frame):
    raise InterruptedError(f"Received signal {signum}")


def main():
    global child
    lock=(a.SETUP/"generation.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    plan=a.get_plan()
    smoke=a.read_json(a.SETUP/"gpu-smoke.json")
    if smoke["status"]!="passed" or smoke["plan_sha256"]!=plan["plan_sha256"]:
        raise ValueError("Real GPU smoke must pass for this exact plan")
    try:
        child=launch("prepare_repeats.py",["worker","--worker","0","--workers","1"],"worker-0",GPU)
        status("building",child_pid=child.pid,total_episodes=858,total_samples=plan["augmented_samples"])
        while child.poll() is None:
            worker=a.read_json(a.SETUP/"worker-0.json") if (a.SETUP/"worker-0.json").exists() else {}
            status("building",child_pid=child.pid,total_episodes=858,total_samples=plan["augmented_samples"],
                   worker=worker)
            time.sleep(15)
        if child.returncode!=0:
            raise RuntimeError(f"Cache worker exited with {child.returncode}; inspect worker-0.log")
        child=launch("prepare_repeats.py",["verify"],"verify","")
        status("verifying",child_pid=child.pid)
        if child.wait()!=0:
            raise RuntimeError("Final cache verification failed; inspect verify.log")
        report=a.read_json(a.OUTPUT/"report.json")
        status("completed",report=report,train_sample_files=str(a.OUTPUT/"train.sample_files.json"))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status("failed",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__=="__main__":
    main()

