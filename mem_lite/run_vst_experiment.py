"""Persistent three-GPU training followed by full validation; status/logs on disk."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from mem_lite.train_vst import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    code = Path(__file__).resolve().parents[1]
    gpus = [int(x) for x in args.gpus.split(",")]
    status = {"runner_pid": os.getpid(), "gpus": gpus, "code": str(code), "config": str(Path(args.config).resolve()),
              "output_dir": config["output_dir"], "epochs": config["epochs"], "started_at": time.time()}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus, "PYTHONPATH": str(code), "OMP_NUM_THREADS": "4",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "NCCL_DEBUG": "WARN", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    child = None
    with (run / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "started.json").exists():
            raise RuntimeError("Run already started; do not launch duplicate jobs")
        write_json(run / "started.json", status)
        try:
            stages = [
                ("training", ["-m", "mem_lite.train_vst", "--config", str(Path(args.config).resolve())]),
                ("validation", ["-m", "mem_lite.evaluate_vst", "--checkpoint", str(Path(config["output_dir"]) / "last.pt"),
                                "--output-dir", str(Path(config["output_dir"]) / "validation-03")]),
            ]
            for phase, command in stages:
                with (run / f"{phase}.log").open("w") as log:
                    child = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={len(gpus)}", *command], cwd=code, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status.update(phase=phase, child_pid=child.pid, phase_started_at=time.time())
                while child.poll() is None:
                    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                                                   "--format=csv,noheader,nounits"], text=True)
                    with (run / "gpu.jsonl").open("a") as stream:
                        stream.write(json.dumps({"time": time.time(), "gpus": gpu.strip().splitlines()}) + "\n")
                    status["updated_at"] = time.time()
                    write_json(run / "status.json", status)
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError(f"{phase} exited {child.returncode}; inspect {phase}.log")
            complete = json.loads((Path(config["output_dir"]) / "training-completed.json").read_text())
            if complete["epochs"] != config["epochs"]:
                raise RuntimeError("Incomplete epoch count")
            metrics = json.loads((Path(config["output_dir"]) / "validation-03/metrics.json").read_text())
            status.update(phase="completed", completed_at=time.time(), training=complete, validation=metrics)
            write_json(run / "status.json", status)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            status.update(phase="failed", error=str(exc), traceback=traceback.format_exc(), updated_at=time.time())
            write_json(run / "status.json", status)
            raise


if __name__ == "__main__":
    main()
