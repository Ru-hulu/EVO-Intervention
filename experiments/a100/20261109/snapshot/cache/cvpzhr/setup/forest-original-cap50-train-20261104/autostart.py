"""One-shot remote chain: completed caches -> GPU smoke -> fresh Forest training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PREP = Path("/cache/cvpzhr/setup/forest-proprimem-cap50-20261103")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def cache_ready(status):
    if status["phase"] == "completed":
        return True
    if status["phase"] == "failed":
        raise RuntimeError("Cache generation failed: " + status.get("error", "see cache-controller.log"))
    pid = status.get("pid")
    if not pid:
        raise RuntimeError("Cache controller PID is missing")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise RuntimeError("Cache controller exited before completing the caches") from exc
    return False


def wait_for_cache():
    while not cache_ready(read(PREP / "pipeline-status.json")):
        time.sleep(30)


def execute_chain(wait, run, verify):
    wait()
    run("smoke")  # This includes cache gates, multi-GPU forward/backward, validation and resume.
    run("train")  # launch.sh requires the matching successful smoke receipt, then uses fresh config.
    verify()


def verify_completion():
    config = read(ROOT / "config.json")
    output = Path(config["output_dir"])
    records = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    if len(records) != 6 or records[-1]["epoch_progress"] != 3.0:
        raise RuntimeError("Trainer exited without the six required validations")
    if any(row["num_samples"] != 66891 for row in records):
        raise RuntimeError("Validation sample counts differ from the full 417-episode validation set")
    if not all((output / name).is_file() for name in ("best.pt", "last.pt")):
        raise RuntimeError("Final best/last checkpoint is missing")


def main():
    with (ROOT / "autostart.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), phase="waiting_for_cache", started_at=now(),
                     training_started=False, automatic_start_authorized=True,
                     gpu_ids=[0, 1, 2, 3], world_size=4, batch_size_per_gpu=4,
                     implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

        def update(**values):
            state.update(values, updated_at=now())
            write(ROOT / "autostart-status.json", state)
            print(json.dumps(state, ensure_ascii=False), flush=True)

        def launch(stage):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", NPROC_PER_NODE="4")
            update(phase="gpu_smoke" if stage == "smoke" else "launching_training",
                   current_stage=stage)
            child = subprocess.Popen([str(ROOT / "launch.sh"), stage],
                                     cwd=ROOT, env=env, start_new_session=True)
            try:
                update(child_pid=child.pid, training_started=stage == "train",
                       phase="gpu_smoke" if stage == "smoke" else "training")
                if stage == "train":
                    for path in (ROOT / "readiness.json", ROOT / "confirmed-settings.json",
                                 PREP / "training-intent.json"):
                        metadata = read(path)
                        metadata.update(training_started=True, training_launched_at=now(),
                                        autostart_status=str(ROOT / "autostart-status.json"))
                        write(path, metadata)
                result = child.wait()
                if result != 0:
                    raise RuntimeError(f"{stage} exited with code {result}; see autostart.log and {stage}.log")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            update(child_pid=None, **({"smoke_passed": True} if stage == "smoke" else {}))

        def interrupt(signum, frame):
            raise InterruptedError(f"Autostart controller received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        update()
        try:
            execute_chain(wait_for_cache, launch, verify_completion)
            update(phase="completed", completed_at=now())
        except BaseException as exc:
            update(phase="failed", error=repr(exc), failed_at=now())
            raise


if __name__ == "__main__":
    main()
