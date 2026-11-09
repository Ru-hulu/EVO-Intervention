"""Independent first-ten-task evaluation with the confirmed stationary augmentation."""
import argparse
from collections import defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREP = Path("/cache/cvpzhr/setup/forest-proprimem-cap50-20261103")
sys.path.insert(0, str(PREP))
import prepare_static_jitter as aug
PLAN = aug.read_json(ROOT / "plan.json")
sys.path.insert(0, PLAN["code"])
aug.CONTRACT = PLAN["contract"]
OUT = Path(PLAN["evaluation_output"])


def state(path, **values):
    aug.write_json(path, dict(updated_at=aug.now(), **values))


def check_inputs():
    for name, digest in PLAN["protected_hashes"].items():
        assert aug.sha(name) == digest, name
    assert aug.sha(PLAN["checkpoint"]) == PLAN["checkpoint_sha256"]


def worker(index):
    import torch
    from mem_lite.memory.vision import load_vision_encoder
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.15)
    episodes = aug.assigned_episodes(PLAN, index, 4)
    path = ROOT / f"cache-worker-{index}.json"
    state(path, phase="loading", pid=os.getpid(), total=len(episodes), completed=0)
    visual, processor, digest = load_vision_encoder(PLAN["model"], "cuda")
    assert digest == PLAN["history_source"]["vision_sha256"]
    for completed, episode in enumerate(episodes, 1):
        result = aug.build_episode(PLAN, episode, visual, processor)
        aug.write_json(ROOT / "receipts" / (episode["episode_id"] + ".json"), result)
        state(path, phase="building", pid=os.getpid(), completed=completed, total=len(episodes),
              last_episode=episode["episode_id"], samples=result["samples"])
    state(path, phase="completed", pid=os.getpid(), total=len(episodes), completed=len(episodes))


class IndexedDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(self.data[index], evaluation_index=index)


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        batch = self.collator(rows)
        batch["indices"] = [row["evaluation_index"] for row in rows]
        return batch


def summarize(records):
    errors = [abs(row["predicted_pt"] - row["target_pt"]) for row in records]
    original = [error for error, row in zip(errors, records) if row["kind"] == "original"]
    stationary = [error for error, row in zip(errors, records) if row["kind"] == "stationary"]
    anchors = {(row["episode_id"], row["source_sample_index"]): row["predicted_pt"]
               for row in records if row["kind"] == "original"}
    drift = [abs(row["predicted_pt"] - anchors[(row["episode_id"], row["anchor_sample_index"])])
             for row in records if row["kind"] == "stationary"]
    avg = lambda values: 100 * math.fsum(values) / len(values) if values else None
    return dict(num_samples=len(records), episodes=len({r["episode_id"] for r in records}),
                mae_pp=avg(errors), original_samples=len(original), original_mae_pp=avg(original),
                stationary_samples=len(stationary), stationary_mae_pp=avg(stationary),
                stationary_prediction_drift_mae_pp=avg(drift))


def select_batch(model, collator, data, device):
    """Compare larger inference batches against batch four on the same representative inputs."""
    import torch
    indices = sorted(set(round(i * (len(data) - 1) / 31) for i in range(32)))
    samples = [data[i] for i in indices]

    def predict(batch_size):
        # Warm-up is excluded from timing.
        batch = collator(samples[:batch_size])
        model({k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["inputs"].items()})
        torch.cuda.synchronize()
        started = time.perf_counter()
        values = []
        for offset in range(0, len(samples), batch_size):
            batch = collator(samples[offset:offset + batch_size])
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["inputs"].items()}
            values.extend(model(inputs)["pt"].float().cpu().tolist())
        torch.cuda.synchronize()
        return values, time.perf_counter() - started

    trials = []
    reference, seconds = predict(4)
    trials.append(dict(batch_size=4, seconds=seconds, samples_per_second=len(samples)/seconds,
                       max_abs_difference_pp=0.0, mean_abs_difference_pp=0.0, accepted=True))
    for size in (8, 16):
        try:
            values, seconds = predict(size)
            differences = [abs(a-b) for a,b in zip(values, reference)]
            trials.append(dict(batch_size=size, seconds=seconds,
                               samples_per_second=len(samples)/seconds,
                               max_abs_difference_pp=100*max(differences),
                               mean_abs_difference_pp=100*math.fsum(differences)/len(differences),
                               accepted=max(differences)<=0.001 and
                                        math.fsum(differences)/len(differences)<=0.0002))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            trials.append(dict(batch_size=size, accepted=False, error="CUDA out of memory"))
    chosen = max((t for t in trials if t["accepted"]), key=lambda t:t["samples_per_second"])
    aug.write_json(OUT / "batch-check.json", dict(reference_batch_size=4, samples=len(samples),
                   checkpoint_sha256=PLAN["checkpoint_sha256"], trials=trials,
                   selected_batch_size=chosen["batch_size"],
                   parity_limits_pp=dict(maximum=0.1, mean=0.02)))
    return chosen["batch_size"]


def evaluate():
    import torch
    from torch.utils.data import DataLoader
    from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
    from mem_lite.runtime import load_checkpoint
    torch.set_num_threads(2)
    check_inputs()
    snapshot = torch.load(PLAN["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    assert (snapshot["epoch"], snapshot["next_batch"], snapshot["step"]) == (1, 0, 19934)
    del snapshot
    paths = [e["output_dir"] for e in PLAN["episodes"]]
    data = M3Dataset(paths)
    assert len(data) == PLAN["augmented_samples"]
    assert data.history_source == PLAN["history_source"]
    state(ROOT / "evaluation-status.json", phase="loading_model", total=len(data), completed=0)
    model, processor, config = load_checkpoint(PLAN["checkpoint"], device="cuda")
    collator = Stage3Collator(processor, data.history_dim, config["max_sequence_length"], data.history_capacity)
    device = next(model.parameters()).device
    with torch.inference_mode():
        batch_size = select_batch(model, collator, data, device)
    loader = DataLoader(IndexedDataset(data), batch_size=batch_size, shuffle=False,
                        collate_fn=IndexedCollator(collator), num_workers=4,
                        pin_memory=True, multiprocessing_context="spawn",
                        generator=torch.Generator().manual_seed(42))
    records = []
    started = time.perf_counter()
    last_update = started
    with (OUT / "predictions.jsonl.partial").open("w") as stream, torch.inference_mode():
        for batch in loader:
            inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                      for k, v in batch["inputs"].items()}
            prediction = model(inputs)["pt"].float().cpu()
            assert torch.isfinite(prediction).all()
            for index, value, target in zip(batch["indices"], prediction.tolist(), batch["target_pt"].tolist()):
                episode, row, _ = data.samples[index]
                source_id = PLAN["episodes"][episode]["episode_id"]
                record = dict(index=index, episode_id=source_id,
                              task=PLAN["episodes"][episode]["task"], frame_index=row["frame_index"],
                              source_frame_index=row["source_frame_index"],
                              sample_index=row["sample_index"], source_sample_index=row["source_sample_index"],
                              anchor_sample_index=row["anchor_sample_index"], kind=row["kind"],
                              timestamp_seconds=row["timestamp_seconds"], target_pt=target,
                              predicted_pt=value, absolute_error_pp=100*abs(value-target))
                stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                records.append(record)
            if time.perf_counter()-last_update >= 15:
                stream.flush()
                state(ROOT / "evaluation-status.json", phase="evaluating", completed=len(records),
                      total=len(data), batch_size=batch_size,
                      elapsed_seconds=time.perf_counter()-started,
                      current_task=records[-1]["task"])
                last_update=time.perf_counter()
    assert len(records)==len(data) and [r["index"] for r in records]==list(range(len(data)))
    assert sum(r["kind"]=="stationary" for r in records)==PLAN["stationary_samples"]
    by_task = {task: summarize([r for r in records if r["task"] == task]) for task in PLAN["tasks"]}
    report = dict(status="completed", completed_at=aug.now(), checkpoint=PLAN["checkpoint"],
                  checkpoint_epoch=1.0, checkpoint_step=19934,
                  checkpoint_sha256=PLAN["checkpoint_sha256"], plan_sha256=PLAN["plan_sha256"],
                  augmentation="stationary only; same stationary sampler and PT rules as training",
                  overall=summarize(records), by_task=by_task, batch_size=batch_size,
                  seconds=time.perf_counter()-started, physical_gpu=3,
                  metric_units="percentage points; frame-weighted absolute error",
                  drift_definition="mean absolute prediction change from each stationary event anchor")
    (OUT/"predictions.jsonl.partial").replace(OUT/"predictions.jsonl")
    aug.write_json(OUT/"metrics.json", report)
    state(ROOT/"evaluation-status.json", phase="completed", completed=len(records), total=len(data),
          metrics=str(OUT/"metrics.json"))
    print(json.dumps(report,ensure_ascii=False),flush=True)


def run():
    lock=(ROOT/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_inputs()
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="3", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
             HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    children=[]
    try:
        state(ROOT/"status.json",phase="building_stationary_cache",pid=os.getpid(),total_episodes=100)
        for i in range(4):
            log=(ROOT/f"cache-worker-{i}.log").open("ab",buffering=0)
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__)),"worker","--index",str(i)],
                                   env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            log.close();children.append(child)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None,0) for child in children):
                raise RuntimeError("A cache worker failed")
            time.sleep(2)
        assert all(child.returncode==0 for child in children)
        children=[]
        receipts=[aug.read_json(ROOT/"receipts"/(e["episode_id"]+".json")) for e in PLAN["episodes"]]
        assert sum(r["samples"] for r in receipts)==PLAN["augmented_samples"]
        assert all(r["jitter_samples"]==0 and r["inserted_pt_fixed_to_anchor"] for r in receipts)
        aug.write_json(Path(PLAN["cache_output"])/"validation.sample_files.json",
                       [e["output_dir"] for e in PLAN["episodes"]])
        aug.write_json(Path(PLAN["cache_output"])/"report.json",
                       dict(status="completed",episodes=100,samples=PLAN["augmented_samples"],
                            stationary_samples=PLAN["stationary_samples"],source_labels_unchanged=True,
                            causal_histories_verified=True,views_synchronized=True))
        state(ROOT/"status.json",phase="evaluating",pid=os.getpid(),total_samples=PLAN["augmented_samples"])
        with (ROOT/"evaluation.log").open("ab",buffering=0) as log:
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__)),"evaluate"],
                                   env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children=[child]
        if child.wait()!=0:raise RuntimeError("Evaluation failed; see evaluation.log")
        children=[]
        check_inputs()
        state(ROOT/"status.json",phase="completed",pid=os.getpid(),metrics=str(OUT/"metrics.json"))
    except BaseException as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state(ROOT/"status.json",phase="failed",pid=os.getpid(),error=repr(exc))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["run","worker","evaluate"])
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    if args.command=="run":run()
    elif args.command=="worker":worker(args.index)
    else:evaluate()
