"""Frozen Count/Order/State manifest -> reused/new dense PT, with a separate cache gate."""
import argparse, copy, csv, fcntl, hashlib, json, math, os, shutil, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

SETUP = Path(__file__).resolve().parent
CODE = SETUP / "code"
sys.path.insert(0, str(CODE))
ROOT = Path("/cache/EVO-Intervention/datasets/ProPriMem-current")
EXPORTS = Path("/cache/cvpzhr/setup/proprimem-count-order-state-split-20261103")
BASE = Path("/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-20261103")
AUG = Path("/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-repeat5-seed42-20261103")
OLD_LABELS = Path("/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030/labels")
MODEL = Path("/cache/cvpzhr/models/Qwen3.5-4B")
VISION = "1cc7c705e96c07fb52bcb1fb32f5e05271cc249fbc619c1128fd4eddbea47aea"
WORKERS = 16
BATCH_SIZE = 16

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_sources(plan):
    for path, expected in plan["protected_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Frozen input or implementation changed: " + path)

def get_plan():
    plan = read(SETUP / "plan.json")
    verify_sources(plan)
    return plan

def progress_check(row, output=BASE):
    metadata = read(output / "labels" / row["source_id"] / "metadata.json")
    assert metadata["vision_sha256"] == VISION
    assert metadata["pt_max_pixels"] == 262144 and metadata["annotations_used"] is False
    assert metadata["feature"] == "post-merger tokens mean, FP32"
    assert metadata["delta"] == "adjacent-frame L2, FP32; all original frames"
    source = metadata["source"]
    for key in ("source_id", "frames", "video_sha256", "pt_method"):
        assert source[key] == row[key], (row["source_id"], key)
    assert abs(source["fps"] - row["fps"]) < 1e-3
    assert len(metadata["units"]) == 1
    unit = metadata["units"][0]
    path = Path(unit["progress_csv"])
    assert sha(path) == unit["progress_sha256"], str(path)
    with path.open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == row["frames"] and unit["frame_range"] == [0, row["frames"]]
    values, deltas = [], []
    for i, item in enumerate(records):
        assert int(item["frame_index"]) == i
        pt, delta = float(item["pt"]), float(item["delta"])
        assert math.isfinite(pt) and 0 <= pt <= 1
        assert math.isfinite(delta) and delta >= 0
        values.append(pt); deltas.append(delta)
    total = sum(deltas)
    assert total > 1e-8 and deltas[0] == values[0] == 0 and values[-1] == 1
    acc = 0.
    for delta, pt in zip(deltas, values):
        acc += delta
        assert abs(acc / total - pt) < 1e-7, row["source_id"]
    return metadata, values

def make_plan():
    import av
    assert not (SETUP / "plan.json").exists()
    assert not BASE.exists() and not AUG.exists()
    sources = []
    manifest = read(ROOT / "manifest.json")
    assert manifest["totals"] == dict(tasks=47, trajectories=2074, train=1656, val=418)
    protected = {str(ROOT / "manifest.json"): sha(ROOT / "manifest.json")}
    for relative, expected in read(SETUP / "source-code-sha256.json").items():
        path = SETUP / relative
        assert sha(path) == expected
        protected[str(path)] = expected
    for split in ("train", "validation"):
        path = EXPORTS / (split + ".trajectories.jsonl")
        protected[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["split"] == split and row["suite"] in ("Count", "Order", "State")
            assert row["pt_method"] == "whole-video-cumulative-visual-delta-v1"
            rates, starts = [], []
            for slot, value in enumerate(row["videos"]):
                if value is None:
                    continue
                path = Path(value)
                assert path.is_file()
                with av.open(str(path)) as video:
                    stream = video.streams.video[0]
                    if stream.frames and stream.frames != row["frames"]:
                        assert row["source_id"] == "Order/03-swap_blocks_with_confirmation/episode_001342"
                        assert slot == 1 and stream.frames == 435 and row["frames"] == 436
                        row["cache_requires_view_decision"] = True
                        row["view_length_exception"] = dict(slot=1, decoded_frames=435, expected_frames=436,
                                                           missing_tail_frames=1, raw_video_unchanged=True)
                    rates.append(stream.average_rate)
                    starts.append(float((stream.start_time or 0) * stream.time_base))
            assert max(map(float, rates)) - min(map(float, rates)) < 1e-3
            assert max(starts) - min(starts) < 1e-6
            assert abs(float(rates[0]) - row["fps"]) < 1e-3
            row["fps_numerator"], row["fps_denominator"] = rates[0].numerator, rates[0].denominator
            row["pt_reused"] = (OLD_LABELS / row["source_id"] / "metadata.json").is_file()
            sources.append(row)
    assert len(sources) == len({r["source_id"] for r in sources}) == 2074
    assert sum(r["pt_reused"] for r in sources) == 816
    loads = [0] * WORKERS
    for row in sorted((r for r in sources if not r["pt_reused"]), key=lambda r: (-r["frames"], r["source_id"])):
        worker = min(range(WORKERS), key=loads.__getitem__)
        row["label_worker"] = worker
        loads[worker] += row["frames"]
    memory = read(CODE / "mem_lite/configs/streamforest.json")
    origin = dict(method="streamforest-v1", vision_sha256=VISION, memory_config=memory,
                  sample_fps=5.0, include_episode_last_frame=True, max_pixels=65536, min_pixels=4096,
                  view_policy="main-first; missing auxiliary views use RGB black 64x64; v1")
    plan = dict(format_version="proprimem-cap50-preparation-v1", created_at=now(),
                dataset_root=str(ROOT), dataset_version=manifest["dataset_version"],
                model=str(MODEL), base_output=str(BASE), augmentation_output=str(AUG),
                workers=WORKERS, pt_batch_size=BATCH_SIZE, pt_max_pixels=262144,
                dense_pt_frames=sum(r["frames"] for r in sources),
                reused_episodes=816, new_episodes=1258, label_worker_frames=loads,
                splits=dict(Counter(r["split"] for r in sources)), protected_hashes=protected,
                history_source=origin, sources=sources)
    BASE.mkdir(parents=True); AUG.mkdir(parents=True)
    for row in sources:
        if not row["pt_reused"]:
            continue
        old_output = OLD_LABELS.parent
        metadata, _ = progress_check(row, old_output)
        metadata = copy.deepcopy(metadata)
        old_unit = metadata["units"][0]
        directory = BASE / "labels" / row["source_id"]
        directory.mkdir(parents=True)
        shutil.copy2(old_unit["progress_csv"], directory / "progress.csv")
        metadata["reuse_provenance"] = dict(metadata=str(OLD_LABELS / row["source_id"] / "metadata.json"),
                                             progress_csv=old_unit["progress_csv"], reused_at=now())
        metadata["source"] = row
        old_unit["progress_csv"] = str(directory / "progress.csv")
        write(directory / "metadata.json", metadata)
    write(SETUP / "plan.json", plan)
    write(BASE / "index.json", plan)
    write(SETUP / "pt-status.json", dict(phase="prepared", reused_episodes=816, new_episodes=1258, updated_at=now()))
    print(json.dumps(dict(phase="prepared", splits=plan["splits"], reused=816, new=1258,
                          dense_frames=plan["dense_pt_frames"], worker_frames=loads)), flush=True)

def label_worker(worker, limit=None):
    import torch
    from mem_lite.memory.vision import load_vision_encoder
    from mem_lite.datasets.prepare_proprimem import label_source
    lock = (SETUP / ("pt-worker-%d.lock" % worker)).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.22)
    plan = get_plan()
    sources = [r for r in plan["sources"] if r.get("label_worker") == worker]
    sources.sort(key=lambda r: (-r["frames"], r["source_id"]))
    if limit is not None:
        sources = sources[:limit]
    status = SETUP / ("pt-worker-%d.json" % worker)
    state = dict(worker=worker, pid=os.getpid(), phase="loading_vision", total=len(sources),
                 completed=0, frames=0, updated_at=now())
    write(status, state); started=time.monotonic()
    try:
        encoder = load_vision_encoder(MODEL, "cuda")
        assert encoder[2] == VISION
        for row in sources:
            state.update(phase="labelling", current=row["source_id"], updated_at=now())
            write(status, state)
            label_source(row, BASE, SETUP / "unused-legacy", encoder, plan["pt_batch_size"], 262144)
            progress_check(row)
            state.update(completed=state["completed"]+1, frames=state["frames"]+row["frames"],
                         elapsed_seconds=time.monotonic()-started, updated_at=now(),
                         peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write(status, state)
            print(json.dumps(dict(worker=worker, source=row["source_id"], completed=state["completed"],
                                  total=len(sources), seconds=state["elapsed_seconds"])), flush=True)
        state.update(phase="completed", current=None, updated_at=now())
        write(status, state)
    except BaseException as exc:
        state.update(phase="failed", error=repr(exc), updated_at=now()); write(status,state)
        raise

def validate_labels():
    plan = get_plan(); count=Counter(); frames=Counter()
    for row in plan["sources"]:
        progress_check(row)
        count[row["suite"]] += 1; frames[row["suite"]] += row["frames"]
    report=dict(status="completed", completed_at=now(), episodes=sum(count.values()),
                frames=sum(frames.values()), reused_episodes=816, generated_episodes=1258,
                suites=dict(count), suite_frames=dict(frames), vision_sha256=VISION,
                pt_method="whole-video-cumulative-visual-delta-v1", source_manifest_sha256=sha(ROOT/"manifest.json"),
                all_frame_indices_contiguous=True, finite_monotonic_pt_0_to_1=True,
                progress_sha256_verified=True, normalized_cumulative_deltas_verified=True)
    assert report["episodes"]==2074
    write(BASE/"labels-report.json",report)
    return report

def run_labels():
    lock=(SETUP/"pt-controller.lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=get_plan(); workers=plan["workers"]; children=[]; logs=[]
    state=dict(pid=os.getpid(),phase="labelling",workers=workers,started_at=now(),
               reused_episodes=816,new_episodes=1258,updated_at=now())
    write(SETUP/"pt-status.json",state)
    try:
        for worker in range(workers):
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(worker%4),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                       TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                       PYTORCH_ALLOC_CONF="expandable_segments:True")
            log=(SETUP/("pt-worker-%d.log"%worker)).open("a");logs.append(log)
            children.append(subprocess.Popen([sys.executable,str(Path(__file__)),"label-worker","--worker",str(worker)],
                                             cwd=SETUP,env=env,stdout=log,stderr=subprocess.STDOUT))
        state["child_pids"]=[p.pid for p in children];write(SETUP/"pt-status.json",state)
        while any(p.poll() is None for p in children):
            failed=[p for p in children if p.poll() not in (None,0)]
            if failed:raise RuntimeError("PT worker failed: "+str([(p.pid,p.returncode) for p in failed]))
            time.sleep(3)
        assert all(p.returncode==0 for p in children)
        state.update(phase="validating",updated_at=now());write(SETUP/"pt-status.json",state)
        report=validate_labels()
        state.update(phase="completed",updated_at=now(),report=report);write(SETUP/"pt-status.json",state)
        print(json.dumps(state),flush=True)
    except BaseException as exc:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        state.update(phase="failed",error=repr(exc),updated_at=now());write(SETUP/"pt-status.json",state)
        raise
    finally:
        for log in logs:log.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["plan","label-worker","labels","validate-labels"])
    parser.add_argument("--worker",type=int,default=0);parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.command=="plan":make_plan()
    elif args.command=="label-worker":label_worker(args.worker,args.limit)
    elif args.command=="labels":run_labels()
    else:print(json.dumps(validate_labels()),flush=True)
