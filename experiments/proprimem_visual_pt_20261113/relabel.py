"""Replace time PT with Forest whole-video visual PT on State/Order/Count only.

Four GPUs run independent frozen-vision workers sharing an episode queue. Recover is hashed
before/after and never written. This pipeline never starts model evaluation.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

SUITES = ("State", "Order", "Count")
POLICY = "whole-video-cumulative-visual-delta-v1"
OLD_POLICY = "normalized_time_v1"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_ROOT = Path("/cache/cvpzhr/datasets/ProPriMem")
DEFAULT_SETUP = Path("/cache/cvpzhr/setup/proprimem-visual-pt-20261113")
MODEL = Path("/cache/cvpzhr/models/Qwen3.5-4B")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def json_lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save_lines(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temp.replace(path)


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes authorized dataset: {path}")
    return path


def strip_time_fields(value):
    """Remove only recognized time-label fields; stage/task annotations survive."""
    out = deepcopy(value)
    prog = out.get("progress")
    old = isinstance(prog, dict) and prog.get("policy") == OLD_POLICY
    old = old or out.get("progress_policy") == OLD_POLICY
    if old:
        for key in list(out):
            if key == "progress" or key == "has_progress" or key.startswith("progress_"):
                del out[key]
    if isinstance(out.get("progress_update"), dict) and out["progress_update"].get("policy") == OLD_POLICY:
        del out["progress_update"]
    return out


def recover_snapshot(root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted((root / "Recover").rglob("*")) if p.is_file()}


def inventory(root):
    result, stage_hashes, splits = [], {}, {}
    for suite in SUITES:
        base = root / suite
        manifest = read(base / "manifest.json")
        splits[suite] = {t["task_id"]: {"train": t["train"], "val": t["val"]} for t in manifest["tasks"]}
        for task in manifest["tasks"]:
            directory = inside(base / task["task_path"], base)
            index = inside(base / task["trajectory_index"], base)
            rows = json_lines(index)
            identifiers = {r["sample_id"] for r in rows}
            train, val = set(task["train"]), set(task["val"])
            assert len(rows) == len(identifiers) and not train & val and identifiers == train | val
            for row in rows:
                video = inside(directory / row["video_path"], directory)
                assert video.is_file() and video.stat().st_size == row["video_size_bytes"], str(video)
                assert row["total_frames"] >= 2 and row["fps"] > 0
                annotation = inside(directory / row["annotation_path"], directory) if row.get("annotation_path") else None
                if annotation:
                    obj = read(annotation)
                    assert obj["total_frames"] == row["total_frames"]
                    stage_hashes[str(annotation)] = object_hash(strip_time_fields(obj))
                out = directory / "progress" / (video.stem + ".progress.csv")
                result.append(dict(source_id=f"{suite}/{task['task_id']}/{row['sample_id']}",
                                   suite=suite, task_id=task["task_id"], sample_id=row["sample_id"],
                                   split="train" if row["sample_id"] in train else "val",
                                   video=str(video), video_sha256=row["video_sha256"],
                                   frames=row["total_frames"], fps=row["fps"],
                                   annotation=str(annotation) if annotation else None,
                                   task_dir=str(directory), metadata_index=str(index),
                                   csv=str(out), provenance=str(out.with_suffix(".json"))))
    assert len({x["source_id"] for x in result}) == len(result)
    return result, stage_hashes, splits


def remove_time_labels(root, setup, sources, expected_counts=None):
    """Atomic per-file edits with a hash-only deletion journal, no label backups."""
    audit = {"status": "applying", "removed_trajectory_labels": {}, "changes": [], "recover_modified": False}
    # Plan the exact annotation edits before any mutation.
    edits = {}
    for suite in SUITES:
        removed = 0
        for p in (root / suite).rglob("*.json"):
            obj = read(p)
            if not isinstance(obj, dict):
                continue
            cleaned = strip_time_fields(obj)
            if obj != cleaned:
                edits[p] = cleaned
                if isinstance(obj.get("progress"), dict) and obj["progress"].get("policy") == OLD_POLICY:
                    values = obj["progress"]["dense_progress"]
                    assert len(values) == obj["total_frames"]
                    assert all(abs(v - i / (len(values) - 1)) < 1e-7 for i, v in enumerate(values))
                    removed += 1
        audit["removed_trajectory_labels"][suite] = removed
    if expected_counts is not None:
        assert audit["removed_trajectory_labels"] == expected_counts, "Time-label inventory changed; no files edited"
    for p, cleaned in edits.items():
        audit["changes"].append({"path": str(p), "before_sha256": sha(p),
                                 "after_object_sha256": object_hash(cleaned)})
    save(setup / "time-label-removal.json", audit)
    for p, cleaned in edits.items():
        save(p, cleaned)
    annotation_hashes = {r["annotation"]: sha(r["annotation"]) for r in sources if r["annotation"]}
    for index in sorted({r["metadata_index"] for r in sources}):
        p = Path(index)
        before = sha(p)
        rows = json_lines(p)
        clean_rows = []
        for row in rows:
            clean = strip_time_fields(row)
            if clean.get("annotation_path"):
                key = str(inside(p.parent / clean["annotation_path"], p.parent))
                clean["annotation_sha256"] = annotation_hashes[key]
            clean_rows.append(clean)
        if rows != clean_rows:
            save_lines(p, clean_rows)
            audit["changes"].append({"path": str(p), "before_sha256": before, "after_sha256": sha(p)})
    for suite in SUITES:
        p = root / suite / "manifest.json"
        manifest = read(p)
        for task in manifest["tasks"]:
            task["trajectory_index_sha256"] = sha(root / suite / task["trajectory_index"])
        save(p, manifest)
    audit.update(status="completed", completed_at=time.time())
    save(setup / "time-label-removal.json", audit)
    return audit


def db_connect(setup):
    db = sqlite3.connect(setup / "queue.sqlite", timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    return db


def claim(db, worker_id):
    db.execute("BEGIN IMMEDIATE")
    row = db.execute("SELECT source_id,payload FROM jobs WHERE status='pending' ORDER BY frames DESC,source_id LIMIT 1").fetchone()
    if row:
        db.execute("UPDATE jobs SET status='running',worker_id=?,started_at=? WHERE source_id=?", (worker_id, time.time(), row[0]))
    db.commit()
    return json.loads(row[1]) if row else None


def verify_csv(row, protocol_sha):
    import numpy as np
    meta = read(row["provenance"])
    assert meta["protocol_sha256"] == protocol_sha and meta["video_sha256"] == row["video_sha256"]
    assert meta["source_id"] == row["source_id"] and meta["frame_count"] == row["frames"]
    assert sha(row["csv"]) == meta["csv_sha256"]
    values = np.loadtxt(row["csv"], delimiter=",", skiprows=1)
    assert values.shape == (row["frames"], 3)
    assert np.array_equal(values[:, 0], np.arange(row["frames"]))
    delta, pt = values[:, 1], values[:, 2]
    assert np.isfinite(values).all() and np.all(delta >= 0) and delta[0] == 0
    assert pt[0] == 0 and pt[-1] == 1 and np.all(np.diff(pt) >= 0) and np.all((pt >= 0) & (pt <= 1))
    cumulative = np.cumsum(delta, dtype=np.float64)
    assert cumulative[-1] > 1e-8
    np.testing.assert_allclose(pt, cumulative / cumulative[-1], atol=1e-12, rtol=1e-12)
    return meta


def worker(setup, worker_id):
    import torch
    import transformers
    from mem_lite.datasets.prepare_proprimem import whole_video_progress, write_progress
    from mem_lite.datasets.prepare_progress import extract_deltas
    from mem_lite.memory.vision import load_vision_encoder

    gpu = worker_id % 4
    lock = (setup / f"worker-{worker_id}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = read(setup / "protocol.json")
    protocol_sha = sha(setup / "protocol.json")
    visual, processor, vision_hash = load_vision_encoder(protocol["model_path"], "cuda")
    assert not visual.training and not any(p.requires_grad for p in visual.parameters())
    audit = dict(gpu=gpu, worker_id=worker_id, pid=os.getpid(), vision_sha256=vision_hash,
                 revision=MODEL_REVISION, dtype=str(next(visual.parameters()).dtype),
                 torch=torch.__version__, transformers=transformers.__version__,
                 model_class=type(visual).__name__, frozen=True, language_model_loaded=False)
    save(setup / f"initialization-{worker_id}.json", audit)
    db = db_connect(setup)
    count = frames = 0
    started = time.time()
    while row := claim(db, worker_id):
        stamp = time.time()
        state = dict(status="running", gpu=gpu, worker_id=worker_id, pid=os.getpid(), source_id=row["source_id"],
                     completed=count, completed_frames=frames, started_at=started, updated_at=stamp)
        save(setup / f"worker-{worker_id}.json", state)
        try:
            if Path(row["provenance"]).exists():
                verify_csv(row, protocol_sha)
            else:
                assert sha(row["video"]) == row["video_sha256"], "Source video hash mismatch"
                before = Path(row["video"]).stat()
                deltas, info = extract_deltas(row["video"], visual, processor,
                                             batch_size=protocol["batch_size"], max_pixels=protocol["max_pixels"])
                assert len(deltas) == row["frames"] and abs(info["fps"] - row["fps"]) < 1e-3
                after = Path(row["video"]).stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                pt = whole_video_progress(deltas)
                csv_path = Path(row["csv"])
                if csv_path.exists():
                    raise RuntimeError("Unverified pre-existing output CSV; preserve and inspect")
                write_progress(csv_path, range(len(pt)), deltas, pt)
                save(row["provenance"], dict(policy=POLICY, source_id=row["source_id"],
                     video=row["video"], video_sha256=row["video_sha256"], frame_count=len(pt),
                     fps=info["fps"], feature_dim=info["feature_dim"],
                     model_path=protocol["model_path"], model_revision=MODEL_REVISION,
                     vision_sha256=vision_hash, protocol_sha256=protocol_sha,
                     csv_sha256=sha(csv_path), delta_sum=float(deltas.sum()),
                     annotations_used=False, all_original_frames=True, first_pt=float(pt[0]), last_pt=float(pt[-1]),
                     elapsed_seconds=time.time() - stamp, gpu=gpu, worker_id=worker_id))
                verify_csv(row, protocol_sha)
            db.execute("UPDATE jobs SET status='complete',finished_at=?,error=NULL WHERE source_id=?", (time.time(), row["source_id"]))
            count += 1
            frames += row["frames"]
            print(json.dumps(dict(gpu=gpu, source_id=row["source_id"], frames=row["frames"],
                                  seconds=round(time.time() - stamp, 3), completed=count)), flush=True)
        except Exception:
            error = traceback.format_exc()
            db.execute("UPDATE jobs SET status='failed',finished_at=?,error=? WHERE source_id=?", (time.time(), error, row["source_id"]))
            print(error, flush=True)
        db.commit()
    save(setup / f"worker-{worker_id}.json", dict(status="complete", gpu=gpu, worker_id=worker_id, pid=os.getpid(), completed=count,
                                           completed_frames=frames, elapsed_seconds=time.time()-started))


def verify_recover(root, setup, additions_audit=None):
    """Protect every original file; admit only explicitly audited new files."""
    before_path = setup / "recover-before.json"
    before, after = read(before_path), recover_snapshot(root)
    removed = sorted(before.keys() - after.keys())
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    assert not removed and not changed, f"Original Recover files changed: removed={removed}, changed={changed}"
    added = {k: after[k] for k in sorted(after.keys() - before.keys())}
    if additions_audit is None:
        assert not added, "Recover has new files; inspect and explicitly audit additions before finalizing"
    else:
        receipt = read(additions_audit)
        assert receipt["before_snapshot_sha256"] == sha(before_path)
        assert receipt["added_files"] == added, "Recover additions differ from the reviewed audit"
    return dict(existing_files_unchanged=True, original_file_count=len(before),
                file_set_unchanged=not added, added_file_count=len(added),
                additions_audit_sha256=sha(additions_audit) if additions_audit else None)


def finalize(root, setup, sources, recover_additions_audit=None):
    protocol_sha = sha(setup / "protocol.json")
    audits = [read(setup / f"initialization-{g}.json") for g in range(read(setup / "protocol.json")["worker_count"])]
    assert len({a["vision_sha256"] for a in audits}) == 1
    results = {row["source_id"]: verify_csv(row, protocol_sha) for row in sources}
    for path, expected in read(setup / "stage-annotation-hashes.json").items():
        assert object_hash(read(path)) == expected, f"Stage annotation changed: {path}"
    recover_check = verify_recover(root, setup, recover_additions_audit)
    protected_splits = read(setup / "original-splits.json")
    by_index = {}
    for source in sources:
        by_index.setdefault(source["metadata_index"], {})[source["sample_id"]] = source
    for index, records in by_index.items():
        path = Path(index)
        rows = json_lines(path)
        for row in rows:
            source = records[row["sample_id"]]
            row.update(has_progress=True, progress_policy=POLICY,
                       progress_path=str(Path(source["csv"]).relative_to(path.parent)),
                       progress_sha256=results[source["source_id"]]["csv_sha256"],
                       progress_provenance_path=str(Path(source["provenance"]).relative_to(path.parent)),
                       progress_annotation_status="computed_and_verified")
        save_lines(path, rows)
        metadata = path.parent / "metadata.json"
        if metadata.exists():
            obj = read(metadata)
            obj.update(progress_policy=POLICY, progress_annotated_trajectory_count=len(rows),
                       progress_annotation_status="computed_and_verified")
            save(metadata, obj)
    summary = {}
    for suite in SUITES:
        path = root / suite / "manifest.json"
        obj = read(path)
        assert {t["task_id"]: {"train": t["train"], "val": t["val"]} for t in obj["tasks"]} == protected_splits[suite]
        for task in obj["tasks"]:
            task["trajectory_index_sha256"] = sha(root / suite / task["trajectory_index"])
        rows = [r for r in sources if r["suite"] == suite]
        counts = dict(Counter(r["split"] for r in rows))
        summary[suite] = dict(trajectories=len(rows), frames=sum(r["frames"] for r in rows), **counts)
        obj["progress_update"] = dict(policy=POLICY, field="metadata.progress_path -> CSV pt",
                                      model="Qwen/Qwen3.5-4B", revision=MODEL_REVISION,
                                      protocol_sha256=protocol_sha, verified_trajectories=len(rows),
                                      generated_at=time.time(), frame_index_base=0)
        save(path, obj)
    report = dict(status="completed_awaiting_evaluation_notification", suites=summary,
                  trajectories=len(sources), frames=sum(r["frames"] for r in sources),
                  all_labels_verified=True, recover_unchanged=recover_check["file_set_unchanged"],
                  recover_existing_files_unchanged=True, recover_verification=recover_check,
                  stage_annotations_unchanged=True,
                  existing_splits_preserved=True, evaluation_started=False, protocol_sha256=protocol_sha,
                  vision_sha256=audits[0]["vision_sha256"], completed_at=time.time(),
                  finalization_code_sha256=sha(Path(__file__)))
    save(setup / "verification.json", report)
    return report


def run(root, setup):
    setup.mkdir(parents=True, exist_ok=True)
    lock = (setup / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert root.resolve() == DEFAULT_ROOT
    assert read(Path('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json'))['all_official_hashes_verified']
    assert not (setup / "queue.sqlite").exists(), "Existing queue: inspect processes and resume explicitly"
    assert not list(root.rglob("*.download-part")), "Dataset download still active"
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    assert not gpu, "Existing GPU work: do not occupy its devices"
    started = time.time()
    save(setup / "status.json", dict(status="preparing", pid=os.getpid(), started_at=started, evaluation_started=False))
    sources, stages, splits = inventory(root)
    assert len(sources) == 1904 and sum(r["frames"] for r in sources) == 1651148
    save(setup / "input-index.json", sources)
    save(setup / "stage-annotation-hashes.json", stages)
    save(setup / "original-splits.json", splits)
    save(setup / "recover-before.json", recover_snapshot(root))
    code = Path(__file__).resolve().parents[2]
    files = [Path(__file__), code / "mem_lite/datasets/prepare_proprimem.py",
             code / "mem_lite/datasets/prepare_progress.py", code / "mem_lite/memory/vision.py"]
    save(setup / "protocol.json", dict(policy=POLICY, model_path=str(MODEL), model_revision=MODEL_REVISION,
         input_view="main_view", source_frames="all original decoded frames; no FPS subsampling",
         feature="mean of post-merger spatial tokens, FP32", delta="adjacent-frame raw L2, FP32",
         normalization="FP64 cumulative delta divided by final cumulative delta", first_delta=0,
         min_total_delta=1e-8, zero_delta_fallback=None, batch_size=16, max_pixels=262144, min_pixels=4096,
         frozen_vision_dtype="torch.bfloat16", attention="sdpa", gpus=[0,1,2,3], workers_per_gpu=4, worker_count=16,
         suites=list(SUITES), recover_excluded=True, stages_used=False, preserve_splits=True,
         evaluation_started=False, source_code_sha256={str(p.relative_to(code)):sha(p) for p in files}))
    audit = remove_time_labels(root, setup, sources, expected_counts={"State":441,"Order":50,"Count":32})
    assert audit["removed_trajectory_labels"] == {"State":441,"Order":50,"Count":32}
    db = db_connect(setup)
    db.execute("CREATE TABLE jobs(source_id TEXT PRIMARY KEY,payload TEXT,frames INTEGER,status TEXT,worker_id INTEGER,started_at REAL,finished_at REAL,error TEXT)")
    db.executemany("INSERT INTO jobs(source_id,payload,frames,status) VALUES(?,?,?,'pending')", [(r['source_id'],json.dumps(r),r['frames']) for r in sources])
    db.commit()
    processes = []
    for worker_id in range(16):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker_id % 4), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        with (setup / f"worker-{worker_id}.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "experiments.proprimem_visual_pt_20261113.relabel", "worker",
                                     "--setup",str(setup),"--worker-id",str(worker_id)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
        processes.append(proc)
    save(setup / "worker-pids.json", {str(i):p.pid for i,p in enumerate(processes)})
    while True:
        counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
        frames = db.execute("SELECT coalesce(sum(frames),0) FROM jobs WHERE status='complete'").fetchone()[0]
        state = dict(status="running", counts=counts, completed_frames=frames, total_frames=1651148,
                     total_trajectories=1904, pid=os.getpid(), started_at=started, updated_at=time.time(),
                     workers={str(i):p.poll() for i,p in enumerate(processes)}, evaluation_started=False)
        save(setup / "status.json", state)
        if all(p.poll() is not None for p in processes):
            break
        time.sleep(10)
    if counts != {"complete":1904} or any(p.returncode for p in processes):
        state.update(status="needs_review", failures=list(db.execute("SELECT source_id,error FROM jobs WHERE status!='complete'")))
        save(setup / "status.json", state)
        raise RuntimeError("Some labels failed; completed labels retained, no evaluation started")
    state.update(status="verifying")
    save(setup / "status.json", state)
    result = finalize(root, setup, sources)
    save(setup / "status.json", dict(result, started_at=started, elapsed_seconds=time.time()-started))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "worker", "finalize"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--recover-additions-audit", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        try:
            run(args.root, args.setup)
        except Exception:
            save(args.setup / "controller-error.json", dict(error=traceback.format_exc(), time=time.time(), evaluation_started=False))
            raise
    elif args.command == "worker":
        worker(args.setup, args.worker_id)
    else:
        with (args.setup / "pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = read(args.setup / "status.json")
            result = finalize(args.root, args.setup, read(args.setup / "input-index.json"),
                              args.recover_additions_audit)
            save(args.setup / "status.json", dict(result, started_at=previous.get("started_at")))
            print(json.dumps(result))
