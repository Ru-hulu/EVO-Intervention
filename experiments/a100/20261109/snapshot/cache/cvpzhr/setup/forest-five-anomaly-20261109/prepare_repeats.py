"""Build an additional frozen-frame training cache; never modify source caches."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

ROOT = Path("/cache/cvpzhr")
SOURCE = ROOT / "datasets/forest-proprimem-5fps-20261030"
OUTPUT = ROOT / "datasets/forest-proprimem-5fps-repeat5-seed42-20261031"
SETUP = ROOT / "setup/forest-repeat5-20261031"
CODE = ROOT / "code/forest-train3-20261031-0009"
MODEL = ROOT / "models/Qwen3.5-4B"
VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CONTRACT = dict(version="frozen-frame-repeat-v1", seed=42, sample_fps=5,
                anchors_per_episode=5, repeat_frames_min=10, repeat_frames_max=25,
                anchors="uniform without replacement among original sampled frames; endpoints allowed",
                duration="independent uniform integer repeat count, inclusive [10,25]",
                scope="858 independent training episodes; one augmented version each",
                pt="copy source PT unchanged, including during holds; no renormalization",
                timeline="original frame position plus inserted elapsed time times source fps",
                views="repeat each view's own anchor image synchronously; preserve missing views",
                history="rebuild causal StreamForest with original frozen vision weights/config",
                images="read-only reuse via hard links; repeated rows reference the same image files",
                validation="unchanged original validation paths", training_started=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_at(directory):
    return [json.loads(x) for x in (Path(directory) / "samples.jsonl").read_text().splitlines() if x.strip()]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_fps(summary):
    return Fraction(str(summary["source"]["fps"])).limit_denominator(1000000)


def number(value):
    return int(value) if value.denominator == 1 else float(value)


def choose_holds(rows, episode_id):
    if len(rows) < CONTRACT["anchors_per_episode"]:
        raise ValueError(f"Too few observations: {episode_id}")
    rng = random.Random(f'{CONTRACT["seed"]}:{episode_id}:frozen-frame-repeat-v1')
    anchors = sorted(rng.sample(range(len(rows)), CONTRACT["anchors_per_episode"]))
    return [dict(source_sample_index=i, source_frame_index=rows[i]["frame_index"],
                 repeat_frames=rng.randint(CONTRACT["repeat_frames_min"], CONTRACT["repeat_frames_max"]))
            for i in anchors]


def expand(rows, holds, fps):
    """Keep original sampling positions; shift all later observations by hold duration."""
    hold_map = {h["source_sample_index"]: h for h in holds}
    added = 0
    sample_index = 0
    for source_index, original in enumerate(rows):
        count = hold_map.get(source_index, {}).get("repeat_frames", 0)
        for repeat_index in range(count + 1):
            position = Fraction(original["frame_index"]) + Fraction(added, CONTRACT["sample_fps"]) * fps
            yield dict(frame_index=number(position), source_frame_index=original["frame_index"],
                       source_sample_index=source_index, sample_index=sample_index,
                       timestamp_seconds=float(position / fps), repeat_index=repeat_index,
                       is_repeat=repeat_index > 0, pt=original["pt"], images=original["images"].copy())
            sample_index += 1
            if repeat_index < count:
                added += 1


def make_plan():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "index.json"
    if target.exists():
        raise FileExistsError(f"Plan already exists: {target}")
    paths = sorted(read_json(SOURCE / "train.sample_files.json"))
    validation = read_json(SOURCE / "validation.sample_files.json")
    if len(paths) != 858 or len(validation) != 213:
        raise ValueError("Unexpected input split; review before proceeding")
    protected = [SOURCE / "train.sample_files.json", SOURCE / "validation.sample_files.json",
                 SOURCE / "index.json", ROOT / "setup/train-4gpu.json",
                 ROOT / "runs/forest-proprimem-e3-20261031-0009/config.json"]
    protected += sorted(CODE.rglob("*.py")) + sorted(CODE.rglob("*.json"))
    protected_hashes = {str(p): sha(p) for p in protected if "__pycache__" not in str(p)}
    episodes = []
    common = None
    fps_counts = Counter()
    for directory in paths:
        directory = Path(directory)
        summary = read_json(directory / "summary.json")
        rows = rows_at(directory)
        if len(rows) != summary["num_samples"] or summary["source"]["split"] != "train":
            raise ValueError(f"Invalid source: {directory}")
        if common is None:
            common = summary["history_source"]
        if summary["history_source"] != common:
            raise ValueError("Mixed source history settings")
        fps = source_fps(summary)
        fps_counts[str(fps)] += 1
        unit = summary["source"]["unit_id"]
        if unit != str(directory.relative_to(SOURCE / "cache/train")):
            raise ValueError("Unexpected episode path")
        holds = choose_holds(rows, unit)
        expanded = list(expand(rows, holds, fps))
        if any(b["frame_index"] <= a["frame_index"] for a, b in zip(expanded, expanded[1:])):
            raise ValueError("Nonmonotonic expanded timeline")
        episodes.append(dict(episode_id=unit, source_dir=str(directory),
                             output_dir=str(OUTPUT / "cache/train" / unit),
                             source_samples_sha256=sha(directory / "samples.jsonl"),
                             source_summary_sha256=sha(directory / "summary.json"),
                             source_samples=len(rows), augmented_samples=len(expanded),
                             source_fps_numerator=fps.numerator, source_fps_denominator=fps.denominator,
                             added_samples=sum(h["repeat_frames"] for h in holds), holds=holds))
    total = sum(e["augmented_samples"] for e in episodes)
    if sum(e["source_samples"] for e in episodes) != 126441:
        raise ValueError("Unexpected training sample total")
    payload = dict(contract=CONTRACT, source=str(SOURCE), output=str(OUTPUT), code=str(CODE),
                   model=str(MODEL), history_source=common, protected_hashes=protected_hashes,
                   source_fps_counts=dict(fps_counts), episodes=episodes)
    plan = dict(**payload, plan_sha256=canonical_hash(payload), created_at=now(),
                original_samples=126441, augmented_samples=total, added_samples=total-126441,
                total_anchors=5*len(episodes), validation_paths=validation)
    write_json(target, plan)
    write_json(OUTPUT / "planned.train.sample_files.json", [e["output_dir"] for e in episodes])
    write_json(OUTPUT / "validation.sample_files.json", validation)
    write_json(SETUP / "plan-summary.json",
               {k: plan[k] for k in ("plan_sha256","original_samples","augmented_samples","added_samples","total_anchors","source_fps_counts")})
    return plan


def get_plan():
    plan = read_json(OUTPUT / "index.json")
    payload = {k: plan[k] for k in ("contract","source","output","code","model","history_source",
                                   "protected_hashes","source_fps_counts","episodes")}
    if plan["plan_sha256"] != canonical_hash(payload) or plan["contract"] != CONTRACT:
        raise ValueError("Plan or generation contract changed")
    return plan


def check_source(episode):
    directory = Path(episode["source_dir"])
    for name, key in (("samples.jsonl","source_samples_sha256"), ("summary.json","source_summary_sha256")):
        if sha(directory / name) != episode[key]:
            raise ValueError(f"Source changed: {directory / name}")


def encode_raw(visual, processor, image, settings):
    """Same frozen ViT/merger operation as QwenFrameEncoder, before temporal positions."""
    import torch
    inputs = processor(images=[image], return_tensors="pt", max_pixels=settings["max_pixels"],
                       min_pixels=settings["min_pixels"])
    parameter = next(visual.parameters())
    grid = inputs["image_grid_thw"].to(parameter.device)
    pixels = inputs["pixel_values"].to(device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad(), torch.autocast(parameter.device.type, enabled=False):
        features = visual(pixels, grid_thw=grid, return_dict=True).pooler_output
    if len(grid) != 1 or int(grid[0, 0]) != 1:
        raise ValueError("Expected one image")
    if not torch.isfinite(features).all().item():
        raise ValueError("Nonfinite visual features")
    merge = visual.config.spatial_merge_size
    return features, int(grid[0, 1]) // merge, int(grid[0, 2]) // merge


def position_features(features, height, width, frame_index):
    from mem_lite.memory.vision import spatiotemporal_positions
    positions = spatiotemporal_positions(height, width, frame_index, features.shape[-1], features.device)
    return features + positions.to(features.dtype)


def build_episode(plan, episode, visual, processor):
    import torch
    from PIL import Image
    from mem_lite.memory.manager import MemoryManager

    check_source(episode)
    output, source = Path(episode["output_dir"]), Path(episode["source_dir"])
    if (output / "summary.json").exists():
        return verify_episode(plan, episode)
    if output.exists():
        raise FileExistsError(f"Unexpected incomplete final directory: {output}")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        ownership = read_json(temporary / ".augmentation-owned.json")
        if ownership != dict(plan_sha256=plan["plan_sha256"], episode_id=episode["episode_id"]):
            raise ValueError("Unexpected partial directory; not removing it")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_json(temporary / ".augmentation-owned.json",
               dict(plan_sha256=plan["plan_sha256"], episode_id=episode["episode_id"]))
    (temporary / "history").mkdir()
    summary, rows = read_json(source / "summary.json"), rows_at(source)
    settings = summary["history_source"]
    memory = MemoryManager(summary["history_dim"], **settings["memory_config"])
    fps = source_fps(summary)
    previous = None
    previous_source = None
    native_features = None
    token_max = 0
    history_bytes = 0
    started = time.monotonic()
    with (temporary / "samples.jsonl").open("w") as samples:
        for row in expand(rows, episode["holds"], fps):
            source_index = row["source_sample_index"]
            if source_index != previous_source:
                original = rows[source_index]
                for view in VIEWS:
                    relative = original["images"][view]
                    if relative is not None:
                        target = temporary / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source / relative, target)
                with Image.open(source / original["images"]["cam_high"]) as im:
                    native_features, height, width = encode_raw(visual, processor, im.convert("RGB"), settings)
                previous_source = source_index
            # Save the state BEFORE the current observation is incorporated.
            history = memory.get_memory_tokens()
            history_path = None
            tokens = 0 if history is None else len(history)
            if tokens > summary["history_capacity"]:
                raise ValueError("History budget exceeded")
            if history is not None:
                history_path = f'history/{row["sample_index"]:08d}.pt'
                torch.save(dict(tokens=history.cpu(), through_frame=previous), temporary / history_path)
                history_bytes += (temporary / history_path).stat().st_size
            row["history"] = history_path
            row["history_token_count"] = tokens
            samples.write(json.dumps(row, ensure_ascii=False) + "\n")
            token_max = max(token_max, tokens)
            # Repeated pixels reuse only the unpositioned frozen visual features.
            # New temporal positions and all memory updates are recomputed on every repeat.
            frame = position_features(native_features, height, width, row["frame_index"])
            memory.update(frame)
            previous = row["frame_index"]
    if row["sample_index"] + 1 != episode["augmented_samples"]:
        raise ValueError("Unexpected output sample count")
    write_json(temporary / "augmentation.json", dict(plan_sha256=plan["plan_sha256"], **episode))
    summary.update(num_samples=episode["augmented_samples"],
                   augmentation=dict(**CONTRACT, plan_sha256=plan["plan_sha256"],
                                     original_episode_id=episode["episode_id"],
                                     original_cache=str(source), source_samples=episode["source_samples"],
                                     added_samples=episode["added_samples"],
                                     frame_index_unit="source-fps frame-equivalent position; may be fractional",
                                     frame_index_note="source_frame_index retains original integer frame number"),
                   generation=dict(completed_at=now(), seconds=time.monotonic()-started,
                                   history_bytes=history_bytes, max_history_tokens=token_max,
                                   raw_visual_forward_calls=episode["source_samples"],
                                   memory_updates=episode["augmented_samples"]))
    write_json(temporary / "summary.json", summary)
    temporary.rename(output)
    return verify_episode(plan, episode)


def verify_episode(plan, episode):
    """Independently check labels, view identity, hold counts, timing and causal snapshots."""
    import torch
    check_source(episode)
    directory, source = Path(episode["output_dir"]), Path(episode["source_dir"])
    summary = read_json(directory / "summary.json")
    original_summary = read_json(source / "summary.json")
    rows, original = rows_at(directory), rows_at(source)
    if summary["augmentation"]["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Output belongs to a different plan")
    if summary["history_source"] != original_summary["history_source"]:
        raise ValueError("History settings changed")
    if len(rows) != episode["augmented_samples"] or summary["num_samples"] != len(rows):
        raise ValueError("Wrong sample count")
    holds = {h["source_sample_index"]: h["repeat_frames"] for h in episode["holds"]}
    frequencies = Counter(r["source_sample_index"] for r in rows)
    if frequencies != Counter({i:1+holds.get(i,0) for i in range(len(original))}):
        raise ValueError("Incorrect repetition count")
    if [r["source_sample_index"] for r in rows if not r["is_repeat"]] != list(range(len(original))):
        raise ValueError("Original observations removed, duplicated or reordered")
    fps = source_fps(original_summary)
    previous = None
    added = 0
    checked_images = set()
    checked_histories = 0
    selected = {0, 1, len(rows)-1}
    for i, r in enumerate(rows):
        if r["repeat_index"] == 1 or (r["is_repeat"] and (i+1 == len(rows) or not rows[i+1]["is_repeat"])):
            selected.update(j for j in (i-1,i,i+1) if 0 <= j < len(rows))
    for i, r in enumerate(rows):
        index = r["source_sample_index"]
        src = original[index]
        if r["is_repeat"]:
            added += 1
        expected_position = Fraction(src["frame_index"]) + Fraction(added, 5) * fps
        if (r["sample_index"] != i or r["source_frame_index"] != src["frame_index"]
            or r["pt"] != src["pt"] or r["images"] != src["images"]
            or abs(r["frame_index"]-float(expected_position)) > 1e-8
            or abs(r["timestamp_seconds"]-float(expected_position/fps)) > 1e-8):
            raise ValueError(f"Label/view/timestamp mismatch: {directory}, row {i}")
        if previous is not None and r["frame_index"] <= previous:
            raise ValueError("Time does not advance")
        if r["is_repeat"] and (i == 0 or rows[i-1]["source_sample_index"] != index
                               or r["repeat_index"] != rows[i-1]["repeat_index"]+1
                               or abs(r["timestamp_seconds"]-rows[i-1]["timestamp_seconds"]-0.2)>1e-8):
            raise ValueError("Broken repeat block")
        for view, relative in r["images"].items():
            if relative is not None and relative not in checked_images:
                a, b = (directory / relative).stat(), (source / relative).stat()
                if (a.st_dev,a.st_ino,a.st_size) != (b.st_dev,b.st_ino,b.st_size):
                    raise ValueError(f"Image differs from source: {view}/{relative}")
                checked_images.add(relative)
        if (r["history"] is None) != (i == 0):
            raise ValueError("Only the first observation may have empty history")
        if r["history"] is not None:
            path = directory / r["history"]
            if not path.is_file() or path.stat().st_size < 100:
                raise ValueError(f"Missing snapshot: {path}")
            if i in selected:
                item = torch.load(path, map_location="cpu", weights_only=True)
                tokens = item["tokens"]
                if (item["through_frame"] != previous or tokens.ndim != 2
                    or tokens.shape[1] != summary["history_dim"] or len(tokens)>summary["history_capacity"]
                    or len(tokens) != r["history_token_count"] or tokens.dtype != torch.bfloat16
                    or not torch.isfinite(tokens).all().item()):
                    raise ValueError(f"Invalid causal snapshot: {path}")
                checked_histories += 1
        previous = r["frame_index"]
    if len(list((directory/"history").glob("*.pt"))) != len(rows)-1:
        raise ValueError("Extra or missing history files")
    return dict(episode_id=episode["episode_id"], samples=len(rows), added_samples=added,
                anchors=len(holds), checked_histories=checked_histories,
                checked_images=len(checked_images), history_bytes=summary["generation"]["history_bytes"],
                seconds=summary["generation"]["seconds"], max_history_tokens=summary["generation"]["max_history_tokens"])


def assigned_episodes(plan, worker, workers):
    groups, loads = [[] for _ in range(workers)], [0]*workers
    for e in sorted(plan["episodes"], key=lambda x:(-x["augmented_samples"],x["episode_id"])):
        i = min(range(workers), key=lambda x:loads[x])
        groups[i].append(e)
        loads[i] += e["augmented_samples"]
    return groups[worker]


def run_worker(worker, workers, limit=None):
    import torch
    from mem_lite.memory.vision import load_vision_encoder
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.08)
    plan = get_plan()
    assigned = assigned_episodes(plan, worker, workers)
    if limit is not None:
        assigned = assigned[:limit]
    status_path = SETUP / f"worker-{worker}.json"
    state = dict(pid=os.getpid(), worker=worker, workers=workers, total=len(assigned), completed=0,
                 samples=0, phase="loading_vision", updated_at=now())
    write_json(status_path, state)
    started = time.monotonic()
    try:
        visual, processor, digest = load_vision_encoder(MODEL, "cuda")
        if digest != plan["history_source"]["vision_sha256"]:
            raise ValueError("Vision weights differ from original cache")
        for episode in assigned:
            if shutil.disk_usage(OUTPUT).free < 200*2**30:
                raise RuntimeError("Less than 200 GiB free; stopping this augmentation worker")
            state.update(phase="building", current=episode["episode_id"], updated_at=now())
            write_json(status_path, state)
            result = build_episode(plan, episode, visual, processor)
            state["completed"] += 1
            state["samples"] += result["samples"]
            state.update(last=result, elapsed_seconds=time.monotonic()-started, updated_at=now())
            write_json(status_path, state)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        state.update(phase="completed", current=None, updated_at=now())
        write_json(status_path, state)
    except Exception as exc:
        state.update(phase="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
        write_json(status_path, state)
        raise


def finalize():
    import torch
    torch.set_num_threads(2)
    plan = get_plan()
    results = [verify_episode(plan, e) for e in plan["episodes"]]
    for path, expected in plan["protected_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Original file changed: {path}")
    if read_json(OUTPUT/"validation.sample_files.json") != plan["validation_paths"]:
        raise ValueError("Validation split changed")
    paths = [e["output_dir"] for e in plan["episodes"]]
    # Only publish the training list after all 858 episodes pass final checks.
    write_json(OUTPUT / "train.sample_files.json", paths)
    report = dict(status="completed", completed_at=now(), plan_sha256=plan["plan_sha256"],
                  episodes=len(results), original_samples=plan["original_samples"],
                  augmented_samples=sum(r["samples"] for r in results),
                  added_samples=sum(r["added_samples"] for r in results),
                  anchors=sum(r["anchors"] for r in results),
                  checked_histories=sum(r["checked_histories"] for r in results),
                  checked_images=sum(r["checked_images"] for r in results),
                  history_bytes=sum(r["history_bytes"] for r in results),
                  max_history_tokens=max(r["max_history_tokens"] for r in results),
                  labels_unchanged=True, views_synchronized=True, histories_causal=True,
                  original_files_unchanged=True, validation_unchanged=True, training_started=False)
    if report["augmented_samples"] != plan["augmented_samples"] or report["anchors"] != 4290:
        raise ValueError("Final totals differ from plan")
    write_json(OUTPUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan","worker","verify"))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args=parser.parse_args()
    if args.command=="plan":
        p=make_plan()
        print(json.dumps({k:p[k] for k in ("plan_sha256","original_samples","augmented_samples","added_samples","total_anchors")}, indent=2))
    elif args.command=="worker":
        run_worker(args.worker,args.workers,args.limit)
    else:
        finalize()


if __name__=="__main__":
    main()

