# ruff: noqa: RUF001
"""Convert curated suite annotations into progress labels without editing videos.

python -m qwen_training.datasets.proprimem.prepare --root .../ProPriMem --out .../labels
"""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from qwen_training.datasets.proprimem.policies import POLICIES
from qwen_training.datasets.rmbench.labels import MAX_COUNT, SAMPLE_FPS, sample_frames, step_of
from qwen_training.datasets.rmbench.tasks import TASKS


def annotation_events(annotation, policy):
    """Return completion events (step text, source frame), distinguishing current/completed states."""
    schema, original, _ = policy
    stages = annotation["stages"]
    events, previous = [], Counter()
    fixed = {
        "drawers": [f"{verb} the {level} drawer." for verb, levels in
                    [("Open", ("bottom", "middle", "top")), ("Close", ("top", "middle", "bottom"))]
                    for level in levels],
        "swap_confirm": ["Move the right block to the middle mat.", "Press the button once.",
                         "Move the left block to the right mat.", "Press the button once.",
                         "Move the right block from the middle to the left mat.", "Press the button once."],
        "xylophone": ["Pick up the mallet with the left hand.",
                      *[f"Strike key {i} from the left." for i in range(1, 5)],
                      "Transfer the mallet to the right hand.",
                      *[f"Strike key {i} from the left." for i in range(5, 9)]],
    }
    if schema in fixed:
        if len(stages) - 1 != len(fixed[schema]):
            raise ValueError("unexpected number of completed-state boundaries")
        return [(text, stage["start_frame"]) for text, stage in zip(fixed[schema], stages[1:], strict=True)]

    for position, stage in enumerate(stages):
        text = stage["state_description"]
        frame = stage["start_frame"]
        if schema == "rmbench":
            clean = re.sub(r"（第\d+次）", "", text)
            step = step_of(clean, position, TASKS[original])
            events.append((TASKS[original].steps[step], stage["end_frame"] - 1))
            continue
        if schema == "libero":
            if "全部子任务已完成" in text:
                continue
            action = re.sub(r"（第\d+个子任务）", "", text)
            events.append((action, stage["end_frame"]))
            continue
        if schema == "cube_place":
            if text.lower().startswith("place"):
                events.append(("Pick up the cube and place it onto the target once.", stage["end_frame"] - 1))
            elif "press" in text.lower():
                events.append(("Press the button to stop.", stage["end_frame"] - 1))
            elif not text.lower().startswith("pick up"):
                raise ValueError(f"unrecognized cube action: {text}")
            continue
        if schema == "cube_swing":
            lower = text.lower()
            action = "Pick up the cube." if lower.startswith("pick up") else \
                "Move the cube right then left once." if "left-side target" in lower else \
                "Put the cube back on the table." if lower.startswith("put") else \
                "Press the button to stop." if "press" in lower else None
            if action:
                events.append((action, stage["end_frame"] - 1))
            elif "right-side target" not in lower:
                raise ValueError(f"unrecognized swing action: {text}")
            continue
        if text == "无阶段性事件":
            continue
        current = Counter()
        if schema in ("source_segments", "completed_counts"):
            unit = "段" if schema == "source_segments" else "次"
            for action, count in re.findall(r"([^；：]+)（(\d+)" + unit + r"）", text):
                current[action.strip()] = int(count)
            if not current:
                raise ValueError(f"cannot parse cumulative source state: {text}")
            if original:
                mapping = {"将两节电池放入电池槽": 0, "翻转一节电池尝试下一种方向组合": 1,
                           "按按钮检查排列": 0, "交换两块积木": 1}
                current = Counter({TASKS[original].steps[mapping[action]]: n for action, n in current.items()})
        else:
            patterns = {
                "number_buttons": [("Press the left red button once.", r"左键已按(\d+)次"),
                                   ("Press the middle red button once.", r"中键已按(\d+)次"),
                                   ("Press the blue confirmation button once.", r"蓝键已按(\d+)次")],
                "blue_button": [("Press the blue button once.", r"已按蓝色按钮(\d+)次")],
                "drum": [("Pick up the drumstick.", r"拿起鼓槌"),
                         ("Strike the drum once.", r"敲鼓(\d+)次"),
                         ("Return the drumstick to the table.", r"放回桌面")],
                "soap": [("Pick up the cloth.", r"拿起清洁布"),
                         ("Move the cloth under the dispenser.", r"移至出液口下"),
                         ("Press the dispenser pump once.", r"按压泵头(\d+)次"),
                         ("Return the cloth to the table.", r"放回桌面")],
                "season": [("Pick up the seasoning bottle.", r"拿起调料瓶"),
                           ("Season the skewers once.", r"完成(\d+)次撒调料"),
                           ("Return the seasoning bottle to the table.", r"放回桌面")],
                "keypad": [("Press the next required number key.", r"已完成按键(\d+)次")],
            }
            for action, pattern in patterns[schema]:
                match = re.search(pattern, text)
                if match:
                    current[action] = int(match.group(1)) if match.lastindex else 1
            if not current:
                raise ValueError(f"unrecognized completed state: {text}")
        changed = [(action, n - previous[action]) for action, n in current.items() if n != previous[action]]
        if len(changed) != 1 or changed[0][1] != 1:
            raise ValueError(f"boundary must mark exactly one completion: {text}")
        events.append((changed[0][0], frame))
        previous.update({action: n - previous[action] for action, n in current.items()})
    return events


def convert(annotation, row, metadata, task_path):
    total, stages = annotation["total_frames"], annotation["stages"]
    if annotation.get("frame_index_base", 0) != 0 \
            or annotation.get("frame_range_convention") != "[start_frame, end_frame)":
        raise ValueError("unsupported frame convention")
    cursor = 0
    for stage in stages:
        if stage["start_frame"] != cursor or not cursor < stage["end_frame"] <= total:
            raise ValueError("stage coverage is not contiguous")
        cursor = stage["end_frame"]
    if cursor != total or len(stages) != annotation["num_stages"]:
        raise ValueError("stage coverage does not match video")
    policy = POLICIES[metadata["task_id"]]
    events = annotation_events(annotation, policy)
    if not events or any(not 0 <= frame < total for _, frame in events):
        raise ValueError("missing or out-of-range completion")
    if any(b[1] <= a[1] for a, b in pairwise(events)):
        raise ValueError("completion boundaries are not strictly increasing")
    steps = list(dict.fromkeys(text for text, _ in events))
    counts, trace, start = [0] * len(steps), [], 0
    for action, frame in events:
        j = steps.index(action)
        counts[j] += 1
        trace.append(dict(step=j, instance=counts[j], start=start, end=frame, annotation=action))
        start = frame + 1
    if max(counts) > MAX_COUNT:
        raise ValueError(f"count exceeds model capacity {MAX_COUNT}")
    frames = []
    for index in sample_frames(total, annotation["fps"], SAMPLE_FPS):
        done = [sum(t["step"] == j and index >= t["end"] for t in trace) for j in range(len(steps))]
        active = next((t for t in trace if index <= t["end"]), None)
        elapsed = (index - active["start"]) / max(1, active["end"] - active["start"] + 1) \
            if active and index < active["end"] else 0
        frames.append(dict(index=index, stage=active["step"] if active else len(steps), counts=done,
                           progress=min(1.0, (sum(done) + max(0, elapsed)) / len(trace))))
    # Completion frames are exact source boundaries, not arbitrary stage midpoints.
    demo = [dict(step=t["step"], frame=t["end"]) for t in trace if t["instance"] == 1]
    if policy[1] in ("battery_try", "blocks_ranking_try"):
        demo.append(dict(step=None, frame=trace[-1]["end"]))
    views = row["video_views"]
    instruction = row.get("source_instruction") or metadata["task_name"]
    return dict(task="", episode=row["episode_index"], instruction=instruction,
                suite=metadata["suite"], task_id=metadata["task_id"], steps=steps, step_counts=counts,
                trace=trace, demo=demo, frames=frames, total_frames=total, source_fps=annotation["fps"],
                source={name: str(task_path / view["video_path"]) for name, view in views.items()},
                cameras=list(views), sample_id=row["sample_id"],
                label_unit="source_action_segment" if policy[0] == "source_segments" else "annotated_action",
                progress_label="derived event-normalized interpolation")


def prepare(root, out, seed=0):
    root, out = Path(root), Path(out)
    if (out / "_splits.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing split: {out}")
    out.mkdir(parents=True, exist_ok=True)
    groups, report = defaultdict(list), {"tasks": [], "excluded": [], "seed": seed}
    for path in sorted(root.glob("*/*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta["suite"] not in ("Count", "Order", "State"):
            continue
        rel = path.parent.relative_to(root)
        rows = [json.loads(line) for line in (path.parent / meta["trajectory_index"]).read_text().splitlines() if line]
        report["tasks"].append({"task_id": meta["task_id"], "suite": meta["suite"], "trajectories": len(rows)})
        for row in rows:
            try:
                if not row.get("has_annotation"):
                    raise ValueError("missing_annotation")
                record = convert(json.loads((path.parent / row["annotation_path"]).read_text()), row, meta, rel)
                groups[(str(rel), tuple(record["steps"]))].append((record, meta))
            except (ValueError, KeyError, OSError) as error:
                report["excluded"].append(dict(task=str(rel), sample_id=row["sample_id"], reason=str(error)))
    splits, policies = {}, {}
    for (task, steps), items in sorted(groups.items()):
        if len(items) < 4:
            report["excluded"] += [dict(task=task, sample_id=r["sample_id"],
                                        reason="insufficient_independent_demo_train_val")
                                   for r, _ in items]
            continue
        variants = sum(key[0] == task for key in groups)
        key = task if variants == 1 else task + "__" + hashlib.sha256(json.dumps(steps).encode()).hexdigest()[:8]
        ordered = sorted(items, key=lambda item: item[0]["episode"])
        ids = [r["episode"] for r, _ in ordered]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate episode ids in {key}")
        meta = items[0][1]
        if meta["source_collection"] == "RMBench" and ids == list(range(50)):
            split = dict(pool=list(range(8)), eval_demo=8, train=list(range(9, 45)), val=list(range(45, 50)))
        else:
            rng = random.Random(f"{seed}:{key}")
            rng.shuffle(ids)
            pool = min(8, max(1, len(ids) // 10))
            val = max(1, math.floor(len(ids) * 0.1))
            split = dict(pool=ids[:pool], eval_demo=ids[pool], val=ids[pool + 1:pool + 1 + val],
                         train=ids[pool + 1 + val:])
        splits[key] = split
        schema, original, safe = POLICIES[meta["task_id"]]
        policies[key] = dict(steps=list(steps), kind="scene_count" if any(max(r["step_counts"]) > 1 for r, _ in items)
                             else "fixed", rewind_safe=safe, source_task=original, suite=meta["suite"],
                             task_id=meta["task_id"], annotation_schema=schema,
                             label_unit=items[0][0]["label_unit"])
        folder = out / key
        folder.mkdir(parents=True, exist_ok=True)
        for record, _ in items:
            record["task"] = key
            (folder / f"episode{record['episode']}.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    if not splits:
        raise ValueError("no trainable annotated groups found")
    for name, value in [("_splits.json", splits), ("_tasks.json", policies), ("report.json", report)]:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return report, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report, splits = prepare(args.root, args.out, args.seed)
    print(json.dumps({"groups": len(splits), "train": sum(len(s["train"]) for s in splits.values()),
                      "val": sum(len(s["val"]) for s in splits.values()), "excluded": len(report["excluded"])}))


if __name__ == "__main__":
    main()
