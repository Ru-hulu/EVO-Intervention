"""Generate standalone concise stage supervision; source annotations/PT stay unchanged."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json

ROOT = Path("/cache/cvpzhr/datasets/failure_recover/pour-water-into-three-cups-with-repeated-pouring")
ANNOTATIONS = ROOT / "videos/main_view/annotation"
NAMES = {"left": "左杯", "middle": "中杯", "right": "右杯"}
NUMBERS = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五"}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def history_text(counts):
    active = [target for target in counts if counts[target]]
    if len(active) == 3:
        return "三个杯子均已倒过水"
    if not active:
        return ""
    if len(active) > 1 and len({counts[t] for t in active}) == 1:
        return "、".join(NAMES[t] for t in active) + "各已倒水" + NUMBERS[counts[active[0]]] + "次"
    parts = []
    for i, target in enumerate(active):
        parts.append(NAMES[target] + ("已倒水" if i == 0 else "") + NUMBERS[counts[target]] + "次")
    return "、".join(parts)

def label_text(action, target, counts, first_put_down):
    history = history_text(counts)
    if action == "idle":
        if len(first_put_down) == 3:
            return "三个杯子均已倒过水，水瓶已放下，任务完成。"
        return (history or "尚未倒水") + "，当前空闲。"
    if action == "put_down":
        return history + "，正在放下水瓶。"
    current = ("再次给" if counts[target] else "开始给") + NAMES[target] + "倒水。"
    return (history + "，" if history else "") + current

def build(path):
    annotation = json.loads(path.read_text())
    progress_path = path.with_suffix(".progress.json")
    progress = json.loads(progress_path.read_text())
    n = annotation["total_frames"]
    assert annotation["frame_index_base"] == progress["frame_index_base"] == 0
    assert annotation["frame_interval"] == "inclusive"
    assert annotation["fps"] == progress["fps"] == 30 and annotation["sample_fps"] == 5
    assert len(progress["frames"]) == n
    assert progress["stage_annotation_sha256"] == digest(path), path.name
    segments = sorted(annotation["stages"] + annotation["idle_intervals"], key=lambda s: s["source_start_frame"])
    counts = Counter()
    first_put_down = set()
    rows = []
    cursor = 0
    for i, segment in enumerate(segments, 1):
        start, end = segment["source_start_frame"], segment["source_end_frame"]
        assert start == cursor and start <= end < n, (path.name, i, "coverage")
        action = segment.get("action", "idle")
        assert action in ("pour", "put_down", "idle")
        target = segment.get("target_position") if action == "pour" else segment.get("after_pour_target")
        if action != "idle":
            assert target in NAMES and segment["outcome"] == "completed"
            expected_occurrence = counts[target] + (action == "pour")
            assert segment["target_occurrence"] == expected_occurrence, (path.name, i, "occurrence")
        repeat = action != "idle" and (counts[target] >= (1 if action == "pour" else 2))
        expected_mode = "idle_hold" if action == "idle" else ("repeat_hold" if repeat else "first_visit")
        for f in progress["frames"][start:end + 1]:
            assert f["frame_index"] == cursor and f["stage_id"] == segment["stage_id"]
            assert f["mode"] == expected_mode, (path.name, cursor, "PT mode")
            cursor += 1
        text = label_text(action, target, counts, first_put_down)
        rows.append({
            "supervision_id": i,
            "stage_id": segment["stage_id"],
            "source_segment_index": segment.get("segment_index", 0),
            "source_start_frame": start,
            "source_end_frame": end,
            "action": action,
            "target_position": target,
            "is_repeat": bool(repeat),
            "completed_pours_before": {t: counts[t] for t in NAMES},
            "text": text,
        })
        # Only subsequent intervals see the completed action.
        if action == "pour":
            counts[target] += 1
        elif action == "put_down" and counts[target] == 1:
            first_put_down.add(target)
    assert cursor == n
    assert all(counts[t] == annotation["target_pour_counts"][t] for t in NAMES)
    assert all(len([r for r in rows if r["source_start_frame"] <= f <= r["source_end_frame"]]) == 1
               for f in range(0, n, 6))
    for row in rows:
        if "任务完成" in row["text"]:
            assert all(abs(f["pt"] - 1) < 1e-9 for f in progress["frames"][row["source_start_frame"]:row["source_end_frame"] + 1])
    result = {
        "schema_version": "pour-stage-supervision-v1",
        "episode": path.stem,
        "video_path": "../" + path.stem + ".mp4",
        "source_annotation": path.name,
        "source_annotation_sha256": digest(path),
        "source_progress": progress_path.name,
        "source_progress_sha256": digest(progress_path),
        "human_verified": annotation.get("human_verified", False),
        "fps": annotation["fps"],
        "total_frames": n,
        "sample_fps": 5,
        "frame_index_base": 0,
        "frame_interval": "inclusive",
        "alignment": "source_start_frame <= original_frame_index <= source_end_frame",
        "text_scope": "简短粗阶段文本；开始给某杯倒水沿用原取瓶至回正区间，不细分瞬时动作。完成次数仅累计此前已结束的倒水片段。",
        "boundary_uncertainty_seconds": 0.2,
        "progress_usage": "按原始frame_index读取source_progress中该帧pt，追加到text；本文件不复制或重算PT。",
        "usage": "监督答案，不作为模型输入。按原始视频划分数据；不将同一视频的帧分入不同集合。",
        "stages": rows,
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    paths = sorted(p for p in ANNOTATIONS.glob("episode_*.json") if p.name.count(".") == 1)
    videos = {p.stem for p in (ROOT / "videos/main_view").glob("episode_*.mp4")}
    assert {p.stem for p in paths} == videos
    for view in ("left_view", "right_view"):
        assert {p.stem for p in (ROOT / "videos" / view).glob("episode_*.mp4")} == videos
    sources = [p for path in paths for p in (path, path.with_suffix(".progress.json"))]
    before = {str(p): digest(p) for p in sources}
    prepared = [(path.with_suffix(".supervision.json"), build(path)) for path in paths]
    expected_example = [
        "尚未倒水，当前空闲。",
        "开始给右杯倒水。",
        "右杯已倒水一次，正在放下水瓶。",
        "右杯已倒水一次，当前空闲。",
        "右杯已倒水一次，开始给中杯倒水。",
        "右杯、中杯各已倒水一次，正在放下水瓶。",
        "右杯、中杯各已倒水一次，当前空闲。",
        "右杯、中杯各已倒水一次，再次给右杯倒水。",
        "右杯已倒水两次、中杯一次，正在放下水瓶。",
        "右杯已倒水两次、中杯一次，当前空闲。",
        "右杯已倒水两次、中杯一次，开始给左杯倒水。",
        "三个杯子均已倒过水，正在放下水瓶。",
        "三个杯子均已倒过水，水瓶已放下，任务完成。",
    ]
    example = next(d for p, d in prepared if d["episode"] == "episode_000001")
    assert [r["text"] for r in example["stages"]] == expected_example
    if args.write:
        assert all(not p.exists() for p, d in prepared), "Will not overwrite existing supervision"
        assert all(digest(Path(p)) == h for p, h in before.items()), "Source changed during preparation"
        for path, data in prepared:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
    assert all(digest(Path(p)) == h for p, h in before.items()), "Source changed during generation"
    report = {
        "status": "generated_and_checked" if args.write else "dry_run_passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(ANNOTATIONS),
        "videos": len(prepared),
        "supervision_intervals": sum(len(d["stages"]) for p, d in prepared),
        "original_frames_covered": sum(d["total_frames"] for p, d in prepared),
        "existing_5fps_points": sum(len(range(0, d["total_frames"], 6)) for p, d in prepared),
        "interval_counts": dict(Counter(r["action"] for p, d in prepared for r in d["stages"])),
        "repeat_pour_intervals": sum(r["action"] == "pour" and r["is_repeat"] for p, d in prepared for r in d["stages"]),
        "videos_without_trailing_idle": [d["episode"] for p, d in prepared if d["stages"][-1]["action"] != "idle"],
        "all_frame_intervals_exactly_cover_original_video": True,
        "source_annotation_and_pt_hashes_unchanged": True,
        "reference_example_matches_approved_text": True,
        "training_started": False,
        "note": "基于现有标注派生，继承human_verified=false和约0.2秒边界误差；没有新增视频视觉审核、未编造末尾完成区间。",
        "source_hashes": before,
        "outputs": [{"path": str(p), "sha256": digest(p) if args.write else None} for p, d in prepared],
    }
    if args.write:
        report_path = Path(__file__).with_name("generation-report.json")
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_hashes", "outputs")}, ensure_ascii=False, indent=2))
    print("EXAMPLE", json.dumps([{"frames": [r["source_start_frame"], r["source_end_frame"]], "text": r["text"]} for r in example["stages"]], ensure_ascii=False))

if __name__ == "__main__":
    main()
