"""为MemLong所有已标注主视角视频生成逐帧delta/PT；模型只加载一次。"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from mem_lite.datasets.annotations import read_annotation
from mem_lite.datasets.prepare_progress import file_sha256, load_vision_encoder, prepare_progress


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_progress_batch(dataset_root, model_path, output_dir, device="cuda", batch_size=8,
                           max_pixels=262144, min_event_delta=1e-8):
    dataset_root, model_path = Path(dataset_root).resolve(), Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    inputs = []
    for path in sorted(dataset_root.glob("*/annotations/*.annotations.json")):
        annotation = read_annotation(path)
        video = path.parent.parent / annotation["video_file"]
        if not video.is_file():
            raise FileNotFoundError(video)
        inputs.append({"task": path.parent.parent.name, "annotation": str(path), "video": str(video),
                       "frames": annotation["total_frames"], "intervals": len(annotation["stages"])})
    if not inputs:
        raise ValueError("未找到已标注视频")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "format_version": "mem-lite-event-equal-progress-v2", "dataset_root": str(dataset_root),
        "model_path": str(model_path), "device": device, "batch_size": batch_size,
        "max_pixels": max_pixels, "min_event_delta": min_event_delta,
        "video_count": len(inputs), "frame_count": sum(item["frames"] for item in inputs),
        "scripts_sha256": {name: file_sha256(Path(__file__).with_name(name))
                           for name in ("prepare_progress.py", "prepare_progress_batch.py")},
        "inputs": inputs,
    }
    write_json(output_dir / "manifest.json", manifest)
    state = {"status": "loading_model", "pid": os.getpid(), "total": len(inputs),
             "completed": 0, "failed": 0, "completed_frames": 0, "current": None, "failures": []}
    write_json(output_dir / "status.json", state)
    try:
        encoder = load_vision_encoder(model_path, device)
        with (output_dir / "index.jsonl").open("w", encoding="utf-8") as index_file:
            for item in inputs:
                state.update(status="running", current=item["video"], elapsed_seconds=perf_counter() - started)
                write_json(output_dir / "status.json", state)
                video_output = output_dir / item["task"] / Path(item["video"]).stem
                try:
                    result = prepare_progress(
                        item["annotation"], item["video"], model_path, video_output, device,
                        batch_size, max_pixels, min_event_delta, vision_encoder=encoder,
                    )
                except Exception as exc:
                    failure = {"video": item["video"], "error": f"{type(exc).__name__}: {exc}"}
                    state["failures"].append(failure)
                    state["failed"] += 1
                    print(json.dumps(failure, ensure_ascii=False), flush=True)
                else:
                    record = {**item, **result}
                    index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    index_file.flush()
                    state["completed"] += 1
                    state["completed_frames"] += result["frame_count"]
                    print(json.dumps({"completed": state["completed"], "total": len(inputs),
                                      "video": item["video"], **result}, ensure_ascii=False), flush=True)
                state["elapsed_seconds"] = perf_counter() - started
                write_json(output_dir / "status.json", state)
        state.update(status="complete_with_errors" if state["failed"] else "complete", current=None)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state.update(elapsed_seconds=perf_counter() - started, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output_dir / "status.json", state)
        write_json(output_dir / "result.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="新建批次输出目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-event-delta", type=float, default=1e-8)
    result = prepare_progress_batch(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
