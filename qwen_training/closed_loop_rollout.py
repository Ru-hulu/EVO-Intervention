"""Run a causal, one-episode rollout with the last generated state as feedback.

No optimizer, training, scheduling, or teacher-forced decoding is involved.
Only the first prompt receives annotated state; later prompts use model output.
"""

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import time

from qwen_training.build_sample_index import state_of
from qwen_training.delta_to_progress import delta_to_progress
from qwen_training.evaluation_metrics import parse_response
from qwen_training.prompts import TASK_INTENT, build_messages, describe_target


STATE_KEYS = ("nowid", "finishid", "restid")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def history_window(frame, episode_start, fps=30):
    """Six observations spaced one second apart, never before the clip start."""
    if frame < episode_start:
        raise ValueError("Prediction precedes the episode")
    return [i if i >= episode_start else None for i in range(frame - 5 * fps, frame + 1, fps)]


def feedback_state(response):
    """Keep generated fields verbatim in meaning, even if mutually inconsistent.

    Unparseable fields abort the run: never substitute ground truth or silently
    repair an output with knowledge of the annotated stage sequence.
    """
    parsed = parse_response(response)
    if any(parsed[key] is None for key in STATE_KEYS):
        raise ValueError("Generated stage state is unparseable; no ground-truth fallback is allowed")
    return {key: parsed[key] for key in STATE_KEYS}


def prepare(args, config):
    root = args.dataset_root
    index = root / "samples/samples.jsonl"
    if str(index) not in config["index_files"]:
        raise ValueError("Dataset index is not a checkpoint training/validation source")
    split = json.loads(args.split.read_text())
    identity = [root.name, args.video_id]
    if identity not in split["validation"] or identity in split["train"]:
        raise ValueError("Requested video is not held out for validation")
    annotation_path = root / "labels" / (args.video_id + ".JSON")
    if not annotation_path.exists():
        annotation_path = annotation_path.with_suffix(".json")
    delta_path = root / "delta" / (args.video_id + "-delta.txt")
    provenance = json.loads((root / "samples/statistics.json").read_text())
    source = next(s for s in provenance["sources"] if s["video_id"] == args.video_id)
    for path, field in [(annotation_path, "annotation_sha256"), (delta_path, "delta_sha256")]:
        if sha256(path) != source[field]:
            raise ValueError(f"Source differs from the existing sample index: {path}")
    annotation = json.loads(annotation_path.read_text())
    episode = next(e for e in annotation["tasks"][args.task_index]["episodes"] if e["episode"] == args.episode)
    segments = episode["subtasks"]
    starts = [s["30FPSstart_frame"] for s in segments]
    start, end = starts[0], segments[-1]["30FPSend_frame"]
    ids = {int(s["id"]) for s in annotation["taskid"]}
    states = [state_of(s, ids) for s in segments]
    deltas = [float(s.split()[0]) for s in delta_path.read_text().splitlines()]
    if not all(math.isfinite(x) and x >= 0 for x in deltas):
        raise ValueError("Invalid delta source")
    progress = delta_to_progress(deltas, annotation)

    def target_at(frame):
        return dict(states[bisect_right(starts, frame) - 1], intent=TASK_INTENT, progress=progress[frame])

    paths = None
    checked = 0
    with index.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["video_id"] != args.video_id or row["task_index"] != args.task_index or row["episode"] != args.episode:
                continue
            paths = row["video_paths"] if paths is None else paths
            if paths != row["video_paths"]:
                raise ValueError("Inconsistent source videos")
            target = target_at(row["frame_indices"][-1])
            if any(target[k] != row["target"][k] for k in STATE_KEYS) or abs(target["progress"] - row["target"]["progress"]) > 1e-7:
                raise ValueError("Dense ground truth disagrees with existing evaluation labels")
            checked += 1
    if paths is None:
        raise ValueError("Episode missing from saved index")
    rows = []
    for step, frame in enumerate(range(start, end + 1, args.step_frames)):
        frames = history_window(frame, start)
        rows.append(dict(sample_id=f"rollout/{root.name}/{args.video_id}/ep{args.episode}/{frame}",
                         dataset=root.name, video_id=args.video_id, task_index=args.task_index,
                         episode=args.episode, sample_type="closed_loop", sampling_method="causal_fixed_seconds",
                         step=step, frame_index=frame, time_seconds=(frame-start)/30,
                         source_time_seconds=frame/30, frame_indices=frames,
                         frame_mask=[[i is not None for i in frames]] * 3,
                         video_paths=paths, input_state=states[0], target=target_at(frame)))
    truth = [dict(frame_index=i, time_seconds=(i-start)/30, source_time_seconds=i/30, **target_at(i)) for i in range(start, end+1)]
    metadata = dict(dataset=root.name, video_id=args.video_id, task_index=args.task_index, episode=args.episode,
                    frame_start=start, frame_end=end, source_fps=30, duration_seconds=(end-start+1)/30,
                    prediction_step_frames=args.step_frames, prediction_hz=30/args.step_frames,
                    observations_per_view=6, history_offsets_seconds=[-5,-4,-3,-2,-1,0],
                    padding="black uint8 zero image and false frame mask; before episode start only",
                    start_policy="selected episode is a virtual clip; its first frame is rollout time zero",
                    previous_state_source="ground_truth_at_first_step_only_then_previous_model_generated_state",
                    feedback_fields=list(STATE_KEYS), unparseable_state_policy="abort_without_ground_truth_fallback",
                    inconsistent_state_policy="pass_generated_fields_without_repair", initial_state=states[0],
                    samples=len(rows), annotation=str(annotation_path), delta=str(delta_path),
                    annotation_sha256=sha256(annotation_path), delta_sha256=sha256(delta_path),
                    video_paths=paths, existing_index_labels_cross_checked=checked,
                    index_sha256=sha256(index), split_sha256=sha256(args.split),
                    evaluation_split="validation", independent_test_set=False)
    return rows, truth, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--step-frames", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.step_frames < 1:
        raise ValueError("Step must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Refusing to repeat or overwrite an existing rollout")
    try:
        import av
        import torch
        from qwen_training.dataset import VLMProgressDataset
        from qwen_training.evaluate_checkpoint import load_model
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
        if not is_fast_path_available:
            raise RuntimeError("Fast inference kernels unavailable")
        digest = sha256(args.checkpoint)
        if digest != args.expected_checkpoint_sha256:
            raise ValueError("Checkpoint hash mismatch")
        torch.set_num_threads(2)
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        rows, truth, manifest = prepare(args, checkpoint["config"])
        for name, path in manifest["video_paths"].items():
            with av.open(path) as video:
                stream = video.streams.video[0]
                if stream.average_rate != 30 or stream.frames <= manifest["frame_end"]:
                    raise ValueError(f"Invalid source video timebase or length: {name}")
        manifest.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest,
                        checkpoint_epoch=checkpoint["epoch"], checkpoint_step=checkpoint["step"],
                        model_path=checkpoint["config"]["model_path"], generation="greedy",
                        max_new_tokens=args.max_new_tokens, started_unix=time.time(),
                        physical_gpu_visibility=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                                       ["closed_loop_rollout.py","model.py","prompts.py","dataset.py","evaluation_metrics.py"]})
        save_json(args.output_dir / "manifest.json", manifest)
        save_json(args.output_dir / "ground_truth.json", truth)
        index_path = args.output_dir / "rollout_index.jsonl"
        index_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
        dataset = VLMProgressDataset(index_path)
        save_json(args.output_dir / "status.json", dict(status="loading_model", samples=len(rows)))
        model = load_model(checkpoint, device)
        del checkpoint
        current_state = manifest["initial_state"]
        started = time.monotonic()
        predictions_path = args.output_dir / "predictions.jsonl"
        torch.cuda.reset_peak_memory_stats()
        with predictions_path.open("x") as output, torch.inference_mode():
            for index, meta in enumerate(rows):
                step_started = time.monotonic()
                sample = dataset[index]
                # Only this state variable supplies the prompt. sample['target']
                # and all subsequent annotations are never passed to the model.
                messages = build_messages([current_state], state_drop_probability=0.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    visual = model.encode_history(sample["images"].unsqueeze(0).to(device), sample["frame_mask"].unsqueeze(0))
                    prediction = model.generate(messages, **visual, max_new_tokens=args.max_new_tokens)
                progress = float(prediction["progress"][0])
                response = prediction["responses"][0]
                if not math.isfinite(progress) or not 0 <= progress <= 1:
                    raise ValueError("Invalid progress prediction")
                record = dict(meta, input_state=current_state, prompt=messages[0], response=response,
                              predicted_progress=progress, parsed=parse_response(response),
                              target_response=describe_target(meta["target"]), mode="closed_loop",
                              input_state_source="initial_ground_truth" if index == 0 else "previous_prediction",
                              seconds=time.monotonic()-step_started)
                output.write(json.dumps(record, ensure_ascii=False)+"\n")
                output.flush()
                current_state = feedback_state(response)
                status = dict(status="running", completed=index+1, samples=len(rows),
                              elapsed_seconds=time.monotonic()-started, last_frame=meta["frame_index"],
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save_json(args.output_dir / "status.json", status)
                print(json.dumps(status), flush=True)
        completion = dict(status="complete", samples=len(rows), elapsed_seconds=time.monotonic()-started,
                          predictions_sha256=sha256(predictions_path), ground_truth_sha256=sha256(args.output_dir/"ground_truth.json"),
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          finished_unix=time.time())
        save_json(result_path, completion)
        save_json(args.output_dir / "status.json", completion)
    except Exception as error:
        failure = dict(status="failed", error=repr(error), finished_unix=time.time())
        save_json(result_path, failure)
        save_json(args.output_dir / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
