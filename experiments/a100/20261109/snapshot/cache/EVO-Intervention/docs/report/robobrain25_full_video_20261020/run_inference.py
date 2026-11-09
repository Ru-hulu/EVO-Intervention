"""Reproducible start-to-current GRM evaluation of the supplied block-swap video."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUN = Path("/root/autodl-tmp/EvalProgress/outputs/full_video_20261020")
MODEL = Path("/root/models/RoboBrain2.5-4B")
SOURCE = Path("/root/autodl-tmp/EvalProgress/RoboBrain2.5")
sys.path.insert(0, str(SOURCE / "scripts"))
from annotate_screw import build_messages, parse_score

TASK = (
    "Swap the two blocks between their mats. Each mat holds at most one block and "
    "only one block may move at a time, so it takes three moves: move the left one "
    "of the two blocks into the empty mat first, then the other block into the mat "
    "it just vacated, then the parked block into the mat the second came from. "
    "After each move, press the button once; pressing it changes nothing visible."
)
SAMPLE_FPS = 3
BATCH_SIZE = 8
MAX_NEW_TOKENS = 128


def write_json(name, data):
    target = RUN / name
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(target)


def update(phase, **data):
    write_json("inference_status.json", {
        "phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **data,
    })
    print(phase, json.dumps(data, ensure_ascii=False), flush=True)


def prepare():
    import cv2
    from PIL import Image
    from transformers import AutoProcessor
    video = RUN / "input/full_video.mp4"
    template = (SOURCE / "scripts/robobrain_grm_prompt.txt").read_text()
    (RUN / "task.txt").write_text(TASK + "\n")
    (RUN / "prompt_template.txt").write_text(template)
    (RUN / "prompt_filled.txt").write_text(template.format(task=TASK))
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open source video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = [i / SAMPLE_FPS for i in range(math.floor((length - 1) / fps * SAMPLE_FPS) + 1)]
        wanted = sorted({round(t * fps) for t in times} | {length - 1})
        images, timestamps = {}, {}
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = count
            count += 1
            if idx in wanted:
                images[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps[idx] = idx / fps
        assert count == length and len(images) == len(wanted), (count, length, wanted)
    finally:
        cap.release()
    start, end = images[0], images[length - 1]
    blank = Image.new("RGB", start.size, (0, 0, 0))
    start.save(RUN / "reference_start.png")
    end.save(RUN / "reference_end.png")
    blank.save(RUN / "missing_wrist_blank.png")
    image_batches = [[start, end, start, blank, blank, images[i], blank, blank] for i in wanted]
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    # Preserve the native 640x480 view, including the small button.
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 307200}
    texts = processor.apply_chat_template(
        [build_messages(TASK, image_batches[0], template)],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=texts, images=image_batches[0], padding=True, return_tensors="pt")
    grid = inputs.image_grid_thw.tolist()
    manifest = {
        "model": "BAAI/RoboBrain2.5-4B",
        "revision": "ec4b3bf5e969b48ad17df432b479c10402a0079b",
        "model_path": str(MODEL),
        "task": TASK, "task_source": "User-provided exact English instruction",
        "video": str(video), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "source_fps": fps, "source_frames": length, "duration_s": length / fps,
        "decoder": "OpenCV sequential decoding; constant-frame-rate frame_index/fps timestamps",
        "sample_fps": SAMPLE_FPS, "sample_frames": wanted,
        "sample_times_s": [timestamps[i] for i in wanted],
        "predicted_points": len(wanted),
        "sampling": "Nearest native frame at k/3 seconds, plus the exact final frame",
        "reference_start": "Video frame 0, main view",
        "reference_end": f"Video frame {length - 1}, main view; endpoint anchor, not verified success",
        "uses_future_goal_reference": True, "goal_success_verified": False,
        "comparison": "Fixed first frame BEFORE versus current sampled frame AFTER",
        "images_order": ["reference_start", "reference_end", "before_main_first_frame",
                         "before_left_wrist_blank", "before_right_wrist_blank",
                         "after_main_current_frame", "after_left_wrist_blank",
                         "after_right_wrist_blank"],
        "history_video": False, "previous_prediction_input": False,
        "expert_trajectory": None, "missing_views": "Black blank 640x480 images",
        "score_semantics": "Signed start-to-current GRM score in [-1,1], not calibrated absolute completion",
        "postprocessing": "Score times 100 for plotting only; no smoothing, clamping, accumulation, forced endpoints, or monotonicity",
        "backend": "transformers AutoModelForImageTextToText, Torch SDPA",
        "batch_size": BATCH_SIZE, "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "image_processor_size": processor.image_processor.size,
        "image_grid_thw": grid,
        "input_token_count_single_sample": int(inputs.attention_mask.sum()),
        "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "limitations": [
            "Start/end block layouts look alike; block identities and button-press history may be unobservable from static frames",
            "The GRM prompt calls the reference endpoint complete; supplying the final frame is an offline anchor assumption",
            "No ground-truth event annotations or accuracy metric supplied",
        ],
    }
    write_json("manifest.json", manifest)
    return processor, template, wanted, timestamps, image_batches, manifest


def infer(prepared):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText
    processor, template, wanted, timestamps, image_batches, manifest = prepared
    torch.manual_seed(42)
    torch.set_num_threads(8)
    load_start = time.perf_counter()
    update("loading_model", points=len(wanted))
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL), dtype="auto", device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    torch.cuda.synchronize()
    manifest.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                    gpu=torch.cuda.get_device_name(), model_load_s=time.perf_counter() - load_start)
    rows, batches = [], []
    eval_start = time.perf_counter()
    with (RUN / "predictions.jsonl").open("w") as raw:
        for begin in range(0, len(wanted), BATCH_SIZE):
            frames = wanted[begin:begin + BATCH_SIZE]
            images = image_batches[begin:begin + BATCH_SIZE]
            torch.cuda.synchronize()
            start = time.perf_counter()
            messages = [build_messages(TASK, im, template) for im in images]
            texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=texts, images=[im for group in images for im in group],
                               padding=True, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            generate_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize()
            generate_s = time.perf_counter() - generate_start
            generated = output[:, inputs.input_ids.shape[1]:]
            answers = processor.batch_decode(generated, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            elapsed = time.perf_counter() - start
            batch_id = len(batches)
            batches.append({"batch_id": batch_id, "size": len(frames), "elapsed_s": elapsed,
                            "generate_s": generate_s})
            for frame, answer, tokens in zip(frames, answers, generated):
                score, status = parse_score(answer)
                row = {"frame_index": frame, "time_s": timestamps[frame],
                       "score": score, "score_percent": None if score is None else score * 100,
                       "status": status, "raw_response": answer, "batch_id": batch_id,
                       "batch_elapsed_s": elapsed, "amortized_s_per_frame": elapsed / len(frames),
                       "generated_nonpad_tokens": int((tokens != processor.tokenizer.pad_token_id).sum())}
                rows.append(row)
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()
            update("inferencing", completed=len(rows), total=len(wanted),
                   last_batch_s=round(elapsed, 3),
                   scores=[r["score_percent"] for r in rows[-len(frames):]])
    elapsed = time.perf_counter() - eval_start
    with (RUN / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["timing"] = {
        "evaluation_wall_s": elapsed, "batches": batches,
        "amortized_s_per_frame": sum(b["elapsed_s"] for b in batches) / len(rows),
        "note": "Batch throughput including preprocessing and decoding, not serial frame latency; excludes model load",
    }
    manifest["valid_points"] = sum(r["status"] == "ok" for r in rows)
    write_json("manifest.json", manifest)
    update("complete", points=len(rows), valid_points=manifest["valid_points"],
           evaluation_wall_s=round(elapsed, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-model", action="store_true")
    args = parser.parse_args()
    prepared = prepare()
    if args.prepare_only:
        update("prepared", points=len(prepared[2]),
               input_tokens=prepared[-1]["input_token_count_single_sample"],
               image_grid_thw=prepared[-1]["image_grid_thw"])
        return
    if args.wait_for_model:
        update("waiting_for_verified_model")
        while True:
            state = json.loads((RUN / "download_status.json").read_text())
            if state["phase"] == "complete":
                break
            if state["phase"] == "failed":
                raise RuntimeError("Model download failed")
            time.sleep(15)
    if not (MODEL / "model.safetensors").exists():
        raise FileNotFoundError("Verified weights are not ready")
    infer(prepared)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update("failed", error=type(error).__name__, message=str(error))
        raise
