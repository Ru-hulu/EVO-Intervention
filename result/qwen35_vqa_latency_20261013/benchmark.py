#!/usr/bin/env python3
"""Single-GPU, batch-one Qwen3.5 image-sequence VQA latency benchmark.

The official processor and model.generate are used unchanged. CUDA events
partition the first forward pass without inserting per-module synchronization.
Wall-clock TTFT includes generation setup and CPU orchestration. Cold loading,
video-to-image preparation, and warmups are kept separate from measured requests.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

PROMPT = (
    "You are given eighteen frames sampled in chronological order from a single "
    "robot manipulation video. Examine the whole sequence and answer this question: "
    "What actions does the robot perform, and what is the final visible state of "
    "the objects? Describe the main actions in the order they occur, identify the "
    "objects involved, and mention any clear evidence that an action finishes. "
    "Base your answer only on the images. Do not infer hidden events or invent "
    "objects. If a detail is unclear, state that briefly. Respond in English using "
    "three concise sentences, with no introduction, numbered list, or reasoning."
)


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    started = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "images"
    frame_dir.mkdir(exist_ok=True)
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total = sum(1 for _ in container.decode(video=0))
    indices = np.linspace(0, total - 1, 18).round().astype(int).tolist()
    assert len(set(indices)) == 18
    selected = set(indices)
    images = []
    records = []
    with av.open(str(args.video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in selected:
                continue
            image = frame.to_image().convert("RGB")
            if image.size != (640, 480):
                image = image.resize((640, 480), Image.Resampling.LANCZOS)
            path = frame_dir / f"frame_{len(records):02d}.png"
            image.save(path)
            records.append({"path": str(path.relative_to(output)), "source_frame": index,
                            "time_seconds": index / fps, "width": 640, "height": 480,
                            "sha256": sha256(path)})
            images.append(image)
    assert len(records) == 18
    sheet = Image.new("RGB", (6 * 240, 3 * 200), "#f5f6fa")
    draw = ImageDraw.Draw(sheet)
    for i, (im, record) in enumerate(zip(images, records)):
        x, y = i % 6 * 240, i // 6 * 200
        sheet.paste(im.resize((240, 180)), (x, y))
        draw.text((x + 6, y + 183), f"{i + 1:02d} / {record['time_seconds']:.2f}s", fill="black")
    sheet.save(output / "contact_sheet.jpg", quality=92)
    (output / "prompt.txt").write_text(PROMPT + "\n")
    manifest = {
        "source_video": str(args.video), "source_video_sha256": sha256(args.video),
        "source_task": "store_laptop_and_headphones", "source_fps": fps,
        "source_frame_count": total, "image_count": 18, "images": records,
        "prompt": PROMPT, "prompt_word_count": len(PROMPT.split()),
        "input_mode": "18 separate image items in temporal order; not the video API",
        "video_decode_and_frame_export_seconds": time.perf_counter() - started,
    }
    save_json(output / "input_manifest.json", manifest)
    print(json.dumps({"prepared": True, "images": 18, "prompt_words": len(PROMPT.split())}), flush=True)


def gpu_snapshot():
    def query(arguments):
        return subprocess.check_output(["nvidia-smi", *arguments], text=True).strip()
    return {
        "gpus": query(["--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv"]),
        "processes": query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"]),
    }


def check_free_gpu(index):
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    free, utilization = [int(x.strip()) for x in raw.strip().split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True).strip()
    if processes or free < 30000 or utilization > 10:
        raise RuntimeError(f"GPU {index} is occupied: free={free} MiB, utilization={utilization}%, pids={processes!r}. No process was stopped.")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run(args):
    if args.gpu is None:
        raise ValueError("--gpu must be specified for a GPU benchmark")
    check_free_gpu(args.gpu)
    gpu_uuid = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader,nounits"
    ], text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    imports_seconds = time.perf_counter() - started
    torch.set_num_threads(4)
    torch.manual_seed(0)
    assert torch.cuda.device_count() == 1
    manifest = json.loads((args.output / "input_manifest.json").read_text())
    paths = [args.output / r["path"] for r in manifest["images"]]
    assert all(sha256(p) == r["sha256"] for p, r in zip(paths, manifest["images"]))
    model_path = Path('/data/evo-inter/models') / ('Qwen3.5-' + args.model_size)
    report = {
        "model_size": args.model_size, "model_path": str(model_path), "gpu_index": args.gpu,
        "gpu_uuid": gpu_uuid,
        "settings": {"dtype": "bfloat16", "enable_thinking": False, "max_new_tokens": 128,
                     "do_sample": False, "batch_size": 1, "warmups": args.warmups,
                     "runs": args.runs, "attention": "sdpa", "torch_compile": False,
                     "image_processor_device": "cpu", "input_manifest": "input_manifest.json"},
        "versions": {"python": __import__('sys').version, "torch": torch.__version__,
                     "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "gpu_before_load": gpu_snapshot(), "imports_seconds": imports_seconds,
        "source_config": json.loads((model_path / "config.json").read_text()),
    }
    for package in ['flash-linear-attention', 'causal-conv1d', 'flash-attn', 'triton']:
        try:
            report['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][package] = None
    report['gated_deltanet_fast_path_available'] = getattr(modeling_qwen3_5, 'is_fast_path_available', None)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    report["processor_load_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, device_map={"": "cuda:0"},
    ).eval()
    torch.cuda.synchronize()
    report["model_load_seconds"] = time.perf_counter() - start
    report["model_parameters"] = sum(p.numel() for p in model.parameters())
    report["device"] = torch.cuda.get_device_name(0)
    print(json.dumps({"model": args.model_size, "loaded_seconds": report["model_load_seconds"],
                      "fast_path": report['gated_deltanet_fast_path_available']}), flush=True)

    # Hooks record CUDA events only: no synchronization at model/vision/LLM boundaries.
    trace = {"enabled": False, "forward_count": 0, "events": {}}
    def root_pre(module, positional, kwargs):
        if not trace["enabled"]:
            return
        trace["forward_count"] += 1
        if trace["forward_count"] == 1:
            mark("forward", "start")
    def root_post(module, positional, kwargs, output):
        if trace["enabled"] and trace["forward_count"] == 1:
            mark("forward", "end")
    def mark(name, side):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        trace["events"][name + "_" + side] = event
    def module_pre(name):
        def hook(module, positional, kwargs):
            if trace["enabled"] and trace["forward_count"] == 1:
                mark(name, "start")
        return hook
    def module_post(name):
        def hook(module, positional, kwargs, output):
            if trace["enabled"] and trace["forward_count"] == 1:
                mark(name, "end")
        return hook
    handles = [model.register_forward_pre_hook(root_pre, with_kwargs=True),
               model.register_forward_hook(root_post, with_kwargs=True)]
    for name, module in [('vision', model.model.visual), ('merger', model.model.visual.merger),
                         ('language_prefill', model.model.language_model), ('lm_head', model.lm_head)]:
        handles += [module.register_forward_pre_hook(module_pre(name), with_kwargs=True),
                    module.register_forward_hook(module_post(name), with_kwargs=True)]

    class FirstTokenClock:
        def __init__(self):
            self.prompt_seen = False
            self.first_token = None
            self.token_count = 0
        def put(self, value):
            # HF generate calls put(input_ids.cpu()) before generated tokens.
            if not self.prompt_seen:
                self.prompt_seen = True
                return
            if self.first_token is None:
                self.first_token = time.perf_counter()
            self.token_count += value.numel()
        def end(self):
            pass

    original_preprocess = processor.image_processor.preprocess
    image_timings = []
    def timed_preprocess(*positional, **kwargs):
        start = time.perf_counter()
        result = original_preprocess(*positional, **kwargs)
        image_timings.append((time.perf_counter() - start) * 1000)
        return result
    processor.image_processor.preprocess = timed_preprocess

    def once(run_number, warmup):
        trace.update(enabled=not warmup, forward_count=0, events={})
        image_timings.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        request_start = time.perf_counter()
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        images_ready = time.perf_counter()
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": manifest["prompt"]})
        messages = [{"role": "user", "content": content}]
        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        cpu = processor(text=[formatted], images=images, return_tensors="pt", device="cpu")
        processed = time.perf_counter()
        sequence_length = cpu['input_ids'].shape[1]
        image_token_count = int((cpu['input_ids'] == model.config.image_token_id).sum())
        grids = cpu['image_grid_thw'].tolist()
        transfer_start = time.perf_counter()
        inputs = {k: v.to(device='cuda:0', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                  for k, v in cpu.items()}
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        streamer = FirstTokenClock()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, do_sample=False, max_new_tokens=128, use_cache=True,
                streamer=streamer, logits_to_keep=1,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generation_finished = time.perf_counter()
        generated = outputs[0, sequence_length:].cpu()
        answer = processor.tokenizer.decode(generated, skip_special_tokens=True)
        completed = time.perf_counter()
        assert streamer.first_token is not None
        assert streamer.token_count == generated.numel()
        assert len(grids) == 18 and image_token_count == 5400, (grids, image_token_count)
        timing = {
            'image_read_decode_ms': (images_ready - request_start) * 1000,
            'image_preprocess_ms': sum(image_timings),
            'prompt_tokenize_pack_ms': (processed - images_ready) * 1000 - sum(image_timings),
            'input_metadata_ms': (transfer_start - processed) * 1000,
            'host_to_device_ms': (transferred - transfer_start) * 1000,
            'generation_setup_gap_ms': (generation_start - transferred) * 1000,
            'ttft_generation_ms': (streamer.first_token - generation_start) * 1000,
            'ttft_end_to_end_ms': (streamer.first_token - request_start) * 1000,
            'decode_after_first_token_ms': (generation_finished - streamer.first_token) * 1000,
            'output_to_text_ms': (completed - generation_finished) * 1000,
            'generation_total_ms': (generation_finished - generation_start) * 1000,
            'end_to_end_ms': (completed - request_start) * 1000,
        }
        if not warmup:
            def span(name):
                return trace['events'][name + '_start'].elapsed_time(trace['events'][name + '_end'])
            timing.update({
                'vision_encoder_cuda_ms': span('vision') - span('merger'),
                'visual_merger_cuda_ms': span('merger'),
                'language_prefill_cuda_ms': span('language_prefill'),
                'first_lm_head_cuda_ms': span('lm_head'),
                'first_forward_cuda_ms': span('forward'),
            })
            components = sum(timing[key] for key in ['vision_encoder_cuda_ms', 'visual_merger_cuda_ms',
                                                      'language_prefill_cuda_ms', 'first_lm_head_cuda_ms'])
            timing['ttft_other_ms'] = timing['ttft_generation_ms'] - components
        generated_count = generated.numel()
        eos = model.generation_config.eos_token_id or model.config.text_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        result = {
            'run': run_number, 'warmup': warmup, 'timings': timing,
            'output_tokens': generated_count,
            'tokens_after_first': generated_count - 1,
            'decode_tokens_per_second': ((generated_count - 1) / (timing['decode_after_first_token_ms'] / 1000)) if generated_count > 1 else None,
            'stopped_on_eos': generated[-1].item() in eos,
            'answer': answer, 'input_total_tokens': sequence_length,
            'input_image_tokens': image_token_count, 'image_grid_thw': grids,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        }
        trace['enabled'] = False
        del inputs, outputs, cpu, generated, images
        return result

    report['warmups'] = []
    report['runs'] = []
    report_path = args.output / ('results_' + args.model_size + '.json')
    for i in range(args.warmups):
        row = once(i + 1, True)
        report['warmups'].append(row)
        save_json(report_path, report)
        print(json.dumps({'model': args.model_size, 'warmup': i + 1,
                          'seconds': row['timings']['end_to_end_ms'] / 1000,
                          'output_tokens': row['output_tokens']}), flush=True)
    for i in range(args.runs):
        row = once(i + 1, False)
        report['runs'].append(row)
        save_json(report_path, report)
        print(json.dumps({'model': args.model_size, 'run': i + 1,
                          'seconds': row['timings']['end_to_end_ms'] / 1000,
                          'output_tokens': row['output_tokens']}), flush=True)
    report['summary'] = {}
    for metric in report['runs'][0]['timings']:
        values = [r['timings'][metric] for r in report['runs']]
        report['summary'][metric] = {'mean': statistics.mean(values), 'median': statistics.median(values),
                                     'min': min(values), 'max': max(values), 'p95': percentile(values, .95)}
    report['gpu_after_runs'] = gpu_snapshot()
    report['notes'] = [
        'All requests reopen and decode the same 18 PNGs; OS disk cache is warm.',
        'Model load, tokenizer load, imports, video extraction and warmups are excluded from steady-state request latency.',
        'End-to-end and TTFT are wall-clock time; named CUDA spans measure first-forward module time with CUDA events.',
        'vision_encoder_cuda_ms excludes the separately reported visual merger; first_forward_cuda_ms is inclusive.',
        'ttft_other_ms is TTFT minus the four non-overlapping CUDA module spans and includes generation setup, embedding/scatter/position work and Python/token-selection/streaming overhead.',
        'decode_after_first_token_ms covers tokens 2..N plus generation finalization; output length is reported separately.',
        'The lightweight streamer timestamps token delivery but does not print during timed generation.',
        'GPU is reserved by visibility for this process only; this script does not stop or alter other jobs.',
    ]
    save_json(report_path, report)
    for handle in handles:
        handle.remove()
    print(json.dumps({'completed': True, 'report': str(report_path)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--model-size', choices=['2B', '4B'], default='2B')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    args = parser.parse_args()
    if args.prepare:
        if args.video is None:
            parser.error('--prepare requires --video')
        prepare(args)
    else:
        run(args)
