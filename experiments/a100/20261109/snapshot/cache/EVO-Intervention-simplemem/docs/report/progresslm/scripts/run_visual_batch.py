#!/usr/bin/env python3
"""Same confirmed visual demonstration, accelerated with vLLM batching and prefix caching."""
import argparse
import collections
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0',
                  VLLM_WORKER_MULTIPROC_METHOD='spawn')
from run_visual_demo_eval import REPO, CORE, MODEL, QUERY_ROOT, parse_response, write_json, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    overall_start = time.perf_counter()
    plan = json.loads((args.baseline / 'confirmed_plan.json').read_text())
    for name in ['confirmed_plan.json', 'verified_query_inputs.json', 'reference_top.jpg', 'reference_three_views.jpg']:
        shutil.copy2(args.baseline / name, out / name)
    shutil.copytree(args.baseline / 'reference_frames', out / 'reference_frames')
    shutil.copytree(args.baseline / 'official_source', out / 'official_source')
    shutil.copy2(__file__, out / 'run_visual_batch.py')
    shutil.copy2(Path(__file__).with_name('run_visual_demo_eval.py'), out / 'run_visual_demo_eval.py')
    refs = [str(out / 'reference_frames' / Path(ref['path']).name) for ref in plan['references']]
    for ref, path in zip(plan['references'], refs):
        assert sha(path) == ref['sha256']
    sys.path.insert(0, str(CORE))
    from prompts.visual_demo_prompt import build_visual_demo_prompt, VISUAL_DEMO_SYSTEM_PROMPT
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    import numpy as np
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    parts = build_visual_demo_prompt(plan['task'], refs, 4, str(QUERY_ROOT / 'frame_000000.png'))
    content = [{'type': 'text', 'text': p['value']} if p['type'] == 'text'
               else {'type': 'image', 'image': 'file://' + p['value']} for p in parts]
    messages = [{'role': 'system', 'content': VISUAL_DEMO_SYSTEM_PROMPT}, {'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert prompt == (args.baseline / 'example_rendered_prompt.txt').read_text(), 'Prompt differs from baseline'
    images, videos = process_vision_info(messages)
    assert len(images) == 6 and videos is None
    reference_images = images[:5]
    assert all(im.size == (644, 476) for im in images)
    (out / 'example_rendered_prompt.txt').write_text(prompt)
    write_json(out / 'example_messages.json', messages)
    print('PROMPT VERIFIED byte-for-byte identical to serial baseline; six images at644x476.', flush=True)
    init_start = time.perf_counter()
    llm = LLM(model=str(MODEL), dtype='bfloat16', gpu_memory_utilization=.75,
              max_model_len=8192, limit_mm_per_prompt={'image': 6},
              enable_prefix_caching=True, mm_processor_cache_gb=2,
              max_num_seqs=32, max_num_batched_tokens=16384,
              enforce_eager=True, seed=0)
    init_s = time.perf_counter() - init_start
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=2048, seed=0)
    manifest = json.loads((args.baseline / 'manifest.json').read_text())
    manifest.update(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                    driver='vLLM batched; official ProgressLM visual prompt and same checkpoint',
                    runtime={x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'vllm', 'qwen-vl-utils']},
                    attention='vLLM backend selected automatically; see run log',
                    batch_size=64, max_concurrent_sequences=32, reference_cache_across_requests=True,
                    generation={'temperature': 0, 'top_p': 1, 'max_tokens': 2048, 'seed': 0},
                    timing={'scope': 'batch preparation plus complete synchronous generate; CUDA engine scheduling included',
                            'metric': 'offline throughput; not single-request latency',
                            'initialization_reported_separately': True},
                    model_initialization_s=init_s, serial_baseline_directory=str(args.baseline),
                    script_sha256=sha(__file__), original_prompt_verified_identical=True,
                    source_configuration_note='User prioritized fastest complete curve over single-frame latency. Same six images/anchors/prompt/greedy decoding; batched vLLM with reusable reference pixels and prefix KV cache.')
    write_json(out / 'manifest.json', manifest)
    print(f'VLLM READY in {init_s:.3f}s', flush=True)
    all_rows, batches = [], []
    inference_start = time.perf_counter()
    with (out / 'predictions.jsonl').open('x') as sink:
        for start in range(0, 690, 64):
            end = min(start + 64, 690)
            t0 = time.perf_counter()
            requests = []
            for i in range(start, end):
                current, no_video = process_vision_info([{'role': 'user', 'content': [
                    {'type': 'image', 'image': 'file://' + str(QUERY_ROOT / f'frame_{i:06d}.png')}
                ]}])
                assert len(current) == 1 and current[0].size == (644, 476)
                requests.append({'prompt': prompt, 'multi_modal_data': {'image': reference_images + current}})
            prepared = time.perf_counter()
            results = llm.generate(requests, sampling, use_tqdm=False)
            generated = time.perf_counter()
            assert len(results) == end - start
            batch_rows = []
            for i, result in zip(range(start, end), results):
                prediction = result.outputs[0]
                parsed = parse_response(prediction.text)
                capped = prediction.finish_reason == 'length'
                if capped:
                    parsed.update(progress_percent=None, status='truncated')
                row = {'frame_index': i, 'timestamp_s': i / 30, **parsed,
                       'batch_index': len(batches), 'input_tokens': len(result.prompt_token_ids),
                       'generated_tokens': len(prediction.token_ids), 'finish_reason': prediction.finish_reason,
                       'hit_token_cap': capped, 'image_sizes': [[644,476]] * 6,
                       'num_cached_tokens': getattr(result, 'num_cached_tokens', None),
                       'raw_response': prediction.text}
                batch_rows.append(row)
            batch_s = time.perf_counter() - t0
            batch = {'batch_index': len(batches), 'start_frame': start, 'end_frame_inclusive': end - 1,
                     'frames': end - start, 'prepare_s': prepared - t0, 'generate_s': generated - prepared,
                     'total_s': batch_s, 'throughput_fps': (end - start) / batch_s}
            batches.append(batch)
            for row in batch_rows:
                sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            sink.flush()
            all_rows.extend(batch_rows)
            status = {'completed': len(all_rows), 'total': 690, 'last_batch': batch,
                      'status_counts': dict(collections.Counter(r['status'] for r in all_rows)),
                      'inference_elapsed_s': time.perf_counter() - inference_start}
            write_json(out / 'running_status.json', status)
            print(json.dumps(status), flush=True)
    inference_wall_s = time.perf_counter() - inference_start
    baseline = [json.loads(x) for x in (args.baseline / 'predictions.jsonl').read_text().splitlines()]
    pairs = [(r, all_rows[r['frame_index']]) for r in baseline]
    valid_pairs = [(a,b) for a,b in pairs if a['status'] == b['status'] == 'ok']
    compare = {'frames_in_serial_baseline': len(baseline), 'jointly_valid_frames': len(valid_pairs),
               'exact_score_matches': sum(a['progress_percent'] == b['progress_percent'] for a,b in valid_pairs),
               'mean_absolute_difference_pp': float(np.mean([abs(a['progress_percent']-b['progress_percent']) for a,b in valid_pairs])) if valid_pairs else None,
               'max_absolute_difference_pp': max([abs(a['progress_percent']-b['progress_percent']) for a,b in valid_pairs], default=None),
               'status_disagreements': sum(a['status'] != b['status'] for a,b in pairs),
               'serial_mean_latency_ms': float(np.mean([r['latency_ms'] for r in baseline])),
               'serial_timing_scope': 'first partial trajectory only, batch1; not full690 latency average',
               'note': 'vLLM/backend/batch floating-point differences can change greedy generated text and scores; no forced consistency.'}
    (out / 'serial_baseline').mkdir()
    for name in ['predictions.jsonl','manifest.json','warmup_results.json','interruption.json']:
        if (args.baseline/name).exists():shutil.copy2(args.baseline/name,out/'serial_baseline'/name)
    write_json(out / 'baseline_comparison.json', compare)
    write_json(out / 'batch_timings.json', batches)
    summary = {'status': 'completed', 'frames': 690,
               'valid_frames': sum(r['status']=='ok' for r in all_rows),
               'status_counts': dict(collections.Counter(r['status'] for r in all_rows)),
               'model_initialization_s': init_s, 'inference_wall_s': inference_wall_s,
               'batch_request_total_s': sum(b['total_s'] for b in batches),
               'throughput_fps': 690 / inference_wall_s,
               'amortized_ms_per_frame_not_latency': inference_wall_s / 690 * 1000,
               'total_wall_s': time.perf_counter() - overall_start,
               'first_progress_percent': all_rows[0]['progress_percent'],
               'last_progress_percent': all_rows[-1]['progress_percent'],
               'mean_generated_tokens': float(np.mean([r['generated_tokens'] for r in all_rows])),
               'token_cap_count': sum(r['hit_token_cap'] for r in all_rows)}
    write_json(out / 'summary.json', summary)
    fields = ['frame_index','timestamp_s','progress_percent','status','reference_state','batch_index','input_tokens','generated_tokens','num_cached_tokens','hit_token_cap']
    with (out / 'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    manifest.update(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'manifest.json',manifest)
    print('COMPLETED '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
