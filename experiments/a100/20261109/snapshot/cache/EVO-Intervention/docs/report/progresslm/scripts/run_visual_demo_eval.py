#!/usr/bin/env python3
"""Batch-one Visual Demo evaluation using the installed official ProgressLM prompt/model.

No query history, cross-request prefix cache, temporal smoothing, or forced endpoints.
The explicit generate path mirrors Qwen2VLChat.generate_inner and adds timing only.
"""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")
REPO = Path('/root/autodl-tmp/EvalProgress/ProgressLM')
CORE = REPO / 'eval/qwen25vl/codes'
MODEL = Path('/root/autodl-tmp/ckpt/baselines/ProgressLM')
INPUT_ROOT = Path('/root/autodl-tmp/EvalProgress/Robo-Dopamine/local_setup/bolt_inputs_20261015')
QUERY_ROOT = INPUT_ROOT / 'query_frames/cam_high'
PLAN = REPO / 'local_setup/bolt_visual_reference_plan_20261015/proposed_plan.json'


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_response(text):
    tags = {tag: re.findall(fr'<{tag}>\s*(.*?)\s*</{tag}>', text, re.I | re.S)
            for tag in ['ref_think', 'ref', 'score_think', 'score']}
    result = {'progress_percent': None, 'status': 'parse_error',
              'reference_state': tags['ref'][0] if len(tags['ref']) == 1 else None,
              'format_valid': all(len(v) == 1 for v in tags.values())}
    if len(tags['score']) != 1:
        return result
    value = tags['score'][0].strip()
    result['score_text'] = value
    if value.lower() in ('n/a', 'na'):
        result['status'] = 'not_applicable'
        return result
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?', value):
        return result
    # The official visual prompt explicitly requests 0..100 percent.
    # In particular, a bare "1" means 1%, not 100%.
    number = float(value.rstrip('%').strip())
    if not math.isfinite(number) or not 0 <= number <= 100:
        result['status'] = 'out_of_range'
    else:
        result.update(progress_percent=number, status='ok')
    return result


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--output', type=Path, required=True)
    opts = args.parse_args()
    out = opts.output
    out.mkdir(parents=True, exist_ok=False)
    run_started = time.perf_counter()
    plan = json.loads(PLAN.read_text())
    assert [x['source_frame_index'] for x in plan['references']] == [720, 870, 1560, 1680, 1740]
    assert [x['progress'] for x in plan['references']] == [0, 25, 50, 75, 100]
    plan.update(status='CONFIRMED_BY_USER', inference_started=False,
                confirmed_boundary='Frame720 at24s is 0%; frame689 is excluded from references.')
    write_json(out / 'confirmed_plan.json', plan)
    shutil.copy2(__file__, out / 'run_visual_demo_eval.py')
    (out / 'official_source').mkdir()
    for src in [CORE / 'core/model.py', CORE / 'prompts/visual_demo_prompt.py']:
        shutil.copy2(src, out / 'official_source' / src.name)
    for name in ['config.json', 'generation_config.json', 'preprocessor_config.json']:
        shutil.copy2(MODEL / name, out / 'official_source' / name)
    shutil.copy2(PLAN.parent / 'proposed_reference_top.jpg', out / 'reference_top.jpg')
    shutil.copy2(PLAN.parent / 'proposed_reference_three_views.jpg', out / 'reference_three_views.jpg')
    (out / 'reference_frames').mkdir()
    refs = []
    for ref in plan['references']:
        src = Path(ref['path'])
        assert sha(src) == ref['sha256']
        dest = out / 'reference_frames' / src.name
        shutil.copy2(src, dest)
        refs.append(str(dest))

    # Validate all cached current images against sequential exact-PTS video decoding.
    import av
    import numpy as np
    from PIL import Image
    source_inputs = json.loads((INPUT_ROOT / 'inputs.json').read_text())
    query_source = source_inputs['query_source_videos'][0]
    assert sha(query_source['path']) == query_source['sha256']
    frame_records = []
    with av.open(query_source['path']) as video:
        for i, frame in enumerate(video.decode(video=0)):
            assert i < 690 and abs(float(frame.pts * frame.time_base) - i / 30) < 1e-5
            p = QUERY_ROOT / f'frame_{i:06d}.png'
            with Image.open(p) as cached:
                assert cached.size == (640, 480)
                assert np.array_equal(np.asarray(cached.convert('RGB')), frame.to_ndarray(format='rgb24')), i
            frame_records.append({'index': i, 'timestamp_s': i / 30, 'path': str(p), 'sha256': sha(p)})
    assert len(frame_records) == 690
    write_json(out / 'verified_query_inputs.json', {'source': query_source, 'frames': frame_records})
    print('INPUT CHECK OK: 690 exact-PTS images, 5 confirmed reference images.', flush=True)

    sys.path.insert(0, str(CORE))
    import torch
    from qwen_vl_utils import process_vision_info
    from core.model import Qwen2VLChat
    from prompts.visual_demo_prompt import build_visual_demo_prompt, VISUAL_DEMO_SYSTEM_PROMPT
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    chat = Qwen2VLChat(model_path=str(MODEL), max_new_tokens=2048,
                      temperature=0, top_p=1, top_k=1, use_custom_prompt=False,
                      system_prompt=VISUAL_DEMO_SYSTEM_PROMPT)
    chat.generate_kwargs = {'max_new_tokens': 2048, 'do_sample': False, 'use_cache': True}
    chat.model.eval()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_started
    manifest = {
        'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': plan['model'], 'model_path': str(MODEL), 'repository': str(REPO),
        'repo_head': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'runtime': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'qwen-vl-utils', 'av', 'Pillow']},
        'gpu': torch.cuda.get_device_name(), 'dtype': str(next(chat.model.parameters()).dtype),
        'attention': chat.model.config._attn_implementation,
        'query_frames': 690, 'source_fps': 30, 'query_history': False,
        'reference_frames': plan['references'], 'task': plan['task'],
        'images_per_call': 6, 'batch_size': 1, 'reference_cache_across_requests': False,
        'generation': chat.generate_kwargs, 'pixel_limits_override': None,
        'image_preprocessing': 'checkpoint/default qwen-vl-utils; original640x480; actual sizes logged',
        'source_configuration_note': plan['runtime_proposal']['note'],
        'output_scale': 'percent0..100, bare numeric output interpreted as percent',
        'postprocessing': 'No clamping, smoothing, interpolation, monotonicity, or forced endpoints',
        'timing': plan['timing_plan'], 'model_initialization_s': model_load_s,
        'script_sha256': sha(__file__),
    }
    write_json(out / 'manifest.json', manifest)
    print(f'MODEL LOADED in {model_load_s:.3f}s', flush=True)
    eos = chat.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def request(index):
        torch.cuda.synchronize()
        start = time.perf_counter()
        parts = build_visual_demo_prompt(plan['task'], refs, 4, str(QUERY_ROOT / f'frame_{index:06d}.png'))
        assert [x['value'] for x in parts if x['type'] == 'image'] == refs + [str(QUERY_ROOT / f'frame_{index:06d}.png')]
        messages = [{'role': 'system', 'content': chat.system_prompt},
                    {'role': 'user', 'content': chat._prepare_content(parts)}]
        text = chat.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        assert len(images) == 6 and videos is None
        inputs = chat.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            generated = chat.model.generate(**inputs, **chat.generate_kwargs)
        torch.cuda.synchronize()
        generated_at = time.perf_counter()
        output_ids = generated[0, inputs.input_ids.shape[1]:]
        raw = chat.processor.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        parsed = parse_response(raw)
        ended_eos = bool(output_ids.numel()) and int(output_ids[-1]) in eos
        capped = output_ids.numel() >= 2048 and not ended_eos
        if capped:
            parsed['parsed_progress_before_truncation_check'] = parsed['progress_percent']
            parsed.update(progress_percent=None, status='truncated')
        torch.cuda.synchronize()
        end = time.perf_counter()
        row = {'frame_index': index, 'timestamp_s': index / 30, **parsed,
               'latency_ms': (end - start) * 1000,
               'preprocess_ms': (prepared - start) * 1000,
               'generation_ms': (generated_at - prepared) * 1000,
               'decode_parse_ms': (end - generated_at) * 1000,
               'input_tokens': inputs.input_ids.shape[1], 'generated_tokens': output_ids.numel(),
               'ended_eos': ended_eos, 'hit_token_cap': capped,
               'image_sizes': [list(image.size) for image in images],
               'image_grid_thw': inputs.image_grid_thw.cpu().tolist(),
               'raw_response': raw}
        return row, messages, text[0]

    warmups = []
    for k in range(3):
        row, messages, rendered = request(0)
        warmups.append(row)
        print(f'WARMUP {k+1}/3: {row["latency_ms"]:.1f}ms, status={row["status"]}, tokens={row["generated_tokens"]}', flush=True)
        if k == 0:
            write_json(out / 'example_messages.json', messages)
            (out / 'example_rendered_prompt.txt').write_text(rendered)
    write_json(out / 'warmup_results.json', warmups)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with (out / 'predictions.jsonl').open('x') as sink:
        for i in range(690):
            row, _, _ = request(i)
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            sink.flush()
            status = {'completed': len(rows), 'total': 690, 'last_frame': i,
                      'mean_latency_ms': float(np.mean([r['latency_ms'] for r in rows])),
                      'status_counts': dict(collections.Counter(r['status'] for r in rows)),
                      'last_progress_percent': row['progress_percent'],
                      'elapsed_s': time.perf_counter() - run_started}
            write_json(out / 'running_status.json', status)
            if i % 10 == 0 or i == 689:
                print(json.dumps(status), flush=True)
    lat = np.array([r['latency_ms'] for r in rows])
    valid = [r for r in rows if r['status'] == 'ok']
    summary = {'status': 'completed', 'frames': len(rows), 'valid_frames': len(valid),
               'status_counts': dict(collections.Counter(r['status'] for r in rows)),
               'mean_latency_ms': float(lat.mean()), 'p50_latency_ms': float(np.percentile(lat, 50)),
               'p95_latency_ms': float(np.percentile(lat, 95)), 'min_latency_ms': float(lat.min()),
               'max_latency_ms': float(lat.max()), 'effective_fps': 1000 / float(lat.mean()),
               'summed_inference_s': float(lat.sum() / 1000),
               'mean_preprocess_ms': float(np.mean([r['preprocess_ms'] for r in rows])),
               'mean_generation_ms': float(np.mean([r['generation_ms'] for r in rows])),
               'mean_decode_parse_ms': float(np.mean([r['decode_parse_ms'] for r in rows])),
               'mean_generated_tokens': float(np.mean([r['generated_tokens'] for r in rows])),
               'max_generated_tokens': max(r['generated_tokens'] for r in rows),
               'token_cap_count': sum(r['hit_token_cap'] for r in rows),
               'format_invalid_count': sum(not r['format_valid'] for r in rows),
               'first_progress_percent': rows[0]['progress_percent'],
               'last_progress_percent': rows[-1]['progress_percent'],
               'model_initialization_s': model_load_s,
               'warmup_s': sum(r['latency_ms'] for r in warmups) / 1000,
               'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
               'total_wall_s': time.perf_counter() - run_started}
    write_json(out / 'summary.json', summary)
    fields = ['frame_index', 'timestamp_s', 'progress_percent', 'status', 'reference_state',
              'latency_ms', 'preprocess_ms', 'generation_ms', 'decode_parse_ms',
              'input_tokens', 'generated_tokens', 'hit_token_cap', 'format_valid']
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'manifest.json', manifest)
    print('COMPLETED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
