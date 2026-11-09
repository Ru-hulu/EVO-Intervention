#!/usr/bin/env python3
"""Re-evaluate both RoboBrain conditions on the same clip at 3 FPS."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime

BASE = Path('/root/autodl-tmp/EvalProgress')
DATASET = Path('/root/autodl-tmp/Datasets/piperx_insert_copper_screw_ep000000_0_23s')
TASK = 'Insert one copper screw into a black sleeve, then place the assembled component into the box.'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0')
sys.path.insert(0, str(BASE / 'RoboBrain2.5/scripts'))
import annotate_screw as annotate
import torch

torch.set_num_threads(8)
original_loader = annotate.load_backend
backend = None
load_seconds = 0.0

def reuse_backend(*args):
    global backend, load_seconds
    if backend is None:
        started = time.perf_counter()
        backend = original_loader(*args)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
    return backend

annotate.load_backend = reuse_backend
run_id = 'single_insertions_3fps_' + datetime.now().strftime('%Y%m%d_%H%M%S')
expected = list(range(10, 690, 10)) + [689]
for variant in ['blank', 'lastframe']:
    output = BASE / 'outputs' / DATASET.name / 'RoboBrain2.5' / (run_id + '_' + variant)
    sys.argv = ['annotate_screw.py', '--dataset-root', str(DATASET), '--episodes', '0',
                '--instruction', TASK, '--output-dir', str(output),
                '--frame-interval', '10', '--batch-size', '8']
    sys.argv += ['--goal-last-frame'] if variant == 'lastframe' else [
        '--goal-image', str(BASE / 'RoboBrain2.5/scripts/references/blank.png')]
    print('START', variant, output, flush=True)
    started = time.perf_counter()
    annotate.main()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows = [json.loads(line) for line in (output / 'episode_000000.jsonl').read_text().splitlines()]
    assert [r['frame_index'] for r in rows] == expected
    assert all(r['status'] == 'ok' and r['instruction'] == TASK and
               abs(r['timestamp_s'] - r['frame_index'] / 30) < 1e-8 and
               r['before_frame_index'] == 0 for r in rows)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    prior = BASE / 'outputs' / DATASET.name / 'RoboBrain2.5' / ('single_insertions_20261015_152852_' + variant)
    previous = json.loads((prior / 'manifest.json').read_text())
    for key in ['model_path', 'batch_size', 'backend', 'dataset_root', 'camera_order',
                'goal_image', 'instruction_override', 'reference_end', 'prompt_sha256']:
        assert manifest[key] == previous[key], key
    manifest.update(task_scope='single', target_insertions=1,
                    source_clip=json.loads((DATASET / 'meta/clip_provenance.json').read_text()),
                    fps_source='User requested 3 FPS: every 10 native frames at 30 FPS, plus terminal frame; frame 0 is the before/reference frame.',
                    replaces_1fps_run=str(prior),
                    generation=dict(do_sample=True, temperature=0.1, top_p=0.9, top_k=50, max_new_tokens=1024),
                    annotator_sha256=hashlib.sha256(Path(annotate.__file__).read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    summary = dict(model='RoboBrain2.5', variant=variant, task_scope='single',
                   dataset_id=DATASET.name, episode_index=0, points=len(rows),
                   statuses={'ok': len(rows)}, sample_fps=3, frame_interval=10,
                   elapsed_s=elapsed,
                   model_load_s=load_seconds if variant == 'blank' else 0,
                   elapsed_excluding_model_load_s=elapsed - (load_seconds if variant == 'blank' else 0),
                   last_time_s=rows[-1]['timestamp_s'], instruction=TASK,
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
                   timing_note='Wall time includes video decoding, preprocessing, batched generation and result writing; not single-request latency.')
    (output / 'completion.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    shutil.copy2(__file__, output / 'rerun_robobrain_3fps.py')
    shutil.copy2(annotate.__file__, output / 'annotate_screw.py')
    print('COMPLETE', str(output), json.dumps(summary), flush=True)
