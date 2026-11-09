"""Frozen, zero-shot Robometer inference. Labels are read only by the scorer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image
import torch

SOURCE = Path('/cache/robot-baselines/viewer-datasets/RMBench-CoverBlocks-40-20261105')
MODEL = Path('/cache/cvpzhr/models/Robometer-4B')
LOADER = Path('/cache/cvpzhr/code/robometer-proprimem-20261031')
sys.path.insert(0, str(LOADER))
sys.path.insert(0, str(LOADER/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'))
from robometer_training.model import load_model, load_processor, progress_batch, current_logits, progress_values

PROMPT = ('From the current viewpoint, cover the three blocks from left to right using the lids, '
          'then uncover the corresponding lids in the order red, green, and blue.')


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2))
    temp.replace(path)


def sample_frames(row):
    n, fps = row['total_frames'], row['fps']
    ann = json.loads((SOURCE/row['annotation_path']).read_text())
    frames = {0, n-1, *(min(n-1, round(t*fps)) for t in range(int((n-1)/fps)+1))}
    for stage in ann['stages']:
        for key in ('start_frame', 'end_frame', 'physical_completion_frame'):
            if stage.get(key) is not None:
                f = int(stage[key])
                frames.update([max(0, min(n-1, f-1)), max(0, min(n-1, f))])
    gt = json.loads((SOURCE/'progress'/(row['sample_id']+'.json')).read_text())['dense_progress']
    for f in range(1, n):
        if gt[f]-gt[f-1] < -.05:
            frames.update([f-1, f, min(n-1, f+round(fps))])
    return sorted(frames), gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output)
    (out/'predictions').mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (SOURCE/'metadata.jsonl').read_text().splitlines() if s.strip()]
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.43)
    write(out/f'manifest-{args.rank}.json', dict(model=str(MODEL), adapter=None, frozen=True, rl_training=False,
        source=str(SOURCE), prompt=PROMPT, camera='main_view', sampling='1 Hz + annotation boundaries + failure events',
        history='8 uniformly spaced frames from the observed prefix; repeated frames when prefix has fewer than 8 frames; includes first and current',
        inference='last progress token; official softmax bin expectation; no smoothing or calibration',
        pixels=dict(min=4096, max=65536), batch_size=args.batch_size, gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
        model_index_sha256=hashlib.sha256((MODEL/'model.safetensors.index.json').read_bytes()).hexdigest()))
    status_path = out/f'status-{args.rank}.json'
    write(status_path, dict(status='loading_model', total=40))
    processor = load_processor(MODEL, min_pixels=4096, max_pixels=65536)
    model, _ = load_model(MODEL, processor)
    model.requires_grad_(False).to('cuda').eval()
    start = time.monotonic()
    for i, row in enumerate(rows):
        if i % args.shards != args.rank:
            continue
        target = out/'predictions'/(row['sample_id']+'.json')
        if target.exists():
            continue
        anchors, gt = sample_frames(row)
        histories = [np.round(np.linspace(0, f, 8)).astype(int).tolist() for f in anchors]
        wanted = set(f for h in histories for f in h)
        frames = {}
        f = 0
        with av.open(str(SOURCE/row['video_path'])) as cap:
            cap.streams.video[0].thread_count = 1
            for decoded in cap.decode(video=0):
                if f in wanted:
                    frames[f] = decoded.to_image()
                f += 1
        assert f == row['total_frames'], (row['sample_id'], f, row['total_frames'])
        predictions = []
        for b in range(0, len(anchors), args.batch_size):
            examples = [(PROMPT, [frames[f] for f in h]) for h in histories[b:b+args.batch_size]]
            inputs = progress_batch(processor, examples)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = current_logits(model, inputs)
                values = progress_values(logits).cpu().tolist()
            assert all(np.isfinite(p) and 0 <= p <= 1 for p in values)
            for j, p in enumerate(values):
                k = b+j
                predictions.append(dict(frame=anchors[k], seconds=anchors[k]/row['fps'],
                    predicted_progress=p, annotated_progress=gt[anchors[k]], history_frames=histories[k]))
            write(status_path, dict(status='predicting', trajectory_index=i, total=40,
                sample=row['sample_id'], completed_points=len(predictions), total_points=len(anchors),
                elapsed_seconds=time.monotonic()-start))
        write(target, dict(sample_id=row['sample_id'], fps=row['fps'], total_frames=f,
            task_success=row['task_success'], batch_size=args.batch_size, predictions=predictions))
        print(json.dumps(dict(completed=i+1, sample=row['sample_id'], points=len(predictions),
                             final_prediction=predictions[-1]['predicted_progress'])), flush=True)
    write(status_path, dict(status='complete', rank=args.rank, shards=args.shards,
                                 elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
