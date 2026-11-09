"""Add real 101-bin model outputs to existing FOREST viewer evaluations."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device

BASE = Path('/cache/cvpzhr/evaluations')
EVALUATIONS = [
    ('forest-s01-val10-step13224-20261106', 8),
    ('forest-s01-val5-step13224-20261106', 4),
    ('forest-s01-val5-20261106', 4),
]
OUT = BASE/'forest-s01-bin-probabilities-20261106'


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', choices=[name for name, _ in EVALUATIONS])
    args = parser.parse_args()
    evaluations = [item for item in EVALUATIONS if not args.evaluation or item[0] == args.evaluation]
    status_file = OUT/(f'{args.evaluation}-status.json' if args.evaluation else 'status.json')
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    started = time.time()
    completed = []
    for name, batch_size in evaluations:
        evaluation = BASE/name
        selection = json.loads((evaluation/'selection.json').read_text())
        rows, step = selection['episodes'], selection['step']
        model, processor, cfg = load_checkpoint(str(evaluation/'checkpoint.pt'))
        dataset = M3Dataset([str(Path(r['cache_dir'])/'samples.jsonl') for r in rows], cfg['model']['history_dim'])
        assert dataset.history_source == cfg['history_source'] and dataset.history_capacity == cfg['history_capacity']
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
            multiprocessing_context='spawn', pin_memory=True,
            collate_fn=Stage3Collator(processor, dataset.history_dim, cfg['max_sequence_length'], dataset.history_capacity))
        points = [[] for _ in rows]
        offset, published = 0, set()
        def publish(ep):
            row = rows[ep]
            distribution = dict(checkpoint_step=step, checkpoint=str(evaluation/'checkpoint.pt'),
                bin_centers=np.linspace(0, 1, 101).tolist(), points=points[ep], batch_size=batch_size,
                source='原权重快照、原 batch 大小及顺序补算 · 5 Hz 因果历史 · 非采样帧保持上一推理分布，不插值')
            target = OUT/f"{name}-episode{row['episode_index']:06d}.json"
            dump(target, distribution)
            dataset_name = json.loads((evaluation/'metrics.json').read_text())['dataset']
            progress_file = evaluation/'viewer-data'/dataset_name/'progress'/f"{row['sample_id']}.json"
            progress = json.loads(progress_file.read_text())
            original = progress['prediction_points']
            assert [p[0] for p in original] == [p['frame_index'] for p in points[ep]]
            delta = max(abs(p[1]-q['predicted_pt'])*100 for p,q in zip(original, points[ep]))
            backup = progress_file.with_suffix('.before-bins.json')
            if not backup.exists(): backup.write_text(progress_file.read_text())
            progress['bin_distribution'] = dict(distribution, max_original_prediction_delta_pp=delta)
            dump(progress_file, progress)
            completed.append(dict(dataset=dataset_name, step=step, episode=row['episode_index'], observations=len(points[ep]), max_delta_pp=delta))
            dump(status_file, dict(phase='running', completed=completed, seconds=time.time()-started))
            print(json.dumps(dict(event='published', **completed[-1])), flush=True)
            published.add(ep)
        with torch.inference_mode():
            for batch in loader:
                result = model(to_device(batch['inputs'], 'cuda'), batch['target_pt'].to('cuda'))
                probs = result['bin_probs'].float().cpu().numpy()
                pred = result['pt'].float().cpu().numpy()
                assert probs.shape[1] == 101 and np.isfinite(probs).all()
                assert (probs >= 0).all() and np.allclose(probs.sum(-1), 1, atol=1e-6)
                assert np.allclose(probs @ np.linspace(0, 1, 101), pred, atol=1e-6)
                for probability, mean in zip(probs, pred):
                    ep, sample, _ = dataset.samples[offset]
                    points[ep].append(dict(frame_index=sample['frame_index'], probabilities=probability.tolist(), predicted_pt=float(mean)))
                    offset += 1
                for ep in range(len(rows)):
                    if ep not in published and points[ep] and (offset == len(dataset) or dataset.samples[offset][0] > ep):
                        publish(ep)
                if offset % 80 == 0:
                    print(json.dumps(dict(event='progress', dataset=name, done=offset, total=len(dataset), seconds=time.time()-started)), flush=True)
        assert offset == len(dataset) and len(published) == len(rows)
        del model, processor, result, loader
        gc.collect()
        torch.cuda.empty_cache()
    dump(status_file, dict(phase='completed', completed=completed, seconds=time.time()-started))


if __name__ == '__main__':
    main()
