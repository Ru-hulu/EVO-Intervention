"""Audit all windows; optionally run a small, read-only 4B CPU evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .data import ProgressCollator, ProgressDataset
from .evaluate import predict
from .model import attach_lora, load_model, load_processor, progress_loss
from .train import make_trainer


def audit(config):
    root = Path(config['train_data']).parent
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['sample_fps'] == config['sample_fps'] == 5
    assert manifest['window_size'] == config['window_size'] == 8
    episode_sets, report = [], {}
    for split in ('train', 'validation'):
        dataset = ProgressDataset(root / f'{split}.jsonl')
        source = ProgressDataset(Path(manifest['source']) / f'{split}.jsonl')
        original = {r['id']: r for r in source.rows}
        assert {r['id'] for r in dataset.rows} == original.keys()
        assert hashlib.sha256((root / f'{split}.jsonl').read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
        episodes, history, current_episode = set(), [], None
        for row in dataset.rows:
            previous = original[row['id']]
            if row['episode_id'] != current_episode:
                history, current_episode = [], row['episode_id']
            history.append(row['frame_index'])
            expected = [history[max(0, i)] for i in range(len(history) - 8, len(history))]
            assert row['history_frame_indices'] == expected
            assert all(frame <= row['frame_index'] for frame in expected)
            assert len(row['image']) == 8 and row['pt'] == previous['pt']
            assert os.path.samefile(row['image'][-1], previous['image'][5])
            for image, index in zip(row['image'], expected, strict=True):
                assert Path(image).stem == f'{index:08d}' and Path(image).is_file()
                assert Path(image).parent == Path(row['image'][-1]).parent
            episodes.add(row['episode_id'])
        episode_sets.append(episodes)
        report[split] = {'samples': len(dataset), 'videos': len(episodes)}
    assert not episode_sets[0] & episode_sets[1]
    return {'splits': report, 'all_windows_causal': True, 'original_pt_preserved': True,
            'images_hardlinked': manifest['linked_images'], 'gpu_used': False, 'optimizer_steps': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--real-forward', action='store_true')
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("Run checks with CUDA_VISIBLE_DEVICES='' to keep the shared GPU untouched")
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    report = audit(config)
    print(json.dumps(report, indent=2), flush=True)
    if args.real_forward:
        processor = load_processor(config['model_path'], min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
        model, _ = load_model(config['model_path'], processor)
        attach_lora(model, **config['lora'])
        rows = ProgressDataset(config['validation_data']).rows
        selected = [rows[0], max(rows, key=lambda row: row['frame_index'])]
        collator = ProgressCollator(processor, config['max_length'])
        values, logits = predict(model, collator, selected)
        target = torch.tensor([r['pt'] for r in selected])
        direct_loss = progress_loss(logits, target).item()
        with tempfile.TemporaryDirectory(prefix='robometer-readonly-eval-') as directory:
            trainer = make_trainer(config, model, processor, output_dir=directory, use_cpu=True)
            evaluation = trainer.evaluate(eval_dataset=selected)
        assert abs(evaluation['eval_loss'] - direct_loss) < 1e-5
        assert model.model.visual.training is False
        assert torch.isfinite(values).all()
        report['real_cpu_forward'] = {'samples': len(selected), 'ids': [r['id'] for r in selected],
                                       'loss': direct_loss, 'predictions': values.tolist(),
                                       'trainer_evaluation': evaluation,
                                       'training_eval_inference_loss_consistent': True}
    report['status'] = 'prepared_cpu_checked_gpu_preflight_pending'
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
