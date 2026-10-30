"""Check prepared data and one real GPU batch without an optimizer or training loop."""
import argparse
from collections import defaultdict
import copy
import gc
import json
from pathlib import Path
import tempfile
import time

import torch
from transformers import set_seed

from .data import ProgressCollator, ProgressDataset, answer_text
from .model import OnlineProgressPredictor, load_model, predict_batch, save_adapter, to_device
from .window import ObservationWindow, history_window


def check_data(config):
    splits, episodes = {}, {}
    for split in ('train', 'validation'):
        rows = ProgressDataset(config[f'{split}_data']).rows
        by_episode = defaultdict(list)
        for row in rows:
            by_episode[row['episode_id']].append(row)
        for episode, sequence in by_episode.items():
            assert not episode.startswith('03-')
            frames = [(r['frame_index'], r['image'][-1]) for r in sequence]
            online = ObservationWindow(8)
            for index, row in enumerate(sequence):
                window = online.push(episode, frames[index])
                assert window == history_window(frames, index, 8)
                assert row['image'] == [p for _, p in window]
                assert row['window_frame_indices'] == [f for f, _ in window]
                assert all(f <= row['frame_index'] for f, _ in window)
                assert row['supervision_eligible'] == bool(row['remaining_actions'] or row['pt'] == 1)
        episodes[split] = set(by_episode)
        splits[split] = {'videos': len(by_episode), 'samples': len(rows),
                         'loss_samples': sum(row['supervision_eligible'] for row in rows)}
    assert not episodes['train'] & episodes['validation']
    return splits


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'data': check_data(config), 'training_started': False, 'optimizer_steps': 0}
    set_seed(config['seed'])
    model, processor = load_model(config, trainable=True, device='cuda:0')
    collator = ProgressCollator(processor, config['max_length'])
    rows = ProgressDataset(config['train_data'], supervised=True).rows
    # Long answers with a full eight-frame visual input exercise the expensive batch shape.
    rows.sort(key=lambda r: len(processor.tokenizer.encode(r['task'] + answer_text(r['remaining_actions'], r['pt']))), reverse=True)
    group = rows[:config['training']['per_device_train_batch_size']]
    inputs = to_device(collator(group), model)
    versions = {name: p._version for name, p in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(**inputs)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    torch.cuda.synchronize()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients.values())
    assert all(any(kind in name for kind in ('lora_', 'pooler.', 'value_head.')) for name in gradients)
    assert not any('visual' in name for name in gradients)
    for kind in ('lora_', 'pooler.', 'value_head.'):
        assert any(kind in name and bool(torch.count_nonzero(g)) for name, g in gradients.items())
    assert all(p._version == versions[name] for name, p in model.named_parameters())
    report['batch_check'] = {'batch_size': len(group), 'sequence_length': inputs['input_ids'].shape[1],
        'loss': float(result.loss.detach()),
        'gradient_norm': float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gradients.values()]))),
        'all_gradients_finite': True, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'weights_unchanged_after_backward': True}
    print(json.dumps(report['batch_check']), flush=True)
    model.zero_grad(set_to_none=True)
    del result, inputs, gradients
    gc.collect()
    torch.cuda.empty_cache()

    inference_config = copy.deepcopy(config)
    inference_config['inference']['max_new_tokens'] = 256
    validation = ProgressDataset(config['validation_data']).rows
    row = next(row for row in validation if len(set(row['window_frame_indices'])) == 8)
    expected = predict_batch(model, collator, [row], **inference_config['inference'])[0]
    assert expected['predicted_pt'] is not None and 0 <= expected['predicted_pt'] <= 1
    online = OnlineProgressPredictor(model, processor, inference_config)
    for frame, image in zip(row['window_frame_indices'][:-1], row['image'][:-1]):
        online.window.push(row['episode_id'], (frame, image))
    actual = online.push_sample(row['episode_id'], row['frame_index'], row['image'][-1], row['task'])
    assert actual['window_frame_indices'] == row['window_frame_indices']
    assert actual['generated_text'] == expected['generated_text']
    assert abs(actual['predicted_pt'] - expected['predicted_pt']) < 1e-6
    del online
    # This unmodified temporary adapter is only an interface check, never a training checkpoint.
    with tempfile.TemporaryDirectory(prefix='adapter-check-', dir=output) as directory:
        save_adapter(model, processor, directory, config)
        del model, collator, processor
        gc.collect()
        torch.cuda.empty_cache()
        model, processor = load_model(config, adapter=directory, device='cuda:0')
        reloaded = predict_batch(model, ProgressCollator(processor), [row], **inference_config['inference'])[0]
        assert reloaded['generated_text'] == expected['generated_text']
        assert abs(reloaded['predicted_pt'] - expected['predicted_pt']) < 1e-6
    report['inference_check'] = {'episode_id': row['episode_id'], 'frame_index': row['frame_index'],
        'window_frame_indices': row['window_frame_indices'], 'online_offline_match': True,
        'checkpoint_reload_match': True, 'max_new_tokens_for_check': 256,
        'untrained_prediction_for_interface_check_only': expected}
    report['status'] = 'passed'
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()), args.output_dir), ensure_ascii=False, indent=2))
