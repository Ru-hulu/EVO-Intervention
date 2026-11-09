import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import Qwen3VLConfig
import yaml

from robometer.models.rbm import RBM
from robometer_training.data import ObservationWindow, ProgressCollator
from robometer_training.evaluate import OnlinePredictor, predict
from robometer_training.model import (MODEL_PATH, attach_lora, load_processor,
                                     progress_loss, progress_targets, progress_values)
from robometer_training.train import RunMonitor, make_trainer


def test_window_prefix_and_episode_reset():
    window = ObservationWindow()
    snapshots = [window.push('a', i) for i in range(12)]
    for i, frames in enumerate(snapshots):
        assert frames == [max(0, j) for j in range(i - 7, i + 1)]
    assert window.push('b', 100) == [100] * 8
    assert snapshots[0] == [0] * 8


def test_c51_preserves_continuous_pt_and_endpoints():
    pt = torch.tensor([0, 0.1, 0.25, 0.5, 0.99, 1.0])
    targets = progress_targets(pt)
    torch.testing.assert_close(targets.sum(-1), torch.ones_like(pt))
    torch.testing.assert_close(targets @ torch.linspace(0, 1, 10), pt)
    assert targets[0, 0] == 1 and targets[-1, -1] == 1
    assert targets[3, 4] == targets[3, 5] == 0.5


def test_only_final_epoch_is_saved():
    monitor = RunMonitor()
    for step, expected in ((885, False), (1770, False), (2655, True)):
        state = SimpleNamespace(global_step=step, max_steps=2655)
        control = SimpleNamespace(should_save=True)
        assert monitor.on_epoch_end(None, state, control).should_save is expected


def test_train_eval_online_and_checkpoint_interfaces(tmp_path):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    processor = load_processor(min_pixels=4096, max_pixels=4096)
    settings = SimpleNamespace(**yaml.safe_load((Path(MODEL_PATH) / 'config.yaml').read_text())['model'])
    config = Qwen3VLConfig(
        text_config={'vocab_size': len(processor.tokenizer), 'hidden_size': 64, 'intermediate_size': 128,
                     'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 16, 'max_position_embeddings': 2048, 'use_cache': False,
                     'rope_scaling': {'rope_type': 'default', 'mrope_section': [2, 3, 3], 'mrope_interleaved': True}},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
                       'out_hidden_size': 64, 'patch_size': 16, 'spatial_merge_size': 2,
                       'temporal_patch_size': 2, 'num_position_embeddings': 16, 'deepstack_visual_indexes': []})
    config._attn_implementation = 'sdpa'
    model = RBM(config, processor, processor.tokenizer, base_model_id=settings.base_model_id, model_config=settings)
    untouched_base = copy.deepcopy(model)
    attach_lora(model, rank=2, alpha=4, dropout=0)
    images = []
    for i in range(9):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (64, 64), (25 * i, 30, 40)).save(path)
        images.append(str(path))
    rows = [{'id': 'a/0', 'episode_id': 'a', 'frame_index': 0, 'task': 'Press the button.',
             'pt': 0.0, 'image': [images[0]] * 8},
            {'id': 'a/8', 'episode_id': 'a', 'frame_index': 8, 'task': 'Press the button.',
             'pt': 0.61, 'image': images[1:9]}]
    data_path = tmp_path / 'data.jsonl'
    data_path.write_text('\n'.join(json.dumps(row) for row in rows))
    run_config = json.loads(Path(__file__).parents[1].joinpath('config.json').read_text())
    run_config.update(train_data=str(data_path), validation_data=str(data_path))
    run_config['eval_strategy'] = 'no'
    trainer = make_trainer(run_config, model, processor, output_dir=str(tmp_path / 'output'), use_cpu=True)
    collator = ProgressCollator(processor)
    batch = collator(rows)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    loss = trainer.compute_loss(model, batch)
    loss.backward()  # Random tiny CPU model only; no optimizer or training loop.
    gradients = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients.values())
    assert all('lora_' in name or name.startswith('progress_head.') for name in gradients)
    assert any('lora_B' in name and g.abs().max() > 0 for name, g in gradients.items())
    assert any(name.startswith('progress_head.') and g.abs().max() > 0 for name, g in gradients.items())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0, 0.01)
        model.progress_head[-1].bias.add_(torch.arange(10) / 100)
    model.eval()
    with torch.no_grad():
        training_loss, output = trainer.compute_loss(model, batch, return_outputs=True)
    values, logits = predict(model, collator, rows)
    torch.testing.assert_close(logits, output['logits'], rtol=0, atol=0)
    torch.testing.assert_close(training_loss, progress_loss(logits, batch['labels']))
    evaluation = trainer.evaluate()
    assert evaluation['eval_loss'] == pytest.approx(training_loss.item(), abs=1e-6)
    online = OnlinePredictor(model, collator)
    initial = online.push('a', rows[0]['task'], images[0])
    for image in images[1:9]:
        last = online.push('a', rows[0]['task'], image)
    assert initial == pytest.approx(values[0].item(), abs=2e-6)
    assert last == pytest.approx(values[1].item(), abs=2e-6)
    assert online.push('b', rows[0]['task'], images[0]) == pytest.approx(initial, abs=2e-6)
    checkpoint = tmp_path / 'adapter'
    trainer.save_model(str(checkpoint))
    assert all('lora_' in key for key in load_file(checkpoint / 'adapter_model.safetensors'))
    assert not (checkpoint / 'model.safetensors').exists()
    attach_lora(untouched_base, rank=2, alpha=4, dropout=0)
    parameter_ids = [id(p) for p in untouched_base.parameters()]
    trainer._load_from_checkpoint(checkpoint, model=untouched_base)
    assert parameter_ids == [id(p) for p in untouched_base.parameters()]
    restored = predict(untouched_base, collator, rows)[0]
    torch.testing.assert_close(restored, values, rtol=0, atol=0)
