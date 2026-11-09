"""Check original Forest initialization and fractional archive behavior."""
from unittest.mock import patch

import pytest
import torch

from mem_lite.tests.test_m3_predictor import model as model
from mem_lite.train_stage3 import initialize_weights, save_checkpoint, trainable_state
from mem_lite.periodic_validation import validation_points


def test_original_forest_weights_only_initialization(model, tmp_path):
    config = dict(history_dim=16, num_bins=101, lora_rank=2, lora_alpha=4, lora_dropout=0)
    expected = {part: {key: value.clone() for key, value in values.items()}
                for part, values in trainable_state(model).items()}
    assert set(expected) == {'m3_lora', 'history_projection', 'progress_head'}
    checkpoint = dict(format='mem-lite-streamforest-v1', model=expected, config={'model': config},
                      epoch=1, step=19934, optimizer={'not_to_load': True})
    path = tmp_path / 'epoch1.pt'
    torch.save(checkpoint, path)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    info = initialize_weights(model, path, config)
    assert (info['epoch'], info['step']) == (1, 19934)
    assert not optimizer.state
    for part, values in trainable_state(model).items():
        for key, value in values.items():
            torch.testing.assert_close(value, expected[part][key])
    with pytest.raises(ValueError, match='lora_rank'):
        initialize_weights(model, path, {**config, 'lora_rank': 8})
    checkpoint['format'] = 'mem-lite-streamforest-crossattn-v2'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='Forest v1'):
        initialize_weights(model, path, config)


def test_each_fifth_epoch_preserves_a_checkpoint(model, tmp_path):
    points = validation_points(1241, 0.2)
    assert list(points.values()) == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert list(points) == [249, 497, 745, 993, 1241]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    with patch('mem_lite.train_stage3.torch.cuda.get_rng_state', return_value=torch.tensor([1], dtype=torch.uint8)):
        for step in [249, 497]:
            save_checkpoint(model, optimizer, {'keep_only_latest': True}, tmp_path, 0, step, step,
                            archive_name=f'step-{step:06d}.pt')
    assert {p.name for p in tmp_path.iterdir()} == {'last.pt', 'step-000249.pt', 'step-000497.pt'}
    assert torch.load(tmp_path/'step-000249.pt', weights_only=True)['format'] == 'mem-lite-streamforest-v1'
