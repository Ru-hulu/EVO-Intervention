import copy
import math
import torch
import pytest
from mem_lite.train_stage3 import build_scheduler


def test_cosine_resume_matches_uninterrupted_updates():
    config = dict(lr_schedule='cosine', warmup_ratio=.03, epochs=1, gradient_accumulation_steps=3)
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([p], lr=1e-4)
    schedule = build_scheduler(optimizer, config, 301)
    assert config['total_optimizer_steps'] == 101
    assert config['warmup_steps'] == 4
    rates = []
    for step in range(101):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step(); schedule.step()
        if step == 36:
            checkpoint = copy.deepcopy((optimizer.state_dict(), schedule.state_dict()))
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    resumed = build_scheduler(other, config, 301)
    other.load_state_dict(checkpoint[0]); resumed.load_state_dict(checkpoint[1])
    actual = []
    for step in range(37, 101):
        actual.append(other.param_groups[0]['lr'])
        other.step(); resumed.step()
    assert actual == pytest.approx(rates[37:], abs=1e-15)
    assert rates[0] == 0 and rates[4] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]['lr'] == 0


def test_schedule_optional_and_invalid_protocol_rejected():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    assert build_scheduler(optimizer, {}, 10) is None
    assert optimizer.param_groups[0]['lr'] == 1e-4
    with pytest.raises(ValueError):
        build_scheduler(optimizer, {'lr_schedule':'cosine', 'warmup_ratio':1}, 10)
