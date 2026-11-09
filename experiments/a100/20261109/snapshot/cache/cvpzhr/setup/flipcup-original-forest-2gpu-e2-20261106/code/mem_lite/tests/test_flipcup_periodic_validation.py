"""Validate cadence, sample-weighted distributed loss, and uninterrupted training RNG/mode."""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mem_lite.periodic_validation import validate_during_training, validation_points


def test_fifth_epoch_boundaries_do_not_drift_across_three_epochs():
    points = validation_points(508, 0.2)
    assert list(points) == [102, 204, 305, 407, 508]
    assert list(points.values()) == [.2, .4, .6, .8, 1.0]
    assert [e * 508 + step for e in range(3) for step in points] == [
        102, 204, 305, 407, 508, 610, 712, 813, 915, 1016, 1118, 1220, 1321, 1423, 1524]
    for step, fraction in points.items():
        assert fraction <= step / 508 < fraction + 1 / 508


@pytest.mark.parametrize('steps,interval', [(508, 0), (508, 1.1), (2, .2)])
def test_invalid_or_unrepresentable_intervals_are_rejected(steps, interval):
    with pytest.raises(ValueError):
        validation_points(steps, interval)


class ProbeDataset:
    def __init__(self):
        self.episodes = [(Path('/fixture/single'), {'source': {'task': 'normal', 'source_episodes': [1]}}),
                         (Path('/fixture/triple'), {'source': {'task': 'normal', 'source_episodes': [1, 2, 3]}})]
        self.samples = [(0 if i < 2 else 1, {'frame_index': i * 6}, None) for i in range(5)]

    def __len__(self):
        return len(self.samples)


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3))

    def forward(self, inputs, target):
        assert not self.training and not torch.is_grad_enabled()
        # Deliberately consume RNG: validation must restore it before training resumes.
        logits = self.logits.expand(len(target), -1) + torch.rand(()) * 0
        return {'bin_logits': logits, 'pt': (logits.softmax(-1) * torch.tensor([0., .5, 1.])).sum(-1)}


def run_probe(directory, rank=0, world=1):
    dataset, model = ProbeDataset(), ProbeModel().train()
    targets = torch.tensor([0., .25, .5, .75, 1.])
    indices = list(range(rank, len(dataset), world))
    batches = [indices[i:i + 2] for i in range(0, len(indices), 2)]
    loader = [{'inputs': {}, 'target_pt': targets[batch]} for batch in batches]
    torch.manual_seed(42 + rank)
    before_rng = torch.get_rng_state().clone()
    before_weights = model.logits.detach().clone()
    metrics = validate_during_training(model, dataset, loader, batches, directory,
                                      step=102, epoch=.2, epoch_actual=102/508,
                                      checkpoint_name='step-000102.pt', training_loss=2.0,
                                      rank=rank, world_size=world)
    assert model.training and model.logits.grad is None
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(before_weights, model.logits)
    assert metrics['num_samples'] == 5
    assert metrics['loss'] == pytest.approx(math.log(3))
    assert metrics['mae_pp'] == pytest.approx(30)
    assert metrics['rmse_pp'] == pytest.approx(100 * math.sqrt(.125))
    assert metrics['bias_pp'] == pytest.approx(0)
    assert metrics['groups']['normal/n1']['num_samples'] == 2
    assert metrics['groups']['normal/n1']['mae_pp'] == pytest.approx(37.5)
    assert metrics['groups']['normal/n3']['num_samples'] == 3
    assert metrics['groups']['normal/n3']['mae_pp'] == pytest.approx(25)


def test_validation_restores_training_mode_rng_and_weights(tmp_path):
    run_probe(tmp_path)
    rows = [json.loads(s) for s in (tmp_path/'validation/step-000102/predictions.jsonl').read_text().splitlines()]
    assert [r['index'] for r in rows] == list(range(5))
    assert json.loads((tmp_path/'validation-status.json').read_text())['phase'] == 'completed'


def distributed_probe(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        run_probe(directory, rank, 2)
    finally:
        dist.destroy_process_group()


def test_distributed_validation_does_not_pad_or_duplicate_unequal_shards(tmp_path):
    mp.spawn(distributed_probe, args=(str(tmp_path/'gloo'), str(tmp_path)), nprocs=2, join=True)
    rows = [json.loads(s) for s in (tmp_path/'validation/step-000102/predictions.jsonl').read_text().splitlines()]
    assert [r['index'] for r in rows] == list(range(5))
