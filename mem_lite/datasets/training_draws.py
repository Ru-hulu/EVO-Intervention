"""Frozen, rank-sharded training draws with per-draw history dropout.

Input caches remain immutable. Draws are fixed before training, so worker
prefetching, rank scheduling and checkpoint replay cannot change dropout.
"""
import hashlib
import json
from pathlib import Path


class DrawDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        for name in ('signature', 'history_dim', 'history_capacity', 'history_source'):
            setattr(self, name, getattr(dataset, name))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, draw):
        index, drop, near = draw
        sample = dict(self.dataset[index])
        had_history = sample['history_tokens'] is not None
        if drop:
            # Before M3Collator: remove tokens AND their sequence placeholders.
            sample['history_tokens'] = None
        sample['sampling_audit'] = [1, int(drop), int(near), int(had_history and drop)]
        return sample


class DrawSampler:
    def __init__(self, plan, rank):
        self.plan, self.rank, self.epoch = plan, rank, 0

    def set_epoch(self, epoch):
        if not 0 <= epoch < len(self.plan['epochs']):
            raise ValueError('Epoch outside the frozen draw plan')
        self.epoch = epoch

    def __len__(self):
        return self.plan['draws_per_epoch'] // self.plan['world_size']

    def __iter__(self):
        for draw in self.plan['epochs'][self.epoch]['draws'][self.rank::self.plan['world_size']]:
            index = draw['index']
            yield index, draw['drop_history'], self.plan['samples'][index]['near_anomaly']


def load_training_draws(dataset, config, world_size, rank):
    reference = config['training_draws']
    raw = Path(reference['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Training draw plan hash changed')
    plan = json.loads(raw)
    if (plan['format'] != 'forest-training-draws-v1' or plan['world_size'] != world_size
            or plan['data_signature'] != dataset.signature
            or len(plan['epochs']) != config['epochs'] or len(plan['samples']) != len(dataset)
            or plan['draws_per_epoch'] % world_size):
        raise ValueError('Training draw plan does not match data/config/world size')
    for item, (episode, sample, _) in zip(plan['samples'], dataset.samples):
        if (item['episode'] != dataset.episodes[episode][0].name
                or item['frame_index'] != sample['frame_index']):
            raise ValueError('Frozen sample order changed')
    for epoch in plan['epochs']:
        if len(epoch['draws']) != plan['draws_per_epoch']:
            raise ValueError('Incomplete epoch draw plan')
        for draw in epoch['draws']:
            if not 0 <= draw['index'] < len(dataset) or type(draw['drop_history']) is not bool:
                raise ValueError('Invalid draw')
    return DrawDataset(dataset), DrawSampler(plan, rank)
