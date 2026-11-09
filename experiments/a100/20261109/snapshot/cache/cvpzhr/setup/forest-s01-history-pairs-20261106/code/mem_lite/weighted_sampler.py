"""A reproducible global weighted draw, sharded across DDP ranks."""
import hashlib
import math
from pathlib import Path
import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, settings, rank, world_size, seed):
        path = Path(settings['weights_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings['weights_sha256']:
            raise ValueError('Sampling weights changed')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        self.weights = payload['weights'].double()
        if payload['data_signature'] != dataset.signature or len(self.weights) != len(dataset):
            raise ValueError('Sampling weights do not match the training data')
        if not torch.isfinite(self.weights).all() or not (self.weights > 0).all():
            raise ValueError('Every training observation must retain positive probability')
        self.size, self.rank, self.world_size, self.seed = len(dataset), rank, world_size, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.size / self.world_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.size, replacement=True, generator=generator).tolist()
        padding = len(self) * self.world_size - self.size
        indices.extend(indices[:padding])
        return iter(indices[self.rank::self.world_size])
