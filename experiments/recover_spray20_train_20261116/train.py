"""Reuse the audited DDP trainer with the original Recovery cache and losses."""
from .common import ROOT
from experiments.recover_finetune_20261114.data import Dataset as RecoveryDataset, CachedCollator as RecoveryCollator


class Dataset(RecoveryDataset):
    def __init__(self, config): super().__init__(config['train_data'])


class CachedCollator(RecoveryCollator):
    def __init__(self, processor, kind, max_length):
        super().__init__(processor.tokenizer, kind, max_length)


if __name__ == '__main__':
    from experiments.soc35_finetune_20261114 import train as native
    from experiments.recover_finetune_20261114.train import robometer_api
    native.ROOT, native.Dataset, native.CachedCollator = ROOT, Dataset, CachedCollator
    native.robometer_api = robometer_api
    native.main()
