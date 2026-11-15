"""Reuse the verified two-rank trainer and the exact native GRM cache collator."""
from .common import ROOT, verify_code
from experiments.soc35_finetune_20261114 import train as native
from experiments.recover_finetune_20261114.data import Dataset as GRMDataset, CachedCollator as GRMCollator


class Dataset(GRMDataset):
    def __init__(self, config):
        super().__init__(config['train_data'])


class CachedCollator(GRMCollator):
    def __init__(self, processor, kind, max_length):
        assert kind in ('robobrain25-4b', 'dopamine-3b')
        super().__init__(processor.tokenizer, kind, max_length)


def main():
    verify_code()
    # These assignments only affect this fresh worker process, never another job.
    native.ROOT = ROOT
    native.Dataset = Dataset
    native.CachedCollator = CachedCollator
    load = native.load_model

    def load_model(config):
        model, processor, audit = load(config)
        audit.update(suite=config['suite'], future_endpoint_reference=True,
                     supervision=config['supervision'], cache_audit_sha256=config['cache_audit_sha256'])
        return model, processor, audit

    native.load_model = load_model
    native.main()


if __name__ == '__main__':
    main()
