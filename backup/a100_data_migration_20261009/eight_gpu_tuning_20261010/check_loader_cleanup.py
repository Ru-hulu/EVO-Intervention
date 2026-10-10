import gc
import multiprocessing
import torch
from torch.utils.data import DataLoader
from qwen_training.benchmark_distributed import BoundedSampler
from qwen_training.dataloader import build_dataloaders
from qwen_training.dataset import collate_fn
from qwen_training.train_distributed import GlobalBatchSampler

if __name__ == '__main__':
    torch.cuda.set_device(0)
    torch.zeros(1, device='cuda')
    train, val = build_dataloaders([
        '/root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl',
        '/root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl'], num_workers=0)
    options=dict(num_workers=4, prefetch_factor=2, persistent_workers=True,
                 multiprocessing_context='spawn', pin_memory=True, collate_fn=collate_fn)
    for mode in ('train', 'validation'):
        if mode == 'train':
            source=GlobalBatchSampler(train.sampler, 32, 0, 8, seed=0)
            loader=DataLoader(train.dataset, batch_sampler=BoundedSampler(source, 2), **options)
        else:
            loader=DataLoader(val.dataset, batch_size=16,
                              sampler=BoundedSampler(range(0, len(val.dataset), 8), 32), **options)
        count=0
        for inputs, targets in loader:
            count += len(targets['progress'])
        assert count == (8 if mode == 'train' else 32), (mode, count)
        del loader, inputs, targets
        gc.collect()
        assert not multiprocessing.active_children(), multiprocessing.active_children()
        print(mode, 'BOUNDED_LOADER_CLEAN_EXIT', count, flush=True)
