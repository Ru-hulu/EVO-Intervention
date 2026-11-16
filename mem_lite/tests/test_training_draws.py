"""History dropout must remove real sequence slots and work across four ranks."""
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from mem_lite.datasets.training_draws import DrawDataset
from mem_lite.tests.test_m3_predictor import collator, sample
from mem_lite.tests.test_stage3_training import OptionalHistoryProbe


def test_draw_dropout_matches_native_no_history_tokens(collator):
    class Dataset:
        signature, history_dim, history_capacity, history_source = 'test', 16, 128, {}
        def __len__(self): return 1
        def __getitem__(self, index): return dict(sample(torch.ones(128,16)), target_pt=.4)
    wrapped = DrawDataset(Dataset())
    dropped = collator([wrapped[(0,True,True)]])
    native = collator([sample()])
    assert dropped.keys() == native.keys()
    for k in native: torch.testing.assert_close(dropped[k],native[k])
    kept = collator([wrapped[(0,False,True)]])
    assert kept['history_valid_mask'].sum().item() == 128
    assert kept['history_mask'].sum().item() == 128


def four_rank_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+path, rank=rank, world_size=4)
    try:
        model = OptionalHistoryProbe()
        ddp = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
        for pattern in ((True,False,True,False),(False,False,False,False),(False,True,False,True)):
            ddp.zero_grad(set_to_none=True)
            ddp(pattern[rank])['loss_sum'].backward()
            assert model.main.grad.item() == 2
            actual = 0 if model.history.grad is None else model.history.grad.item()
            assert actual == sum(pattern)*3/4
    finally: dist.destroy_process_group()


def test_four_rank_mixed_and_all_missing_history(tmp_path):
    mp.spawn(four_rank_worker,args=(str(tmp_path/'gloo'),),nprocs=4,join=True)
