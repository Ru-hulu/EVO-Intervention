"""第三阶段离线检查：缓存/冻结范围、损失归一化及断点读写，不执行训练循环。"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext

from mem_lite.tests.test_m3_predictor import collator as collator, model as model, sample
from mem_lite.train_stage3 import (
    DEFAULT_CONFIG,
    load_trainable_state,
    normalize_gradients,
    read_config,
    save_checkpoint,
    trainable_state,
)


def test_no_history_batch_keeps_projection_unused_and_other_modules_trainable(model, collator):
    model.train()
    result = model(collator([sample(), sample()]), torch.tensor([0.0, 0.9]))
    result["loss_sum"].backward()
    assert all(p.grad is None for p in model.history_projection.parameters())
    assert model.progress_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_sample_normalization_handles_partial_microbatches_and_rank_average():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    # 两个rank各自含不等大小微批，DDP先按rank取平均；最终按实际样本数归一化。
    parameter.grad = torch.tensor((7.0 + 11.0) / 2)
    unused = torch.nn.Parameter(torch.tensor(1.0))
    normalize_gradients([parameter, unused], torch.tensor(5.0), world_size=2)
    assert parameter.grad.item() == pytest.approx(18 / 5)
    assert unused.grad is None


def test_stage3_checkpoint_roundtrip_saves_only_own_trainable_weights(model, tmp_path):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    optimizer.state[parameters[0]] = {"step": torch.tensor(7.0), "exp_avg": torch.full_like(parameters[0], 0.125),
                                      "exp_avg_sq": torch.full_like(parameters[0], 0.25)}
    before = {name: {key: value.clone() for key, value in part.items()}
              for name, part in trainable_state(model).items()}
    rng = torch.get_rng_state()
    with patch("mem_lite.train_stage3.torch.cuda.get_rng_state", return_value=torch.tensor([1], dtype=torch.uint8)):
        save_checkpoint(model, optimizer, {"data_signature": "fixture", "seed": 42},
                        tmp_path, epoch=1, next_batch=3, step=7)
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    assert saved["format"] == "mem-lite-streamforest-v1"
    assert (saved["epoch"], saved["next_batch"], saved["step"]) == (1, 3, 7)
    torch.testing.assert_close(saved["rng_by_rank"][0]["torch"], rng)
    assert set(saved["model"]) == {"m3_lora", "history_projection", "progress_head"}
    assert all("lora_" in key for key in saved["model"]["m3_lora"])
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(1)
    load_trainable_state(model, saved["model"])
    optimizer.state.clear()
    optimizer.load_state_dict(saved["optimizer"])
    for name, part in trainable_state(model).items():
        for key, value in part.items():
            torch.testing.assert_close(value, before[name][key])
    restored = optimizer.state[parameters[0]]
    assert restored["step"] == 7 and restored["exp_avg"].eq(0.125).all()
    assert restored["exp_avg_sq"].eq(0.25).all()


def test_stage3_resume_preserves_model_and_optimization_config(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(model_path="/model", sample_files=["/samples"], data_signature="old-data")
    args = SimpleNamespace(model_path=None, sample_files=[str(tmp_path / "migrated")], output_dir=str(tmp_path),
                           epochs=4, max_steps=100, config="/must-not-read.json")
    updated = read_config(args, {"config": config})
    assert updated["epochs"] == 4 and updated["max_steps"] == 100
    assert updated["model"] == config["model"]
    assert updated["gradient_accumulation_steps"] == config["gradient_accumulation_steps"]
    assert updated["data_signature"] == "old-data"


class OptionalHistoryProbe(torch.nn.Module):
    """只验证DDP控制流的小模型，不替代M3的真实离线前向检查。"""

    def __init__(self):
        super().__init__()
        self.main = torch.nn.Parameter(torch.tensor(1.0))
        self.history = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, has_history):
        loss = self.main * 2
        if has_history:
            loss = loss + self.history * 3
        return {"loss_sum": loss}


def ddp_history_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = OptionalHistoryProbe()
        wrapped = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
        # 依次覆盖：历史只出现在no_sync微批、只在末微批、跨rank不一致、全批无历史。
        patterns = (((True, False), (False, False)), ((False, True), (True, False)),
                    ((False, False), (False, False)))
        for pattern in patterns:
            wrapped.zero_grad(set_to_none=True)
            for microbatch, has_history in enumerate(pattern[rank]):
                with wrapped.no_sync() if microbatch == 0 else nullcontext():
                    wrapped(has_history)["loss_sum"].backward()
            normalize_gradients(list(model.parameters()), torch.tensor(4.0), world_size=2)
            assert model.main.grad.item() == pytest.approx(2.0)
            count = sum(sum(row) for row in pattern)
            actual = model.history.grad.item() if model.history.grad is not None else 0.0
            assert actual == pytest.approx(count * 3 / 4)
    finally:
        dist.destroy_process_group()


def test_ddp_optional_history_and_accumulation(tmp_path):
    # 两个CPU进程，仅forward/backward及all-reduce，不调用优化器或任何训练入口。
    mp.spawn(ddp_history_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)
