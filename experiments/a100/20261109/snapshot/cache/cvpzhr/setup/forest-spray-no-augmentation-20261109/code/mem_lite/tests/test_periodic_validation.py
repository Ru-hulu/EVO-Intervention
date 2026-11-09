"""CPU tests for validation scheduling, exact distributed metrics and best checkpoints."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mem_lite import train_stage3 as train
from mem_lite import periodic_validation as validation
from mem_lite.tests.test_m3_predictor import model as model


@pytest.mark.parametrize("batches,accumulation,expected", [
    (10, 1, {5, 10}), (9, 1, {5, 9}), (10, 3, {2, 4}), (1, 1, {1}),
])
def test_fractional_epoch_boundaries(batches, accumulation, expected):
    assert validation.validation_update_points(batches, accumulation, 0.5) == expected


@pytest.mark.parametrize("interval", [0, -1, 1.5, 0.3])
def test_invalid_validation_interval(interval):
    with pytest.raises(ValueError):
        validation.validation_update_points(10, 1, interval)


class MetricProbe(torch.nn.Module):
    def forward(self, inputs):
        torch.rand(1)  # Even if inference consumes RNG, the training stream must be restored.
        return {"pt": inputs["prediction"]}


class MetricLoader:
    def __init__(self, rank=0, world_size=1, count=5):
        self.dataset = range(count)
        self.indices = list(range(rank, count, world_size))

    def __iter__(self):
        errors = torch.tensor([0.2, -0.4, 0.1, 0.0, 0.5])
        suites = torch.tensor([0, 0, 1, 2, 2])
        for i in self.indices:
            yield {"inputs": {"prediction": errors[i:i+1]}, "target_pt": torch.zeros(1),
                   "suite_indices": suites[i:i+1]}


def test_frame_weighted_metrics_and_rng_restore():
    model = MetricProbe().train()
    before = torch.get_rng_state().clone()
    result = validation.validate(model, MetricLoader(), torch.device("cpu"))
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert result["num_samples"] == 5 and result["mae_pp"] == pytest.approx(24)
    assert result["by_suite"]["Count"]["mae_pp"] == pytest.approx(30)
    assert result["by_suite"]["Order"]["mae_pp"] == pytest.approx(10)
    assert result["by_suite"]["State"]["mae_pp"] == pytest.approx(25)


def distributed_validation_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        result = validation.validate(MetricProbe(), MetricLoader(rank, 2), torch.device("cpu"))
        assert result["num_samples"] == 5 and result["mae_pp"] == pytest.approx(24)
        # One rank receives zero rows; both ranks still participate in the final reduction.
        result = validation.validate(MetricProbe(), MetricLoader(rank, 2, 1), torch.device("cpu"))
        assert result["num_samples"] == 1 and result["mae_pp"] == pytest.approx(20)
    finally:
        dist.destroy_process_group()


def test_uneven_distributed_validation_has_no_padding(tmp_path):
    mp.spawn(distributed_validation_worker, args=(str(tmp_path / "validation-gloo"),),
             nprocs=2, join=True)


def test_worse_checkpoint_updates_last_and_preserves_best(model, tmp_path):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    config = {"keep_only_latest": True, "best_validation_mae_pp": 10.0, "best_validation_step": 1}
    with patch.object(train.torch.cuda, "get_rng_state", return_value=torch.tensor([1], dtype=torch.uint8)):
        train.save_checkpoint(model, optimizer, config, tmp_path, 0, 2, 1, is_best=True)
        train.save_checkpoint(model, optimizer, config, tmp_path, 1, 0, 2, is_best=False)
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    assert best["step"] == 1 and last["step"] == 2
    assert last["config"]["best_validation_step"] == 1


class LoopDataset:
    history_dim, history_capacity, history_source = 1, 1, {"fixture": True}
    signature = "loop-fixture"

    def __init__(self, paths, history_dim=None):
        self.episode = paths[0]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"value": index / 7, "target_pt": index / 7, "episode_id": self.episode}


class LoopCollator:
    def __init__(self, *args):
        pass

    def __call__(self, rows):
        return {"inputs": {"value": torch.tensor([[r["value"]] for r in rows])},
                "target_pt": torch.tensor([r["target_pt"] for r in rows])}


class LoopModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.qwen = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(out_hidden_size=1)))

    def forward(self, inputs, target_pt=None):
        pt = self.linear(inputs["value"]).sigmoid().flatten()
        result = {"pt": pt}
        if target_pt is not None:
            result.update(loss_sum=(pt - target_pt).square().sum(), num_samples=len(pt))
        return result


def test_training_loop_validates_six_times_and_resume_keeps_best(tmp_path, monkeypatch):
    # Exercise the real trainer control flow with a CPU probe; production GPU smoke is separate.
    for module in (train, validation):
        monkeypatch.setattr(module, "M3Dataset", LoopDataset)
        monkeypatch.setattr(module, "Stage3Collator", LoopCollator)
    monkeypatch.setattr(train, "M3ProgressPredictor", LoopModel)
    monkeypatch.setattr(train.AutoProcessor, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(train, "verify_vision_source", lambda *a: None)
    monkeypatch.setattr(train, "trainable_state", lambda model: {"probe": model.state_dict()})
    monkeypatch.setattr(train, "load_trainable_state", lambda model, state: model.load_state_dict(state["probe"]))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([1], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *a: None)
    config = json.loads(train.DEFAULT_CONFIG.read_text())
    config.update(model_path="/fixture", sample_files=["/train"], validation_sample_files=["/val"],
                  validation_suite_by_episode={"/val": "Count"}, validation_interval_epochs=0.5,
                  batch_size=2, validation_batch_size=2, epochs=3, gradient_accumulation_steps=1,
                  num_workers=0, validation_num_workers=0, output_dir=str(tmp_path),
                  max_steps=6, save_every_steps=None, keep_only_latest=True)
    config["model"]["history_dim"] = 1
    train._train(config, None, 0, 1, torch.device("cpu"))
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert checkpoint["step"] == 6 and checkpoint["next_batch"] == 2
    config = dict(checkpoint["config"], max_steps=None)
    train._train(config, checkpoint, 0, 1, torch.device("cpu"))
    logs = [json.loads(line) for line in (tmp_path / "validation.jsonl").read_text().splitlines()]
    assert [row["epoch_progress"] for row in logs] == [0.5, 1, 1.5, 2, 2.5, 3]
    assert all(row["num_samples"] == 8 for row in logs)
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert last["step"] == 12 and last["epoch"] == 3 and last["next_batch"] == 0
    assert best["step"] == last["config"]["best_validation_step"]
    assert last["config"]["best_validation_mae_pp"] == min(row["mae_pp"] for row in logs)
