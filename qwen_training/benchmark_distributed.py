"""用正式 DDP 入口做有限批次性能测试；输出只用于调参，不作为正式训练结果。"""

import argparse
import itertools
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Sampler

from qwen_training import train_distributed as training


class BoundedSampler(Sampler):
    """在向 worker 派发任务前限长，避免预取未使用视频并强行中止解码进程。"""

    def __init__(self, source, limit):
        self.source, self.limit = source, limit

    def __iter__(self):
        return itertools.islice(iter(self.source), self.limit)

    def __len__(self):
        return min(self.limit, len(self.source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--benchmark-epochs", type=int, default=1,
                        help="多轮短程测试用于覆盖训练、验证和下一轮训练的进程切换")
    parser.add_argument("--output-dir", required=True)
    args, remaining = parser.parse_known_args()
    if min(args.train_batches, args.val_batches, args.benchmark_epochs) < 1 or "--resume" in remaining:
        parser.error("性能测试需要正数批次数，不能恢复正式检查点")
    captured = {}

    class LimitedLoader(DataLoader):
        def __init__(self, *loader_args, **loader_kwargs):
            if isinstance(loader_kwargs.get("batch_sampler"), training.GlobalBatchSampler):
                loader_kwargs["batch_sampler"] = BoundedSampler(
                    loader_kwargs["batch_sampler"], args.train_batches,
                )
            elif "sampler" in loader_kwargs:
                loader_kwargs["sampler"] = BoundedSampler(
                    loader_kwargs["sampler"], args.val_batches * loader_kwargs["batch_size"],
                )
            super().__init__(*loader_args, **loader_kwargs)

    class MeasuredModel(training.ReferenceProgressModel):
        def __init__(self, *model_args, **model_kwargs):
            super().__init__(*model_args, **model_kwargs)
            captured["model"] = self
            parameters = dict(self.named_parameters())
            names = {
                "lora": next(n for n in parameters if "lora_B" in n),
            }
            captured["before"] = {
                group: (name, parameters[name].detach().flatten()[:4096].float().cpu().clone())
                for group, name in names.items()
            }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if int(os.environ["RANK"]) == 0:
        (output / "BENCHMARK_ONLY.json").write_text(json.dumps(vars(args), indent=2))
    training.DataLoader = LimitedLoader
    training.ReferenceProgressModel = MeasuredModel
    sys.argv = [sys.argv[0], *remaining, "--output-dir", args.output_dir,
                "--epochs", str(args.benchmark_epochs), "--save-epochs", str(args.benchmark_epochs),
                "--log-every", "1"]
    training.main()
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.synchronize(device)
    parameters = dict(captured["model"].named_parameters())
    deltas = {
        group: (parameters[name].detach().flatten()[:4096].float().cpu() - before).abs().max().item()
        for group, (name, before) in captured["before"].items()
    }
    result = {
        "rank": int(os.environ["RANK"]),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "parameter_max_abs_change": deltas,
    }
    (output / f"rank-{result['rank']}.json").write_text(json.dumps(result, indent=2))
    print("BENCHMARK_GPU " + json.dumps(result), flush=True)
    if not all(0 < value < float("inf") for value in deltas.values()):
        raise RuntimeError("LoRA 必须产生有限的参数更新")


if __name__ == "__main__":
    main()
