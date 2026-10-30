"""训练进度记录；只在最后一个 epoch 保存一次完整 LoRA 检查点。"""

import json
import math
from pathlib import Path
import shutil
import time

import torch
from transformers import TrainerCallback


class RunMonitor(TrainerCallback):
    def __init__(self):
        self.latest = {}
        self.started = time.time()

    def write_status(self, args, state, phase="training", **extra):
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        status = {"phase": phase, "step": state.global_step, "max_steps": state.max_steps,
                  "epoch": state.epoch or 0, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "elapsed_seconds": time.time() - self.started, "latest": self.latest,
                  "free_gib": shutil.disk_usage(output).free / 1024**3,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                  **extra}
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(output / "status.json")

    def on_train_begin(self, args, state, control, **kwargs):
        self.write_status(args, state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if any(not math.isfinite(logs[key]) for key in ("loss", "grad_norm", "eval_loss") if key in logs):
            raise RuntimeError("Training produced a non-finite loss or gradient")
        self.latest.update(logs)
        with (Path(args.output_dir) / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n")
        self.write_status(args, state)
        if shutil.disk_usage(args.output_dir).free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB free; stopping this run")

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step >= state.max_steps
        return control

    def on_save(self, args, state, control, **kwargs):
        self.write_status(args, state, checkpoint=str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))

    def on_train_end(self, args, state, control, **kwargs):
        self.write_status(args, state, phase="training_completed",
                          checkpoint=str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))
