"""Short two-GPU throughput trial, using the exact training path without saving."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import TrainerCallback, set_seed
from robometer_training.model import load_processor, load_model, attach_lora, load_adapter
from robometer_training.train import make_trainer, RunMonitor


class Timing(TrainerCallback):
    def __init__(self, output):
        self.output = output
        self.previous = None
        self.times = []

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.previous is not None:
            self.times.append(now - self.previous)
        self.previous = now

    def on_log(self, args, state, control, logs=None, **kwargs):
        for name in ('loss', 'grad_norm'):
            if name in (logs or {}):
                assert torch.isfinite(torch.tensor(logs[name])), (name, logs[name])

    def on_train_end(self, args, state, control, **kwargs):
        values = self.times[3:]
        result = {
            'rank': args.process_index, 'steps': state.global_step,
            'per_device_batch': args.per_device_train_batch_size,
            'steady_step_seconds': statistics.mean(values),
            'steady_median_seconds': statistics.median(values),
            'global_samples_per_second': args.per_device_train_batch_size * args.world_size / statistics.mean(values),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'step_seconds': self.times,
        }
        (self.output / f'timing-rank{args.process_index}.json').write_text(json.dumps(result, indent=2))
        print('BENCHMARK_RESULT', json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--batch', required=True, type=int)
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    torch.multiprocessing.set_start_method('spawn', force=True)
    c = json.loads(Path(args.config).read_text())
    output = Path(c['setup_dir']) / 'benchmarks' / f'batch{args.batch}'
    output.mkdir(parents=True, exist_ok=True)
    c['output_dir'] = str(output)
    c['eval_strategy'] = 'no'
    c['training'].update(per_device_train_batch_size=args.batch, max_steps=16, logging_steps=4)
    set_seed(c['seed'])
    processor = load_processor(c['model_path'], min_pixels=c['min_pixels'], max_pixels=c['max_pixels'])
    model, _ = load_model(c['model_path'], processor)
    attach_lora(model, **c['lora'])
    load_adapter(model, c['initial_checkpoint'], trainable=True)
    trainer = make_trainer(c, model, processor)
    trainer.remove_callback(RunMonitor)
    trainer.args.save_strategy = 'no'
    trainer.add_callback(Timing(output))
    torch.cuda.reset_peak_memory_stats()
    trainer.train()


if __name__ == '__main__':
    main()
