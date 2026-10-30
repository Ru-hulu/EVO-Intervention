"""Compare real data-loader/forward/backward throughput without optimizer updates."""
import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import os
import subprocess
import time

import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from .data import ProgressCollator, ProgressDataset, answer_text
from .model import load_model, to_device
from .train import make_trainer


def check_exclusive_gpu():
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(int(pid.strip()) != os.getpid() for pid in pids.splitlines() if pid.strip()):
        raise RuntimeError('Other GPU task appeared; discard this benchmark and wait for an idle GPU')


def benchmark(config, output):
    set_seed(config['seed'])
    torch.set_num_threads(8)
    model, processor = load_model(config, trainable=True, device='cuda:0')
    trainer = make_trainer(config, model, processor, output_dir=str(output.parent / 'benchmark-trainer'))
    # Match Trainer's BF16 wrapper, including its output conversion; never call Trainer.train().
    model = trainer.accelerator.prepare_model(model)
    model.train()
    collator = ProgressCollator(processor, config['max_length'])
    rows = ProgressDataset(config['train_data'], supervised=True).rows
    random.Random(42).shuffle(rows)
    longest = sorted(rows, key=lambda row: len(processor.tokenizer.encode(row['task'] +
                     answer_text(row['remaining_actions'], row['pt']))), reverse=True)
    versions = {name: p._version for name, p in model.named_parameters()}
    report = {'optimizer_steps': 0, 'training_started': False, 'cases': [],
              'method': 'Trainer BF16 wrapper, two warmup plus eight measured updates; includes CPU data wait, H2D, forward/backward and gradient clearing; excludes optimizer update'}
    for batch_size, workers, accumulation in ((8, 0, 1), (4, 4, 2), (8, 4, 1), (8, 8, 1), (16, 4, 1), (20, 4, 1), (24, 4, 1)):
        check_exclusive_gpu()
        case = {'batch_size': batch_size, 'workers': workers, 'gradient_accumulation_steps': accumulation}
        loader = iterator = inputs = result = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Check a long-answer batch first; keep at least 8 GiB for runtime variation/state.
            inputs = to_device(collator(longest[:batch_size]), model)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(**inputs)
            result.loss.backward()
            assert torch.isfinite(result.loss)
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            case['long_batch_peak_gib'] = torch.cuda.max_memory_allocated() / 1024**3
            model.zero_grad(set_to_none=True)
            inputs = result = None
            if case['long_batch_peak_gib'] > torch.cuda.get_device_properties(0).total_memory / 1024**3 - 8:
                case['status'] = 'insufficient_memory_margin'
                continue
            loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collator,
                                num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                                **({'prefetch_factor': 2} if workers else {}))
            iterator = iter(loader)
            elapsed, waits, computes = [], [], []
            for step in range(10):
                check_exclusive_gpu()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                data_wait = 0.0
                for _ in range(accumulation):
                    waiting = time.perf_counter()
                    batch = next(iterator)
                    data_wait += time.perf_counter() - waiting
                    inputs = to_device(batch, model)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(**inputs)
                    (result.loss / accumulation).backward()
                    inputs = result = batch = None
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if step >= 2:
                    elapsed.append(end - begin)
                    waits.append(data_wait)
                    computes.append(end - begin - data_wait)
            check_exclusive_gpu()
            case.update(status='passed', samples_per_second=batch_size * accumulation / statistics.mean(elapsed),
                        mean_step_seconds=statistics.mean(elapsed), data_wait_seconds=statistics.mean(waits),
                        compute_and_transfer_seconds=statistics.mean(computes),
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                        measured_batches=len(elapsed))
        except torch.OutOfMemoryError:
            case['status'] = 'oom'
        finally:
            if iterator is not None and workers:
                iterator._shutdown_workers()
            del iterator, loader, inputs, result
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            report['cases'].append(case)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(case), flush=True)
        if case['status'] == 'oom':
            break
    assert all(p._version == versions[name] for name, p in model.named_parameters())
    report['parameters_unchanged'] = True
    report['status'] = 'completed'
    output.write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark(json.loads(args.config.read_text()), args.output)
