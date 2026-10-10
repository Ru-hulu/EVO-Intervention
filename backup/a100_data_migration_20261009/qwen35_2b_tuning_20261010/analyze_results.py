"""Summarize completed bounded benchmarks without launching any training."""
import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def throughput(rows):
    return sum(x['samples'] for x in rows) / sum(x['samples'] / x['samples_per_second'] for x in rows) if rows else None

cases = []
for summary_path in sorted(BASE.glob('batch*/summary.json')):
    root = summary_path.parent
    summary = json.loads(summary_path.read_text())
    launch = json.loads((root / 'launch.json').read_text())
    args = launch['parameters']
    n = args['train_batches']
    metrics = read_jsonl(root / 'output/metrics.jsonl')
    train = [x for x in metrics if x['kind'] == 'train']
    steady = [x for x in train if (x['step'] - 1) % n >= 4]
    epoch_rows = defaultdict(list)
    for row in steady:
        epoch_rows[row['epoch']].append(row)
    input_rows = [row for file in (root / 'output').glob('input-timing-rank-*.jsonl') for row in read_jsonl(file)
                  if row['phase'] == 'train' and row['phase_batch'] > 4]
    input_steps = defaultdict(list)
    for row in input_rows:
        input_steps[row.get('epoch', 1), row['phase_batch']].append(row['input_wait_seconds'])
    ranks = summary['rank_results']
    passed_updates = len(ranks) == 8 and all(
        all(math.isfinite(v) and v > 0 for v in row['parameter_max_abs_change'].values()) for row in ranks)
    all_train = [x for x in metrics if x['kind'] == 'train_epoch']
    validations = summary['validation']
    phases = summary['phases']
    transitions_passed = (
        len(phases) == args['benchmark_epochs'] * 3
        and all(x['active_data_workers'] == 0 for x in phases)
        and [(x['epoch'], x['phase']) for x in phases] == [
            (epoch, phase) for epoch in range(1, args['benchmark_epochs'] + 1)
            for phase in ('before_train', 'before_validation', 'after_validation')])
    checkpoints = [str(x.relative_to(root)) for x in (root / 'output').glob('epoch-*.pt')]
    # A local small-artifact copy excludes checkpoints; existence evidence is in the remote audit.
    case = {
        'name': root.name,
        'exit_code': summary['exit_code'],
        'per_gpu_batch': args['per_gpu_batch'],
        'global_batch': args['per_gpu_batch'] * 8,
        'workers_per_gpu': args.get('num_workers', 4),
        'validation_batch_per_gpu': args['val_batch_size'],
        'short_epochs': args['benchmark_epochs'],
        'train_batches_per_epoch': n,
        'validation_batches_per_epoch': args['val_batches'],
        'train_samples_tested': sum(x['samples'] for x in all_train),
        'validation_samples_tested': sum(x['samples'] for x in validations),
        'steady_samples_per_second': throughput(steady),
        'steady_samples_per_second_by_epoch': {str(epoch): throughput(rows) for epoch, rows in epoch_rows.items()},
        'late_half_samples_per_second': throughput([x for x in train if (x['step'] - 1) % n >= n // 2]),
        'cuda_peak_allocated_gib': summary['peak_allocated_mib'] / 1024 if summary['peak_allocated_mib'] else None,
        'cuda_peak_reserved_gib': summary['peak_reserved_mib'] / 1024 if summary['peak_reserved_mib'] else None,
        'sampled_peak_device_used_gib': summary['sampled_peak_device_used_mib'] / 1024,
        'minimum_host_available_gib': summary['minimum_host_available_gib'],
        'all_eight_ranks_updated_all_trainable_components': passed_updates,
        'phase_cleanup_checks_passed': transitions_passed,
        'checkpoint_paths': checkpoints,
        'mean_rank_input_wait_seconds': sum(x['input_wait_seconds'] for x in input_rows) / len(input_rows) if input_rows else None,
        'mean_step_slowest_rank_input_wait_seconds': sum(max(x) for x in input_steps.values()) / len(input_steps) if input_steps else None,
        'mean_consumer_wall_seconds': sum(x['consumer_seconds'] for x in input_rows) / len(input_rows) if input_rows else None,
        'validation_metrics': validations,
    }
    case['passed'] = summary['exit_code'] == 0 and passed_updates and transitions_passed and len(validations) == args['benchmark_epochs']
    cases.append(case)
report = {
    'model': '/data/evo-inter/models/Qwen3.5-2B',
    'hardware': '8 x NVIDIA A100-PCIE-40GB',
    'formal_training_started': False,
    'measurement': {
        'throughput': 'Global samples / summed step time, excludes first four steps of EACH short epoch, validation and phase startup/shutdown.',
        'memory': 'CUDA allocated and reserved are exact allocator maxima across all ranks and train/validation. NVML snapshots are every five seconds and can miss brief peaks.',
        'input_wait': 'Time blocked getting a batch. Consumer wall time includes DDP waits for other ranks; these quantities are not independent and must not be added.',
        'limits': 'Real-data short benchmarks only; throughput depends on video mix and does not prove full-epoch stability or model quality. Different batch sizes consume different sample counts.',
    },
    'cases': cases,
}
(BASE / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for case in cases:
    print(json.dumps({k: case[k] for k in ['name', 'passed', 'steady_samples_per_second', 'steady_samples_per_second_by_epoch', 'cuda_peak_allocated_gib', 'cuda_peak_reserved_gib', 'mean_step_slowest_rank_input_wait_seconds']}, ensure_ascii=False))
