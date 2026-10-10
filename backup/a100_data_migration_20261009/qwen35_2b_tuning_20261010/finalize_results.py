"""Save a verified recommendation and launch configuration; never start training."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--selected-case', required=True)
args = p.parse_args()
base = Path(__file__).resolve().parent
subprocess.run(['python3', str(base / 'analyze_results.py')], check=True)
comparison = json.loads((base / 'comparison.json').read_text())
cases = {x['name']: x for x in comparison['cases']}
required = ['batch32-baseline', 'batch48', 'batch64-input-timing', 'batch64-workers6-confirm', 'batch48-workers6-comparison']
assert all(name in cases and cases[name]['passed'] for name in required)
assert args.selected_case in required
checkpoint = json.loads((base / 'checkpoint_audit.json').read_text())
assert checkpoint['passed']
active = json.loads((base / 'active_case.json').read_text())
assert active.get('exit_code') == 0 and active.get('completed_unix')
busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
assert not busy, 'A GPU compute process is still active; do not finalize yet.'
selected = cases[args.selected_case]
best = max(x['steady_samples_per_second'] for x in cases.values())
assert selected['steady_samples_per_second'] >= best * 0.97
batch = selected['global_batch']
workers = selected['workers_per_gpu']
launcher = Path('/data/evo-inter/code/qwen_training/run_a100_8gpu_2b.sh')
script = f'''#!/usr/bin/env bash
# Qwen3.5-2B configuration selected by bounded 8 x A100 tests on 2026-10-10.
# This file was saved and syntax-checked only; formal training has NOT been started.
# Usage: bash run_a100_8gpu_2b.sh [output_directory] [additional_training_arguments...]
# Start from the 2B base model. A 4B checkpoint is not compatible with this backbone.
set -euo pipefail
EVO_2B_SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
EVO_2B_RUN_DIR=${{1:-/data/evo-inter/outputs/qwen35-2b-8gpu-$(date +%Y%m%d-%H%M%S)}}
if (( $# )); then shift; fi
exec bash "$EVO_2B_SCRIPT_DIR/run_a100_8gpu.sh" "$EVO_2B_RUN_DIR" \\
  --model-path /data/evo-inter/models/Qwen3.5-2B \\
  --global-batch-size {batch} --val-batch-size 16 \\
  --num-workers {workers} --prefetch-factor 2 \\
  "$@"
'''
if launcher.exists() and launcher.read_text() != script:
    raise RuntimeError('A different 2B launcher already exists; inspect before replacing it.')
launcher.write_text(script)
launcher.chmod(0o755)
subprocess.run(['bash', '-n', str(launcher)], check=True)
config = {
    'model_path': '/data/evo-inter/models/Qwen3.5-2B',
    'world_size': 8,
    'per_gpu_batch': selected['per_gpu_batch'],
    'global_batch_size': batch,
    'gradient_accumulation_steps': 1,
    'val_batch_size_per_gpu': 16,
    'num_workers_per_gpu': workers,
    'prefetch_factor': 2,
    'persistent_workers': False,
    'gpu_image_processing': True,
    'require_fast_kernels': True,
    'gradient_checkpointing': False,
    'max_pixels': 262144,
    'epochs_when_formal_training_is_requested': 12,
    'save_epochs_when_formal_training_is_requested': [2, 4, 6, 8, 10, 12],
    'formal_training_started': False,
    'launcher': str(launcher),
    'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
    'selected_case': args.selected_case,
    'selection_policy': 'Prefer sustained throughput; within 3% of best measured throughput, favor lower activation memory. Best among tested settings only.',
    'steady_samples_per_second': selected['steady_samples_per_second'],
    'cuda_peak_allocated_gib': selected['cuda_peak_allocated_gib'],
    'cuda_peak_reserved_gib': selected['cuda_peak_reserved_gib'],
}
(base / 'recommended_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
report = {
    'status': 'complete',
    'completed_unix': time.time(),
    'download_verified': json.loads((base.parent / 'qwen35_2b_20261010/download_result.json').read_text())['status'] == 'complete',
    'hardware': comparison['hardware'],
    'recommendation': config,
    'cases': comparison['cases'],
    'measurement': comparison['measurement'],
    'checkpoint_audit': checkpoint,
    'tokenization_parity': json.loads((base / 'tokenization_parity.json').read_text()),
    'provenance': json.loads((base / 'reproducibility.json').read_text()),
    'training_implementation_modified_during_tuning': False,
    'formal_training_started': False,
    'gpu_compute_processes_at_completion': [],
    'all_benchmark_processes_exited': True,
    'two_short_train_validation_cycles_passed': cases['batch64-workers6-confirm']['passed'],
    'total_train_samples_tested': sum(x['train_samples_tested'] for x in cases.values()),
    'total_validation_samples_tested': sum(x['validation_samples_tested'] for x in cases.values()),
}
(base / 'final_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(config, ensure_ascii=False))
