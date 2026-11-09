"""Verify the rebuilt caches with the real 4B model, without parameter updates."""
import json
from pathlib import Path
import time

import torch
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.memory.vision import verify_vision_source
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.train_stage3 import configure_history


torch.set_num_threads(2)
torch.manual_seed(42)
root = Path('/root/autodl-tmp/streamforest-cache/rebuild-20261030')
assert json.loads((root / 'verification.json').read_text())['status'] == 'PASS'
config = json.loads((root / 'train.config.json').read_text())
dataset = M3Dataset(config['sample_files'])
configure_history(config, dataset)
verify_vision_source(config['model_path'], dataset.history_source)
processor = AutoProcessor.from_pretrained(config['model_path'], max_pixels=config['max_pixels'], local_files_only=True)
collator = Stage3Collator(processor, dataset.history_dim, config['max_sequence_length'], dataset.history_capacity)
model = M3ProgressPredictor(config['model_path'], **config['model']).to('cuda').eval()
checks = []
offset, seen = 0, set()
for directory, summary in dataset.episodes:
    family = summary['source']['dataset']
    if family not in seen:
        for local in [0, summary['num_samples'] - 1]:
            batch = collator([dataset[offset + local]])
            inputs = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch['inputs'].items()}
            target = batch['target_pt'].to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                result = model(inputs, target)
            torch.cuda.synchronize()
            assert torch.isfinite(result['loss']) and torch.isfinite(result['pt']).all()
            checks.append({'dataset': family, 'episode': summary['episode_id'],
                           'sample': local, 'history_tokens': int(inputs['history_valid_mask'].sum()),
                           'tokens': inputs['input_ids'].shape[1],
                           'forward_seconds': time.perf_counter() - start, 'loss_finite': True})
        seen.add(family)
    offset += summary['num_samples']
    if len(seen) == 2:
        break
assert seen == {'memlong', 'insert-screw-recover'}
report = {'status': 'PASS', 'model': config['model_path'], 'train_samples': len(dataset),
          'history_dim': dataset.history_dim, 'history_capacity': dataset.history_capacity,
          'checks': checks, 'parameter_updates': 0, 'training_started': False}
(root / 'training-readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
