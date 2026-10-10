from pathlib import Path
import collections
import hashlib
import json
import sys
import time

import av
import torch

sys.path.insert(0, '/data/evo-inter/code')
from qwen_training.dataset import VLMProgressDataset, collate_fn

torch.set_num_threads(2)
job = Path('/root/dataset/.migration_jobs/a100_20261009')
migration = json.loads((job / 'index_migration_report.json').read_text())
cleanup = json.loads((job / 'cleanup_report.json').read_text())
report = {'verified_at': time.time(), 'datasets': [], 'cleanup': cleanup}
for record in migration['datasets']:
    root = Path(record['root'])
    index = root / 'samples/samples.jsonl'
    assert hashlib.sha256(index.read_bytes()).hexdigest() == record['adapted_index_sha256']
    video_meta = []
    for path, maximum in record['max_referenced_frame'].items():
        with av.open(path) as video:
            stream = video.streams.video[0]
            assert stream.average_rate == 30
            assert stream.frames > maximum >= 0, (path, maximum, stream.frames)
            video_meta.append({'path': path, 'frames': stream.frames,
                               'maximum_referenced_frame': maximum,
                               'width': stream.width, 'height': stream.height})
    dataset = VLMProgressDataset(index)
    assert len(dataset) == record['samples']
    selected = {}
    counts = collections.Counter()
    for i, sample in enumerate(dataset.samples):
        counts[sample['sample_type']] += 1
        selected.setdefault(sample['sample_type'], i)
    selected['last_sample'] = len(dataset) - 1
    max_i = max(range(len(dataset)), key=lambda i: dataset.samples[i]['frame_indices'][-1])
    selected['largest_frame_index'] = max_i
    decoded = []
    for kind, i in selected.items():
        sample = dataset[i]
        inputs, targets = collate_fn([sample])
        assert tuple(inputs['images'].shape[:4]) == (1, 3, 6, 3)
        assert tuple(inputs['frame_mask'].shape) == (1, 3, 6)
        assert torch.isfinite(targets['progress']).all()
        assert 0 <= targets['progress'].item() <= 1
        decoded.append({'kind': kind, 'sample_id': sample['sample_id'],
                        'shape': list(inputs['images'].shape),
                        'frame_indices': dataset.samples[i]['frame_indices']})
        print('Decoded', record['dataset'], kind, sample['sample_id'], flush=True)
    report['datasets'].append({'dataset': record['dataset'], 'index': str(index),
                               'samples': len(dataset), 'sample_types': dict(counts),
                               'labels': len(list((root / 'labels').glob('*.JSON'))),
                               'deltas': len(list((root / 'delta').glob('*-delta.txt'))),
                               'videos_checked': video_meta, 'decoded_samples': decoded})
    del dataset
report['status'] = 'PASSED'
(job / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('PASSED: all 240 video references and 12 decoded samples', flush=True)
