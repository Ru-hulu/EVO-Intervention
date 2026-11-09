"""Reuse existing 5 FPS images/PT and build independent causal window indexes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from .data import ObservationWindow


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output, window_size=8):
    source, output = Path(source).resolve(), Path(output).resolve()
    parent = json.loads((source / 'manifest.json').read_text())
    if parent['status'] != 'complete' or parent['sample_fps'] != 5:
        raise ValueError('Expected the completed 5 FPS source dataset')
    metadata = {row['episode_id']: row for row in parent['episodes']}
    grouped = {}
    for split in ('train', 'validation'):
        path = source / f'{split}.jsonl'
        if sha256(path) != parent['splits'][split]['sha256']:
            raise ValueError(f'Source dataset changed: {split}')
        grouped[split] = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            grouped[split][row['episode_id']].append(row)
    if grouped['train'].keys() & grouped['validation'].keys():
        raise ValueError('Train/validation episodes overlap')
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'robometer-causal-progress-v1', 'status': 'building',
              'sample_fps': 5, 'window_size': window_size, 'view': 'cam_high',
              'padding': 'repeat_first_frame_on_left', 'supervision': 'current_frame_pt_c51',
              'source': str(source), 'source_manifest_sha256': sha256(source / 'manifest.json'),
              'expert_videos_excluded': parent['expert_videos_excluded'],
              'splits': {}, 'episodes': [], 'linked_images': 0}
    for split, episodes in grouped.items():
        count = 0
        with (output / f'{split}.jsonl').open('w') as stream:
            for episode, rows in sorted(episodes.items()):
                meta = metadata[episode]
                if meta['split'] != split:
                    raise ValueError(f'Unexpected split: {episode}')
                task = json.loads(Path(meta['annotation_source']).read_text())['task_prompt']
                with Path(meta['labels_source']).open() as labels_file:
                    labels = {int(r['frame_index']): float(r['pt']) for r in csv.DictReader(labels_file)}
                rows.sort(key=lambda row: row['frame_index'])
                if len({r['frame_index'] for r in rows}) != len(rows):
                    raise ValueError(f'Duplicate sampled frames: {episode}')
                window = ObservationWindow(window_size)
                for row in rows:
                    frame, pt = row['frame_index'], row['pt']
                    if not math.isfinite(pt) or not 0 <= pt <= 1 or abs(pt - labels[frame]) > 1e-8:
                        raise ValueError(f'Invalid or changed PT: {episode}/{frame}')
                    # Only the current main view is read; the source's end-state reference is excluded.
                    original = Path(row['image'][5])
                    image = output / 'images' / episode / f'{frame:08d}.png'
                    image.parent.mkdir(parents=True, exist_ok=True)
                    os.link(original, image)
                    history = window.push(episode, (frame, str(image)))
                    sample = {'id': row['id'], 'episode_id': episode, 'frame_index': frame,
                              'task': task, 'pt': pt,
                              'history_frame_indices': [item[0] for item in history],
                              'image': [item[1] for item in history]}
                    stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    count += 1
                    report['linked_images'] += 1
                report['episodes'].append({'episode_id': episode, 'split': split, 'samples': len(rows),
                                           'task': task, 'labels_sha256': sha256(meta['labels_source'])})
        report['splits'][split] = {'samples': count, 'videos': len(episodes),
                                   'sha256': sha256(output / f'{split}.jsonl')}
    report['status'] = 'complete'
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.source, args.output_dir)
    print(json.dumps({k: report[k] for k in ('status', 'splits', 'linked_images')}, indent=2))


if __name__ == '__main__':
    main()
