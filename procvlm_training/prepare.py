"""Join the new seven-task labels to existing sampled images without copying images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import answer_text, user_prompt
from .window import ObservationWindow, history_window, sample_frame_index


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config):
    source = Path(config['source_dataset'])
    label_path = Path(config['remaining_actions_manifest'])
    labels = json.loads(label_path.read_text())
    source_manifest = json.loads((source / 'manifest.json').read_text())
    if labels['sample_fps'] != 5 or source_manifest['sample_fps'] != 5 or config['window_size'] != 8:
        raise ValueError('Expected the existing 5 FPS / 8-frame configuration')
    metadata = {row['episode_id']: row for row in source_manifest['episodes']}
    original = {}
    for split in ('train', 'validation'):
        for line in (source / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            original[(row['episode_id'], row['frame_index'])] = row
    output = Path(config['dataset_dir'])
    output.mkdir(parents=True, exist_ok=False)
    rows_by_split = {'train': [], 'validation': []}
    episode_reports, sidecar_hashes = [], {}
    for entry in labels['episodes']:
        if entry['split'] == 'expert':
            continue
        if entry['episode_id'].startswith('03-'):
            raise ValueError('输入数字序列 must remain excluded')
        path = label_path.parent / entry['file']
        sidecar_hashes[entry['file']] = digest(path)
        label = json.loads(path.read_text())
        episode, split = label['episode_id'], label['split']
        meta = metadata[episode]
        if meta['split'] != split:
            raise ValueError('Source split mismatch')
        frames = [(r['frame_index'], original[(episode, r['frame_index'])]['image'][5]) for r in label['samples_5fps']]
        expected = [sample_frame_index(i, meta['fps']) for i in range(len(frames))]
        if [frame for frame, _ in frames] != expected or len(frames) != meta['sample_count']:
            raise ValueError('5 FPS sampling mismatch')
        online = ObservationWindow(8)
        episode_rows = []
        for i, target in enumerate(label['samples_5fps']):
            old = original[(episode, target['frame_index'])]
            if old['pt'] != target['pt'] or not Path(frames[i][1]).is_file():
                raise ValueError('Original PT or image is unavailable')
            window = online.push(episode, frames[i])
            if window != history_window(frames, i, 8) or any(frame > frames[i][0] for frame, _ in window):
                raise ValueError('Non-causal history')
            eligible = not target['needs_pt_terminal_review']
            row = {'id': f'{episode}/{target["frame_index"]:08d}', 'episode_id': episode,
                'frame_index': target['frame_index'], 'timestamp_seconds': target['frame_index'] / meta['fps'],
                'task': label['task_prompt'], 'pt': target['pt'], 'remaining_actions': target['remaining_actions'],
                'image': [p for _, p in window], 'window_frame_indices': [f for f, _ in window],
                'supervision_eligible': eligible,
                'conversations': [{'from': 'human', 'value': '<image>\n' * 8 + user_prompt(label['task_prompt'])},
                                  {'from': 'gpt', 'value': answer_text(target['remaining_actions'], target['pt'])}]}
            episode_rows.append(row)
        rows_by_split[split].extend(episode_rows)
        episode_reports.append({'episode_id': episode, 'split': split, 'source_fps': meta['fps'],
            'samples': len(episode_rows), 'loss_samples': sum(r['supervision_eligible'] for r in episode_rows)})
    counts = {}
    for split, rows in rows_by_split.items():
        with (output / f'{split}.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        counts[split] = {'videos': len({r['episode_id'] for r in rows}), 'samples': len(rows),
                        'loss_samples': sum(r['supervision_eligible'] for r in rows),
                        'excluded_from_loss': sum(not r['supervision_eligible'] for r in rows),
                        'sha256': digest(output / f'{split}.jsonl')}
    manifest = {'schema': 'procvlm-seven-task-sft-v1', 'sample_fps': 5, 'window_size': 8,
        'view': 'cam_high', 'splits': counts, 'episodes': episode_reports,
        'task_video_counts': dict(Counter(r['episode_id'].split('/')[0] for r in episode_reports)),
        'source_manifest': str(source / 'manifest.json'), 'source_manifest_sha256': digest(source / 'manifest.json'),
        'remaining_actions_manifest': str(label_path), 'remaining_actions_manifest_sha256': digest(label_path),
        'sidecar_sha256': sidecar_hashes,
        'expert_videos_excluded': labels['splits']['expert']['videos'],
        'terminal_policy': 'Keep all original frames/PT. Filter conflicts for training and teacher-forced eval loss only; generate predictions for the full validation set.',
        'images_copied': 0, 'training_started': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(json.loads(args.config.read_text()))
    print(json.dumps(manifest['splits'], ensure_ascii=False, indent=2))
