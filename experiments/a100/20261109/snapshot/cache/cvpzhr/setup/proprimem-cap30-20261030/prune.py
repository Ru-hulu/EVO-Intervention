"""Cap the A100-local ProPriMem derivative; plan first, then apply explicitly."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import re
import sys

ROOT = Path('/cache/cvpzhr/datasets/ProPriMem')
REPORT = Path('/cache/cvpzhr/setup/proprimem-cap30-20261030')
REVISION = '9ea7859fd65408152cab93a3fb4fb0ad0acf6dca'


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.cap30-tmp')
    tmp.write_text(text)
    tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_plan():
    manifest = read(ROOT / 'manifest.json')
    edits, deletions, tasks = {}, set(), []
    for task in manifest['tasks']:
        rel = task['task_path']
        base = ROOT / rel
        index = base / 'metadata.jsonl'
        original = rows(index)
        assert len({r['episode_index'] for r in original}) == len(original), rel
        assert set(task['train'] + task['val']) == {r['sample_id'] for r in original}, rel
        keep_ids = {r['episode_index'] for r in sorted(original, key=lambda r: int(r['episode_index']))[:30]}
        kept = [r for r in original if r['episode_index'] in keep_ids]
        removed = [r for r in original if r['episode_index'] not in keep_ids]
        removed_ids = {int(r['episode_index']) for r in removed}
        info = dict(task_path=rel, before=len(original), after=len(kept),
                    kept_episode_ids=sorted(keep_ids), deleted_episode_ids=sorted(removed_ids),
                    annotated_before=sum(bool(r['has_annotation']) for r in original),
                    annotated_after=sum(bool(r['has_annotation']) for r in kept),
                    train_before=len(task['train']), val_before=len(task['val']))
        ids = {r['sample_id'] for r in kept}
        for split in ('train', 'val'):
            task[split] = [s for s in task[split] if s in ids]
            info[split + '_after'] = len(task[split])
        tasks.append(info)
        if not removed:
            continue
        for path in base.rglob('*'):
            if not path.is_file():
                continue
            match = re.match(r'episode_(\d+)(?:\.|$)', path.name)
            if match and int(match[1]) in removed_ids:
                deletions.add(str(path.relative_to(ROOT)))
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept)
        edits[str(index.relative_to(ROOT))] = text
        task['trajectory_index_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        meta = read(base / 'metadata.json')
        meta.update(trajectory_count=len(kept), annotated_trajectory_count=info['annotated_after'],
                    unannotated_trajectory_count=len(kept)-info['annotated_after'],
                    domain_counts=dict(Counter(r['domain'] for r in kept)),
                    view_trajectory_counts=dict(Counter(v for r in kept for v in r['video_views'])),
                    video_file_count=sum(len(r['video_views']) for r in kept),
                    annotation_file_count=sum(bool(r['annotation_path']) for r in kept))
        for source in meta['source_tasks']:
            source['trajectory_count'] = sum(r['source_task_id'] == source['source_task_id'] for r in kept)
        edits[rel + '/metadata.json'] = encode(meta)

    for name in ('insert-screw', 'insert-screw-recover'):
        base = ROOT / name
        custom = read(base / 'multiview_manifest.json')
        original = custom['episodes']
        keep = {r['episode_id'] for r in sorted(original, key=lambda r: int(r['episode_id'].split('-')[-1]))[:30]}
        removed = {r['episode_id'] for r in original} - keep
        tasks.append(dict(task_path=name, before=len(original), after=len(keep),
                          kept_episode_ids=sorted(keep), deleted_episode_ids=sorted(removed)))
        if not removed:
            continue
        for path in base.rglob('*'):
            if not path.is_file():
                continue
            match = re.match(r'(file-\d+)(?:[.-]|$)', path.name)
            if match and match[1] in removed:
                deletions.add(str(path.relative_to(ROOT)))
        # This whole note documents videos 048--057, which are all removed.
        note = base / 'splitlable/特殊进展记录_048-057.txt'
        assert all(f'file-{i:03d}' in removed for i in range(48, 58))
        deletions.add(str(note.relative_to(ROOT)))
        custom['episodes'] = [r for r in original if r['episode_id'] in keep]
        edits[name + '/multiview_manifest.json'] = encode(custom)

    for suite in ('Count', 'Order', 'State'):
        group = [t for t in manifest['tasks'] if t['suite'] == suite]
        manifest['summary'][suite] = dict(tasks=len(group), train=sum(len(t['train']) for t in group),
                                         val=sum(len(t['val']) for t in group))
        count = sum(t['after'] for t in tasks if t['task_path'].startswith(suite + '/'))
        path = ROOT / suite / 'README.md'
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if re.fullmatch(r'\d+ tasks · [\d,]+ trajectories\.', line):
                lines[i] = f'{len(group)} tasks · {count:,} trajectories.'
            for t in tasks:
                if t['task_path'].startswith(suite + '/') and f"({t['task_path'].split('/')[-1]}/)" in line:
                    parts = line.split('|')
                    assert len(parts) == 8
                    parts[5] = f" {t['after']} "
                    lines[i] = '|'.join(parts)
        edits[str(path.relative_to(ROOT))] = '\n'.join(lines) + '\n'
    manifest['totals'] = dict(tasks=len(manifest['tasks']),
                             trajectories=sum(t['after'] for t in tasks if '/' in t['task_path']),
                             train=sum(len(t['train']) for t in manifest['tasks']),
                             val=sum(len(t['val']) for t in manifest['tasks']))
    manifest['local_subset'] = dict(source_repository='MINT-SJTU/ProPriMem', source_revision=REVISION,
                                   episode_cap_per_task=30, selection='lowest numeric episode IDs',
                                   split_handling='Filter existing assignments; do not reassign retained episodes',
                                   report=str(REPORT / 'receipt.json'))
    edits['manifest.json'] = encode(manifest)
    counts = manifest['totals']
    text = ROOT.joinpath('README.md').read_text()
    text = re.sub(r'31 tasks · [\d,]+ trajectories\.', f"31 tasks · {counts['trajectories']:,} trajectories.", text)
    for suite, group in manifest['summary'].items():
        text = re.sub(rf'(\| \[{suite}\].*?\| \d+ \| )[\d,]+( \|)',
                      lambda m: m[1] + str(group['train']+group['val']) + m[2], text)
    text = re.sub(r'[\d,]+ train / [\d,]+ val', f"{counts['train']:,} train / {counts['val']:,} val", text)
    text += ('\nLocal A100 subset: at most 30 trajectories per task, retaining the lowest numeric episode IDs. '
             'Original split assignments are filtered without reassignment; some tasks now have no validation episodes.\n'
             'Additional root tasks (outside the 31-task manifest): insert-screw 30 trajectories; '
             'insert-screw-recover 22 trajectories. Total across all 33 tasks: 868 trajectories.\n'
             f'Source snapshot: {REVISION}. This is a locally reduced derivative, not the full upstream snapshot.\n')
    edits['README.md'] = text
    inventory = {str(p.relative_to(ROOT)): p.stat().st_size for p in ROOT.rglob('*') if p.is_file()}
    assert not any(p.is_symlink() for p in ROOT.rglob('*'))
    assert not set(edits) & deletions
    assert deletions <= inventory.keys()
    plan = dict(root=str(ROOT), source_revision=REVISION, selection='numeric episode order; first 30 per task',
                tasks=tasks, episodes_before=sum(t['before'] for t in tasks), episodes_after=sum(t['after'] for t in tasks),
                changed_tasks=sum(t['before'] != t['after'] for t in tasks), files_before=len(inventory),
                bytes_before=sum(inventory.values()), deleted_files=[dict(path=p, bytes=inventory[p]) for p in sorted(deletions)],
                deleted_bytes=sum(inventory[p] for p in deletions),
                edited_files=[dict(path=p, before_sha256=sha(ROOT/p), after_sha256=hashlib.sha256(s.encode()).hexdigest()) for p,s in edits.items()])
    for rel, content in edits.items():
        atomic(REPORT / 'prepared' / rel, content)
    atomic(REPORT/'plan.json', encode(plan))
    return plan


def verify(plan):
    manifest = read(ROOT/'manifest.json')
    mismatches = []
    for task in manifest['tasks']:
        base = ROOT/task['task_path']
        index = base/'metadata.jsonl'
        data = rows(index)
        meta = read(base/'metadata.json')
        assert len(data) <= 30 and meta['trajectory_count'] == len(data), task['task_path']
        assert sha(index) == task['trajectory_index_sha256'], task['task_path']
        assert set(task['train']) | set(task['val']) == {r['sample_id'] for r in data}
        assert not set(task['train']) & set(task['val'])
        video_paths = {v['video_path'] for r in data for v in r['video_views'].values()}
        actual_videos = {str(p.relative_to(base)) for p in (base/'videos').rglob('*.mp4')}
        assert video_paths == actual_videos, task['task_path']
        assert meta['video_file_count'] == len(actual_videos)
        assert meta['view_trajectory_counts'] == dict(Counter(v for r in data for v in r['video_views']))
        assert meta['annotated_trajectory_count'] == sum(bool(r['annotation_path']) for r in data)
        for r in data:
            if r['annotation_path']:
                assert (base/r['annotation_path']).is_file()
        retained_ids = {int(r['episode_index']) for r in data}
        for p in base.rglob('*'):
            match = re.match(r'episode_(\d+)(?:\.|$)', p.name)
            if match:
                assert int(match[1]) in retained_ids, str(p)
    for name in ('insert-screw', 'insert-screw-recover'):
        base = ROOT/name
        data = read(base/'multiview_manifest.json')['episodes']
        assert len(data) <= 30
        expected = {v['path'] for r in data for v in r['views'].values()}
        assert expected == {str(p.relative_to(base)) for p in base.rglob('*.mp4')}
        retained = {r['episode_id'] for r in data}
        for p in base.rglob('*'):
            match = re.match(r'(file-\d+)(?:[.-]|$)', p.name)
            if match:
                assert match[1] in retained, str(p)
    assert all(not (ROOT/f['path']).exists() for f in plan['deleted_files'])
    upstream = {f['rfilename']: f for f in read(Path('/cache/cvpzhr/setup/dataset-replacement-20261030/repo-info.json'))['siblings']}
    edited = {e['path']: e['after_sha256'] for e in plan['edited_files']}
    verified = 0
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        rel = str(p.relative_to(ROOT))
        if rel in edited:
            assert sha(p) == edited[rel], rel
        else:
            source = upstream[rel]
            lfs = source.get('lfs')
            if lfs:
                actual, expected = sha(p), lfs['sha256']
            else:
                actual = hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()
                expected = source['blobId']
            if actual != expected:
                mismatches.append(rel)
        verified += 1
    assert not mismatches, mismatches
    return dict(all_retained_files_verified=True, verified_files=verified, orphan_episode_files=0,
                dangling_dataset_references=0, max_episodes_per_task=30)


def apply():
    plan = read(REPORT/'plan.json')
    assert plan['root'] == str(ROOT) and not (REPORT/'receipt.json').exists()
    for f in plan['deleted_files']:
        assert (ROOT/f['path']).stat().st_size == f['bytes'], f['path']
    for f in plan['edited_files']:
        assert sha(ROOT/f['path']) == f['before_sha256'], f['path']
        assert sha(REPORT/'prepared'/f['path']) == f['after_sha256']
    atomic(REPORT/'status.json', encode(dict(phase='applying', started_at=datetime.now(timezone.utc).isoformat())))
    for f in plan['deleted_files']:
        (ROOT/f['path']).unlink()
    for f in plan['edited_files']:
        atomic(ROOT/f['path'], (REPORT/'prepared'/f['path']).read_text())
    checks = verify(plan)
    remaining = [p for p in ROOT.rglob('*') if p.is_file()]
    receipt = {k:v for k,v in plan.items() if k not in ('deleted_files','edited_files')}
    receipt.update(status='completed', deleted_file_count=len(plan['deleted_files']),
                   files_after=len(remaining), bytes_after=sum(p.stat().st_size for p in remaining), checks=checks,
                   completed_at=datetime.now(timezone.utc).isoformat(), edited_records=len(plan['edited_files']),
                   training_started=False)
    receipt['bytes_freed'] = receipt['bytes_before'] - receipt['bytes_after']
    atomic(REPORT/'receipt.json', encode(receipt))
    ready_path = Path('/cache/cvpzhr/setup/readiness.json')
    ready = read(ready_path)
    ready['raw_dataset'] = dict(status='local_cap30_subset', path=str(ROOT), source_revision=REVISION,
                                files=receipt['files_after'], bytes=receipt['bytes_after'], episodes=receipt['episodes_after'],
                                tasks=len(plan['tasks']), selection=plan['selection'], checks=checks, receipt=str(REPORT/'receipt.json'))
    ready['training_data_ready'] = False
    ready['training_configuration_valid'] = False
    atomic(ready_path, encode(ready))
    atomic(REPORT/'status.json', encode(dict(phase='completed', receipt=str(REPORT/'receipt.json'))))
    return receipt


REPORT.mkdir(parents=True, exist_ok=True)
with (REPORT/'lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = apply() if sys.argv[1] == 'apply' else build_plan()
    print(encode({k:v for k,v in result.items() if k not in ('deleted_files','edited_files','tasks')}))
    print(encode([dict(task=t['task_path'], before=t['before'], after=t['after'],
                       annotated_after=t.get('annotated_after'), val_after=t.get('val_after')) for t in result['tasks']]))
