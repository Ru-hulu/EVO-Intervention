"""Refresh the existing FOREST viewer from four independent evaluation shards."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path('/cache/cvpzhr/evaluations')
OUT = BASE / 'forest-s01-critical-val12-step13224-20261106'
MODEL = Path('/cache/cvpzhr/outputs/forest-s01-critical-sampling-20261106/last.pt')
RUNTIME = '/cache/cvpzhr/setup/forest-s01-critical-sampling-20261106/code'
VIEWER = BASE / 'forest-s01-val5-20261106'
SOURCE = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline/State/01-cover_uncover_blocks')
NAME = 'FOREST-S01-Latest'


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copytree(Path(__file__).parent, OUT / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(MODEL, OUT / 'checkpoint.pt')
    sha = digest(OUT / 'checkpoint.pt')
    selected = {}
    for old in ['forest-s01-val10-step13224-20261106', 'forest-s01-val5-step13224-20261106']:
        for row in json.loads((BASE / old / 'selection.json').read_text())['episodes']:
            selected.setdefault(row['episode_index'], row)
    assert len(selected) == 12
    for prefix in ['fa_', 'rec_']:
        assert sum(row['category'].startswith(prefix) for row in selected.values()) == 5
    costs = {ep: sum(1 for line in (Path(row['cache_dir']) / 'samples.jsonl').open() if line.strip())
             for ep, row in selected.items()}
    groups, loads = [[], [], [], []], [0, 0, 0, 0]
    for ep in sorted(selected, key=costs.get, reverse=True):
        gpu = loads.index(min(loads))
        groups[gpu].append(ep)
        loads[gpu] += costs[ep]
    dump(OUT / 'selection.json', dict(episodes=list(selected.values()), checkpoint_sha256=sha,
         source_checkpoint=str(MODEL), shards=groups, expected_samples=sum(costs.values())))
    processes = []
    started = time.time()
    try:
        for gpu, episodes in enumerate(groups):
            folder = OUT / f'gpu{gpu}'
            folder.mkdir()
            with (folder / 'eval.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-u', str(OUT / 'code/evaluate.py'),
                    '--checkpoint', str(OUT / 'checkpoint.pt'), '--output-dir', str(folder),
                    '--episodes', *map(str, episodes), '--batch-size', '8'],
                    stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=RUNTIME,
                             OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            processes.append(process)
        while any(p.poll() is None for p in processes):
            codes = [p.poll() for p in processes]
            dump(OUT / 'status.json', dict(phase='inference', pids=[p.pid for p in processes],
                 returncodes=codes, shards=groups, seconds=time.time() - started))
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'Evaluation shard failed: {codes}')
            time.sleep(5)
        assert all(p.returncode == 0 for p in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()

    target = OUT / 'viewer-data' / NAME
    target.mkdir(parents=True)
    rows, predictions, reports = [], [], []
    for gpu in range(4):
        folder = OUT / f'gpu{gpu}'
        report = json.loads((folder / 'metrics.json').read_text())
        reports.append(report)
        assert report['step'] == 13224 and digest(folder / 'checkpoint.pt') == sha
        exported = folder / 'viewer-data' / report['dataset']
        shard_rows = [json.loads(line) for line in (exported / 'metadata.jsonl').read_text().splitlines()]
        for row in shard_rows:
            row['task_name'] = '覆盖与揭开积木 · 最新加权 FOREST'
            for key in ['annotation_path', 'subtask_boundary_path']:
                assert digest(exported / row[key]) == digest(SOURCE / row[key])
            for path in exported.rglob('*'):
                if path.is_file() and path.name != 'metadata.jsonl':
                    destination = target / path.relative_to(exported)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        os.link(path, destination)
            progress_path = target / 'progress' / (row['sample_id'] + '.json')
            progress = json.loads(progress_path.read_text())
            progress['source'] = '最新加权 FOREST · step 13224 · 5 Hz 因果记忆前向；采样点之间仅显示插值'
            progress['checkpoint_sha256'] = sha
            dump(progress_path, progress)
            rows.append(row)
        predictions.extend(json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines())
    assert len(rows) == 12 and len({r['sample_id'] for r in rows}) == 12
    assert {r['episode_index'] for r in rows} == set(selected)
    assert len(predictions) == sum(costs.values())
    rows.sort(key=lambda row: row['episode_index'])
    (target / 'metadata.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    predictions.sort(key=lambda row: (row['episode_index'], row['frame_index']))
    (OUT / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    errors = [row['predicted_pt'] - row['target_pt'] for row in predictions]
    count = len(errors)
    summary = dict(step=13224, checkpoint_sha256=sha, dataset=NAME, num_samples=count,
        mae_pp=sum(abs(e) for e in errors) / count * 100,
        rmse_pp=(sum(e*e for e in errors) / count)**0.5 * 100,
        bias_pp=sum(errors) / count * 100,
        loss=sum(r['loss'] * r['num_samples'] for r in reports) / count,
        by_episode=[item for report in reports for item in report['by_episode']],
        seconds=time.time() - started, mode='cached causal history; four GPUs; batch 8 per GPU')
    dump(OUT / 'metrics.json', summary)
    for model in ['fa', 'rec']:
        subprocess.run([sys.executable, str(OUT / 'code/plot.py'), str(OUT), '--model', model], check=True)

    # Only switch the visible entry after all predictions, labels and plots validate.
    viewer_code = VIEWER / 'val5-viewer'
    names_file, archive_file = viewer_code / 'catalog_names.json', viewer_code / 'catalog_archive.json'
    names = json.loads(names_file.read_text()) if names_file.exists() else {}
    archive = json.loads(archive_file.read_text()) if archive_file.exists() else {}
    dump(OUT / 'previous-viewer-config.json', dict(names=names, archive=archive))
    live = VIEWER / 'viewer-data' / NAME
    if live.exists():
        assert live.is_symlink(), 'Do not replace a physical dataset directory'
        dump(OUT / 'previous-latest.json', dict(target=str(live.resolve())))
    temporary = live.with_name(NAME + '.next')
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(live)
    names[NAME] = '最新 FOREST · 加权版 step13224 · IMPRINT 各5条 + 脚本2条'
    dump(names_file, names)
    archive['archived'] = sorted((set(archive.get('archived', [])) | {
        'FOREST-S01-Val5-step6600', 'FOREST-S01-Val5-step13224', 'FOREST-S01-Val10-step13224'}) - {NAME})
    dump(archive_file, archive)
    dump(OUT / 'status.json', dict(phase='completed', published=True, result=summary,
         seconds=time.time() - started))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            dump(OUT / 'error.json', dict(error=repr(exc)))
        raise
