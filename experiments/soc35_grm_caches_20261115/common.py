import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/cache/cvpzhr/setup/soc35-grm-caches-20261115')
OUT = Path('/cache/cvpzhr/datasets/soc35-grm-5fps-20261115')
SOURCE = Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114/selection.json')
OLD = Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')
RECOVER = Path('/cache/cvpzhr/setup/recover-five-finetune-20261114')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
MODELS = ('robobrain25-4b', 'dopamine-3b')
SUITES = ('State', 'Order', 'Count')
VIEWS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def write_rows(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in data))
    tmp.replace(p)


def slots(i, n):
    if not 0 <= i < n: raise ValueError('Invalid observation')
    return [(0, 0), (n-1, 0)] + [(0, v) for v in range(3)] + [(i, v) for v in range(3)]


def check_selection(selected, manifests):
    from collections import Counter
    assert len({e['episode_id'] for e in selected}) == len(selected) == 105
    result = {}
    for suite in SUITES:
        episodes = [e for e in selected if e['suite'] == suite]
        counts = Counter(e['task_id'] for e in episodes)
        tasks = {t['task_id']: t for t in manifests[suite]['tasks']}
        assert len(episodes) == 35 and set(counts) == set(tasks)
        assert min(counts.values()) >= 2
        for e in episodes:
            task = tasks[e['task_id']]
            assert not set(task['train']) & set(task['val'])
            assert e['sample_id'] in task['train'] and e['sample_id'] not in task['val']
        result[suite] = dict(episodes=35, tasks=dict(counts), observations=sum(e['samples'] for e in episodes))
    return result


def resource_check():
    import shutil
    base = Path('/sys/fs/cgroup/memory')
    stat = dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
    oom = dict(line.split() for line in (base/'memory.oom_control').read_text().splitlines())
    limit = int((base/'memory.limit_in_bytes').read_text())
    rss = int(stat['total_rss']) + int(stat['total_shmem'])
    assert rss < .55 * limit and int(oom['under_oom']) == 0, 'CPU cache memory pressure'
    baseline = ROOT/'initial-resources.json'
    if baseline.exists(): assert int(oom['oom_kill']) == read(baseline)['oom_kill'], 'New OOM; stop preparation'
    free = shutil.disk_usage('/cache').free
    assert free > 100 * 2**30, 'Insufficient disk reserve'
    return dict(rss_and_shmem=rss, memory_limit=limit, disk_free=free, oom_kill=int(oom['oom_kill']))
