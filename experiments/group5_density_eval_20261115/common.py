import sqlite3
from pathlib import Path
from experiments.group5_local_density_20261115.common import read, save, sha, digest, DATA, SOURCE, model_path

ROOT = Path('/cache/cvpzhr/setup/group5-local-density-eval-20261115')
OUT = Path('/cache/cvpzhr/evaluations/group5-local-density-eval-20261115')
ORIGINAL = Path('/cache/cvpzhr/setup/group5-soc35-memory-fps-20261115')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
PACKAGE = 'experiments.group5_density_eval_20261115'
MODELS = ('forest', 'robometer-4b', 'procvlm-2b')
SUITES = ('Order', 'State', 'Count')


def source():
    execution = read(ROOT/'execution.json')
    assert sha(DATA/'inputs.json') == execution['inputs_sha256']
    return read(DATA/'inputs.json'), execution


def verify_code():
    for p,h in read(ROOT/'source-checksums.json')['files'].items():
        assert sha(ROOT/'code'/p) == h, p


def streams(data):
    result = {}
    for level,ref in data['streams'].items():
        assert sha(ref['path']) == ref['sha256']
        result[int(level)] = {e['key']:e for e in read(ref['path'])['episodes']}
    return result


def identity(ih, info, row, entry):
    return dict(inputs_sha256=ih, model=info['model'], suite=info['suite'], key=row['key'],
        checkpoint_sha256=info['checkpoint_sha256'], row_sha256=digest(row), entry_sha256=digest(entry),
        density_level=entry['level'], memory_enabled=True)


def result_path(model, level, key):
    assert level in range(2,6) and '..' not in Path(key).parts and not Path(key).is_absolute()
    return OUT/f'level{level}'/model/'episodes'/(key.replace('/','__')+'.json')


def verify_cache(data, model, row, entry):
    from experiments.group5_memoryfps_20261115.protocol import model_episode
    p = model_path(model, entry['level'], row['key'])
    assert sha(p) == data['cache_files_sha256'][str(p)]
    c = read(p)
    assert c['entry_sha256'] == digest(entry)
    assert c['inference'] == model_episode(row, entry)
    assert c['checkpoint_sha256'] == data['registry'][f'{model}-{row["suite"]}']['checkpoint_sha256']
    assert c['memory_enabled'] and not c['future_endpoint_reference']
    return c


def choose(table):
    passed = [r for r in table if r['passed']]
    assert passed, 'No safe and numerically valid configuration'
    best = max(r['observations_per_second'] for r in passed)
    return min((r for r in passed if r['observations_per_second'] >= .97*best),
               key=lambda r:r['replicas_per_gpu'])


def db(path):
    c = sqlite3.connect(path, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA busy_timeout=30000')
    return c


def initialize_queue(path, items):
    c=db(path)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE IF NOT EXISTS jobs (key TEXT, level INT, observations INT, status TEXT, owner TEXT, PRIMARY KEY(key,level))')
    for key,level,n in items:
        c.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,\'pending\',NULL)',(key,level,n))
    assert c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == len(items)
    assert not c.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone(), 'Old claim requires audited recovery'
    c.close()


def claim(path, owner):
    c=db(path);c.execute('BEGIN IMMEDIATE')
    r=c.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY observations DESC,key,level LIMIT 1").fetchone()
    if r:c.execute("UPDATE jobs SET status='running',owner=? WHERE key=? AND level=?",(owner,r['key'],r['level']))
    c.execute('COMMIT');c.close()
    return dict(r) if r else None


def finish(path, owner, key, level, state='completed'):
    assert state in ('completed','pending')
    c=db(path)
    n=c.execute('UPDATE jobs SET status=?,owner=NULL WHERE key=? AND level=? AND owner=? AND status=\'running\'',
                (state,key,level,owner)).rowcount
    c.close();assert n==1
