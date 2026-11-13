"""Frozen zero-shot protocol and episode-weighted PT correlations."""
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path('/cache/cvpzhr/evaluations/nine-model-zeroshot-cleanval-voc-20261113')
PREVIOUS = Path('/cache/cvpzhr/setup/three-model-e1-cleanval-20261108')
NEW_ENV = '/cache/robot-baselines/envs/procvlm/bin/python'
MODELS = {
    'robometer-4b': dict(kind='robometer', path='/cache/cvpzhr/models/Robometer-4B', python='/cache/cvpzhr/envs/robometer/bin/python', batch=24),
    'robobrain25-4b': dict(kind='grm', path='/cache/cvpzhr/models/RoboBrain2.5-4B', python=NEW_ENV, batch=64, max_tokens=128, min_pixels=4096, max_pixels=65536),
    'robobrain25-8b': dict(kind='grm', path='/cache/robot-baselines/models/RoboBrain2.5-8B-NV', python=NEW_ENV, batch=64, max_tokens=128, min_pixels=4096, max_pixels=65536),
    'dopamine-3b': dict(kind='grm', path='/cache/robot-baselines/models/Robo-Dopamine-GRM-3B', python=NEW_ENV, batch=64, max_tokens=128, min_pixels=12544, max_pixels=76800),
    'dopamine-8b': dict(kind='grm', path='/cache/robot-baselines/models/Robo-Dopamine-GRM-8B', python=NEW_ENV, batch=64, max_tokens=128, min_pixels=12544, max_pixels=76800),
    'procvlm-2b': dict(kind='proc', path='/cache/robot-baselines/models/ProcVLM-2B', python=NEW_ENV, batch=64, max_tokens=4096),
    'vlac-2b': dict(kind='vlac', path='/cache/robot-baselines/models/VLAC', python='/cache/robot-baselines/envs/vlac/bin/python', batch=32, max_tokens=64),
    'vlac-8b': dict(kind='vlac', path='/cache/robot-baselines/models/VLAC-8b', python='/cache/robot-baselines/envs/vlac/bin/python', batch=16, max_tokens=64),
    'r2vlm-alfred-rl': dict(kind='r2', path='/cache/robot-baselines/models/R2VLM-Alfred-RL', python=NEW_ENV, batch=48, max_tokens=1024),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def uniform_indices(i):
    import numpy as np
    return list(range(i + 1)) + [i] * (7 - i) if i < 7 else np.rint(np.linspace(0, i, 8)).astype(int).tolist()


def episode_path(model, episode):
    return OUT / model / 'episodes' / (episode.replace('/', '__') + '.json')


def correlations(pred, target):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    if any(x is None for x in pred):
        return None, None, 'invalid_predictions'
    x, y = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, None, 'nonfinite'
    if len(x) < 2:
        return None, None, 'fewer_than_two_samples'
    if np.ptp(x) == 0:
        return None, None, 'constant_prediction'
    if np.ptp(y) == 0:
        return None, None, 'constant_target'
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic), None


def aggregate(model):
    manifest = read(ROOT / 'manifest.json')
    episodes = []
    missing = []
    invalid = 0
    for e in manifest['episodes']:
        path = episode_path(model, e['episode_id'])
        if not path.exists():
            missing.append(e['episode_id'])
            continue
        r = read(path)
        assert r['protocol_sha256'] == sha(ROOT / 'protocol.json')
        assert r['manifest_sha256'] == sha(ROOT / 'manifest.json')
        assert r['episode_id'] == e['episode_id'] and len(r['predictions']) == len(e['frames'])
        assert r['frame_indices'] == [f['frame_index'] for f in e['frames']]
        assert r['target_pt'] == [f['pt'] for f in e['frames']]
        invalid += sum(p is None for p in r['predictions'])
        p, s, reason = correlations(r['predictions'], r['target_pt'])
        episodes.append(dict(episode_id=e['episode_id'], suite=e['suite'], task=e['task_id'],
                             samples=len(r['predictions']), pearson=p, spearman=s, undefined_reason=reason,
                             invalid_predictions=sum(v is None for v in r['predictions'])))
    def summarize(rr):
        valid = [e for e in rr if e['pearson'] is not None]
        return dict(episodes=len(rr), samples=sum(e['samples'] for e in rr), valid_episodes=len(valid),
                    pearson=sum(e['pearson'] for e in valid) / len(valid) if valid else None,
                    spearman=sum(e['spearman'] for e in valid) / len(valid) if valid else None,
                    undefined_reasons=dict(Counter(e['undefined_reason'] for e in rr if e['undefined_reason'])),
                    invalid_predictions=sum(e['invalid_predictions'] for e in rr))
    result = dict(model=model, checkpoint=MODELS[model]['path'], zero_shot=True,
                  uses_future_goal_reference=MODELS[model]['kind'] == 'grm',
                  complete=not missing, missing_episodes=missing, invalid_predictions=invalid,
                  aggregation='episode equal; predictions vs original PT; tied ranks averaged; constants undefined',
                  overall=summarize(episodes), by_suite={s:summarize([e for e in episodes if e['suite'] == s]) for s in ['State','Order','Count']},
                  by_episode=episodes, protocol_sha256=sha(ROOT / 'protocol.json'), updated_at=now())
    write(OUT / model / 'metrics.json', result)
    return result
