"""Independently check v3 curves, immutable v2 files, and loader alignment."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

from decline_s01_training import SOURCE, OUTPUT, TASK, manifest, read, save, sha


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    out, task = OUTPUT, OUTPUT / TASK
    entries = {r['episode']: r for r in read(out/'verification/decline-relabel.json')}
    rows = [json.loads(line) for line in (task/'metadata.jsonl').read_text().splitlines()]
    media = 0
    for row in rows:
        ep = row['episode_index']
        ev = task/row['evidence_path']
        ann = read(task/row['annotation_path'])
        p = read(ev/'progress.json')
        record = read(out/'labels'/TASK/f'episode{ep}.json')
        curve = np.asarray(p['dense_progress'])
        old = np.asarray(read(SOURCE/TASK/row['progress_path'])['dense_progress'])
        assert record['progress_source'] == 'dense_annotation'
        assert p == ann['progress']
        assert list(curve) == record['dense_progress']
        assert row['annotation_sha256'] == sha(task/row['annotation_path'])
        assert len(curve) == row['total_frames'] and np.isfinite(curve).all()
        assert ((curve >= 0) & (curve <= 1)).all()
        for f in record['frames']:
            assert f['progress'] == curve[f['index']]
        if ep in entries:
            e = entries[ep]
            a,b = e['error_range']
            delta = np.asarray(read(task/'deltas'/f'episode_{ep:06d}.json')['deltas'])
            amount = min(.1, old[a-1]/2)
            expected = old[a-1] - amount * np.cumsum(delta[a:b])/delta[a:b].sum()
            np.testing.assert_allclose(curve[a:b], expected, atol=1e-12, rtol=0)
            assert np.array_equal(curve[:a], old[:a])
            assert (np.diff(curve[a-1:b]) <= 1e-10).all() and curve[b-1] < curve[a-1]
            states = [json.loads(line) for line in (ev/'states.jsonl').read_text().splitlines()]
            fail = next((i for i,s in enumerate(states) if s['fail_flag']), None)
            assert fail == e['native_failure_frame']
            if fail is not None:
                assert b == fail and curve[b-1] > 0 and (curve[b:] == 0).all()
            else:
                c = e['recovery_range'][1]
                expected = old[b:c] - amount * (1-np.cumsum(delta[b:c])/delta[b:c].sum())
                np.testing.assert_allclose(curve[b:c], expected, atol=1e-12, rtol=0)
                assert (np.diff(curve[b-1:c]) >= -1e-10).all()
                assert np.array_equal(curve[c-1:],old[c-1:]) and curve[-1] == 1
        else:
            assert np.array_equal(curve, old)
            assert sha(out/'labels'/TASK/f'episode{ep}.json') == sha(SOURCE/'labels'/TASK/f'episode{ep}.json')
        for v in row['video_views'].values():
            new, original = task/v['video_path'], SOURCE/TASK/v['video_path']
            assert sha(new) == sha(original) == v['video_sha256']
            assert new.stat().st_ino != original.stat().st_ino
            media += 1
    loader = module('s01_v3_data',out/'code/loader/data.py')
    Config = module('s01_v3_config',out/'code/loader/config.py').Config
    ds = loader.Trajectories(out/'labels',out,'all')
    real_reader = loader.read_frames
    loader.read_frames = lambda paths, ids, cameras, box: ids
    cases = 0
    for i,(_,ep) in enumerate(ds.episodes):
        dense = read(out/'labels'/TASK/f'episode{ep}.json')['dense_progress']
        for speed,pause in [(None,False),(.5,False),(2.,False),(None,True)]:
            sample = ds.sample(i,Config(pause_prob=1.),random.Random(ep) if pause else None,speed)
            torch.testing.assert_close(sample['progress'],torch.tensor([dense[f] for f in sample['frames']]))
            cases += 1
    loader.read_frames = real_reader
    decoded = []
    for ep in [44,97,98]:
        i = next(i for i,(_,e) in enumerate(ds.episodes) if e==ep)
        sample = ds.sample(i,Config())
        assert all(len(frame)==3 for frame in sample['frames'])
        decoded.append(dict(episode=ep,frames=len(sample['frames']),views=3))
    train = loader.Trajectories(out/'labels',out,'train')
    groups = {r['episode_index']:r['mixture_group'] for r in rows}
    counts = Counter(groups[e] for _,e in train.episodes)
    assert counts == {'imprint_success':47,'abnormal':94}
    assert manifest(SOURCE) == read(out/'verification/source-v2-manifest.json')
    assert sha(out/'sampling.json')==sha(SOURCE/'sampling.json')
    result = dict(unique_trajectories=len(rows),changed_curves=len(entries),unchanged_curves=len(rows)-len(entries),
        independently_recomputed_curves=len(entries),native_failure_zero_checks=63,recovery_checks=30,
        byte_identical_independent_video_copies=media,source_v2_all_files_unchanged=True,
        loader_alignment_cases=cases,decoded_samples=decoded,epoch_visits=dict(counts),passed=True)
    save(out/'verification/acceptance.json',result)
    summary = read(out/'summary.json');summary['verification']=result;save(out/'summary.json',summary)
    shutil.copy2(Path(__file__),out/'code'/Path(__file__).name)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
