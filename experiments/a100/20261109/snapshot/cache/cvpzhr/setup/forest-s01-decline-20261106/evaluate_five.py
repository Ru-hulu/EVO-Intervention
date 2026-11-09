"""Evaluate a fixed, diverse five-trajectory validation subset; export Dataset Viewer."""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device

SETUP = Path('/cache/cvpzhr/setup/forest-s01-decline-20261106')
SOURCE = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline/State/01-cover_uncover_blocks')
OUT = Path('/cache/cvpzhr/evaluations/forest-s01-val5-20261106')
EPISODES = [4, 54, 81, 99, 132]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def metrics(records):
    e = np.array([r['predicted_pt'] - r['target_pt'] for r in records])
    return dict(num_samples=len(e), mae_pp=float(abs(e).mean()*100),
                rmse_pp=float(np.sqrt((e*e).mean())*100), bias_pp=float(e.mean()*100))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT/'checkpoint.pt'
    assert not checkpoint.exists(), 'Keep existing evaluation immutable'
    shutil.copyfile('/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106/last.pt', checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    step = saved['step']
    del saved
    rows = [r for r in json.loads((SETUP/'split.json').read_text())['records'] if r['episode_index'] in EPISODES]
    assert len(rows)==5 and all(r['split'] in ('val','validation') for r in rows)
    rows.sort(key=lambda r: EPISODES.index(r['episode_index']))
    dump(OUT/'selection.json', dict(step=step, episodes=rows, selection='Fixed category coverage, not selected by prediction quality', mode='cached causal history; 5 Hz; no augmentation'))
    print(json.dumps(dict(event='loading', step=step, episodes=EPISODES)), flush=True)
    model, processor, cfg = load_checkpoint(str(checkpoint))
    dataset = M3Dataset([str(Path(r['cache_dir'])/'samples.jsonl') for r in rows], cfg['model']['history_dim'])
    assert dataset.history_source == cfg['history_source'] and dataset.history_capacity == cfg['history_capacity']
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2,
        multiprocessing_context='spawn', pin_memory=True,
        collate_fn=Stage3Collator(processor, dataset.history_dim, cfg['max_sequence_length'], dataset.history_capacity))
    records, offset, loss_sum = [], 0, 0.0
    started = time.monotonic()
    with torch.inference_mode(), (OUT/'predictions.jsonl').open('w') as f:
        for batch in loader:
            result = model(to_device(batch['inputs'], 'cuda'), batch['target_pt'].to('cuda'))
            prediction = result['pt'].float().cpu().tolist()
            assert all(np.isfinite(prediction))
            loss_sum += float(result['loss_sum'])
            for p, target in zip(prediction, batch['target_pt'].tolist()):
                ep, sample, _ = dataset.samples[offset]
                record = dict(episode_index=rows[ep]['episode_index'], frame_index=sample['frame_index'], target_pt=target, predicted_pt=p)
                records.append(record);f.write(json.dumps(record)+'\n');offset += 1
            if offset % 80 == 0:
                f.flush();print(json.dumps(dict(event='progress', done=offset, total=len(dataset), seconds=time.monotonic()-started)),flush=True)
    assert len(records)==len(dataset)
    name = f'FOREST-S01-Val5-step{step}'
    root = OUT/'viewer-data'/name
    root.mkdir(parents=True)
    exported, by_episode = [], []
    for row in rows:
        subset = [p for p in records if p['episode_index']==row['episode_index']]
        annotation = json.loads((SOURCE/row['annotation_path']).read_text())
        truth = annotation['progress']['dense_progress']
        assert subset[0]['frame_index']==0 and subset[-1]['frame_index']==len(truth)-1
        assert all(abs(p['target_pt']-truth[p['frame_index']])<1e-6 for p in subset)
        prediction = np.interp(np.arange(len(truth)), [p['frame_index'] for p in subset], [p['predicted_pt'] for p in subset]).tolist()
        m = dict(episode_index=row['episode_index'], category=row['category'], sample_id=row['sample_id'], **metrics(subset))
        by_episode.append(m)
        item = dict(row)
        item['task_name'] = '覆盖与揭开积木 · FOREST 验证预测'
        item['failure_reason'] = f"{row['category']} | MAE {m['mae_pp']:.2f} pp | step {step}"
        # Export private small media copies on Huawei; never edit source labels or viewers.
        for key in ['annotation_path', 'subtask_boundary_path']:
            if not item.get(key): continue
            dest=root/item[key];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/item[key],dest)
        for view in item['video_views'].values():
            dest=root/view['video_path'];dest.parent.mkdir(parents=True,exist_ok=True)
            try: os.link(SOURCE/view['video_path'],dest)
            except OSError: shutil.copyfile(SOURCE/view['video_path'],dest)
        ppath=root/'progress'/f"{item['sample_id']}.json";ppath.parent.mkdir(exist_ok=True)
        dump(ppath,dict(policy='forest_prediction_v1', total=1, dense_progress=truth,
             predicted_progress=prediction, prediction_points=[[p['frame_index'],p['predicted_pt']] for p in subset],
             source=f'FOREST step {step} · 5 Hz 因果记忆前向；预测点之间仅作显示插值', metrics=m, checkpoint_step=step))
        exported.append(item)
    (root/'metadata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exported))
    report=dict(step=step,mode='cached causal history',loss=loss_sum/len(records),seconds=time.monotonic()-started,
        **metrics(records),by_episode=by_episode,dataset=name)
    dump(OUT/'metrics.json', report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
