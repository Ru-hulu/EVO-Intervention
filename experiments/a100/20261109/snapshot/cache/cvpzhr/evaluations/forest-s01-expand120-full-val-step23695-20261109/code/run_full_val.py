"""Forward every frozen validation episode and audit post-failure false progress."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

SETUP = Path('/cache/cvpzhr/setup/forest-s01-order102-expand120-20261109')
BASE = Path('/cache/cvpzhr/evaluations')
OUT = BASE/'forest-s01-expand120-full-val-step23695-20261109'
VIEW = BASE/'forest-s01-val5-20261106'
CODE = Path(__file__).resolve().parent
PY = '/cache/cvpzhr/envs/forest/bin/python'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def native_frames(row):
    evidence = Path(row['dataset_task_root'])/row['evidence_path']
    if (evidence/'geometry.json').exists():
        geometry = read(evidence/'geometry.json')
        order = geometry['open_order']
        frames = [(f['frame'], f['fail'], f['covers'], f['blocks']) for f in geometry['frames']]
    else:
        order = read(evidence/'result.json')['memory']['color_order_positions']
        frames = [(f['frame'], f['fail_flag'], [p['p'] for p in f['poses']['covers']],
                   [p['p'] for p in f['poses']['blocks']])
                  for f in map(json.loads, (evidence/'states.jsonl').read_text().splitlines())]
    failure = next((i for i, failed, _, _ in frames if failed), None)
    same_state = set()
    for index, failed, covers, blocks in frames:
        covered = [abs(c[0]-b[0])<.03 and abs(c[1]-b[1])<.03 and c[2]<.742 for c,b in zip(covers,blocks)]
        if failed and covered == [i == order[2] for i in range(3)]:
            same_state.add(index)
    return failure, same_state


def longest(points, row, threshold, allowed=None):
    current = best = 0.
    for i, point in enumerate(points):
        end = points[i+1]['frame_index'] if i+1<len(points) else row['total_frames']
        active = point['predicted_pt']>threshold and (allowed is None or point['frame_index'] in allowed)
        current = current+(end-point['frame_index'])/row['fps'] if active else 0.
        best = max(best,current)
    return best


def analyze(row, points):
    failure, same_state = native_frames(row)
    error = np.array([p['predicted_pt']-p['target_pt'] for p in points])
    result = dict(episode=row['episode_index'], sample_id=row['sample_id'], category=row['category'],
        source=row['trajectory_source'], layout=row['layout'], hard102=row['hard102'],
        task_success=row['task_success'], samples=len(points), mae_pp=float(abs(error).mean()*100),
        failure_frame=failure, sustained_post_failure_over20=False, sustained_same_state_over20=False)
    if failure is None:
        return result
    post = [p for p in points if p['frame_index']>=failure]
    assert post and all(p['target_pt']==0 for p in post)
    joined = [p for p in post if p['frame_index'] in same_state]
    run = longest(post,row,.2)
    same_run = longest(post,row,.2,same_state)
    result.update(post_failure_samples=len(post),post_failure_mae_pp=float(np.mean([p['predicted_pt'] for p in post])*100),
        post_failure_peak_pct=max(p['predicted_pt'] for p in post)*100,
        post_failure_above20_pct=sum(p['predicted_pt']>.2 for p in post)/len(post)*100,
        longest_over20_seconds=run,longest_over80_seconds=longest(post,row,.8),
        sustained_post_failure_over20=run>=1.,same_state_samples=len(joined),
        same_state_mae_pp=float(np.mean([p['predicted_pt'] for p in joined])*100) if joined else None,
        longest_same_state_over20_seconds=same_run,sustained_same_state_over20=same_run>=1.)
    return result


def main():
    assert not OUT.exists(), 'Do not overwrite a previous evaluation'
    OUT.mkdir()
    shutil.copytree(CODE,OUT/'code',ignore=shutil.ignore_patterns('__pycache__'))
    checkpoint=OUT/'source-checkpoint.pt'
    shutil.copy2(Path('/cache/cvpzhr/outputs/forest-s01-order102-expand120-20261109/best.pt'),checkpoint)
    assert torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['step']==23695
    rows=[r for r in read(SETUP/'split.json')['records'] if r['split']=='validation']
    assert len(rows)==47 and all(r['episode_index']!=55 for r in rows)
    save(OUT/'selection.json',dict(rows=rows,checkpoint_step=23695,ep55='separate diagnostic; excluded from validation',
        thresholds=dict(high=.2,very_high=.8,sustained_seconds=1),mode='normal cached causal history, 5 Hz, batch 2'))
    # Balance the two free GPUs by actual cached observation counts.
    loads=[0,0];shards=[[],[]]
    for row in sorted(rows,key=lambda r:read(Path(r['cache_dir'])/'summary.json')['num_samples'],reverse=True):
        gpu=min(range(2),key=lambda g:loads[g]);shards[gpu].append(row)
        loads[gpu]+=read(Path(row['cache_dir'])/'summary.json')['num_samples']
    save(OUT/'status.json',dict(phase='running',gpu_samples=loads,episodes=47,updated=time.time()))
    jobs=[]
    for gpu,shard in enumerate(shards):
        with (OUT/f'gpu{gpu}.log').open('w') as log:
            jobs.append(subprocess.Popen([PY,'-u',str(OUT/'code/evaluate_full_val.py'),
                '--checkpoint',str(checkpoint),'--output-dir',str(OUT/f'gpu{gpu}'),
                '--episodes',*[str(r['episode_index']) for r in shard],'--batch-size','2'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(SETUP/'code'),
                    OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
                stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert codes==[0,0],codes
    records=[];exported=[]
    target=OUT/'viewer-data'/'FOREST-S01-Val47-step23695'
    target.mkdir(parents=True)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for gpu in range(2):
        folder=OUT/f'gpu{gpu}'
        records.extend(json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines())
        source=folder/'viewer-data'/read(folder/'metrics.json')['dataset']
        for row in map(json.loads,(source/'metadata.jsonl').read_text().splitlines()):
            paths=[row['annotation_path'],row['subtask_boundary_path'],*[v['video_path'] for v in row['video_views'].values()]]
            for path in paths:
                dst=target/path;dst.parent.mkdir(parents=True,exist_ok=True)
                assert not dst.exists(), ('Export path collision',path)
                os.link(source/path,dst)
            pp=Path('progress')/(row['sample_id']+'.json');(target/pp).parent.mkdir(exist_ok=True)
            data=read(source/pp);data.update(checkpoint_sha256=sha,source='扩充120条脚本 · 最佳 step23695 · 全部47条固定val · 正常因果历史')
            save(target/pp,data)
            row['task_name']='覆盖与揭开积木 · 全部验证集 · step23695'
            exported.append(row)
    assert len(exported)==47 and len(records)==15816
    assert len({(p['episode_index'],p['frame_index']) for p in records})==15816
    groups=collections.defaultdict(list)
    for point in records:groups[point['episode_index']].append(point)
    episodes=[analyze(r,sorted(groups[r['episode_index']],key=lambda p:p['frame_index'])) for r in rows]
    failed=[e for e in episodes if e['failure_frame'] is not None]
    errors=np.array([r['predicted_pt']-r['target_pt'] for r in records])
    report=dict(checkpoint_step=23695,checkpoint_sha256=sha,episodes=47,samples=15816,
        mae_pp=float(abs(errors).mean()*100),native_failure_episodes=len(failed),
        sustained_post_failure_over20=sum(e['sustained_post_failure_over20'] for e in failed),
        sustained_same_state_over20=sum(e['sustained_same_state_over20'] for e in failed),
        by_episode=sorted(episodes,key=lambda e:e.get('post_failure_mae_pp',-1),reverse=True),
        caveat='Same state means native lid coverage configuration, not pixel-identical images. Thresholds fixed before results.')
    save(OUT/'report.json',report)
    with (OUT/'predictions.jsonl').open('w') as f:
        for point in sorted(records,key=lambda p:(p['episode_index'],p['frame_index'])):f.write(json.dumps(point)+'\n')
    exported.sort(key=lambda r:r['episode_index'])
    (target/'metadata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exported))
    alias=VIEW/'viewer-data'/'FOREST-S01-Val-Latest';nextlink=alias.with_name(alias.name+'.next')
    assert not alias.exists() or alias.is_symlink()
    nextlink.symlink_to(target,target_is_directory=True);nextlink.replace(alias)
    namespath=VIEW/'val5-viewer/catalog_names.json';names=read(namespath)
    names[alias.name]='FOREST 全部47条验证轨迹 · 扩充120条脚本 · 最佳 step23695'
    save(namespath,names)
    save(OUT/'status.json',dict(phase='completed',published=True,checkpoint_step=23695,episodes=47,updated=time.time()))
    print(json.dumps({k:v for k,v in report.items() if k!='by_episode'}),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():save(OUT/'status.json',dict(phase='failed',error=repr(exc),updated=time.time()))
        raise
