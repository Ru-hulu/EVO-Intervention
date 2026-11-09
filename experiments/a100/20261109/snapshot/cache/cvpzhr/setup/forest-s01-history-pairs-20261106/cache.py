"""Build only missing v4 caches and verify all labels against the frozen split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

ROOT = Path('/cache/cvpzhr/setup/forest-s01-history-pairs-20261106')
DATA = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v4-history-pairs')
TASK = DATA/'State/01-cover_uncover_blocks'
CACHE = Path('/cache/cvpzhr/datasets/forest-s01-history-pairs-5fps-20261106')


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def worker(worker_id,workers):
    import torch
    from mem_lite.datasets.prepare_stream import iter_observations, build_episode
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder
    torch.set_num_threads(2)
    index=read(CACHE/'index.json');origin=index['history_source']
    visual,processor,vision_sha=load_vision_encoder(index['model_path'],'cuda')
    assert vision_sha==origin['vision_sha256']
    stream=StreamMemory(QwenFrameEncoder(visual),processor,origin['memory_config'],origin['max_pixels'],origin['min_pixels'])
    records=index['records'][worker_id::workers]
    status=ROOT/f'cache-worker-{worker_id}.json'
    for i,row in enumerate(records):
        start=time.time();ep=row['episode_index'];out=Path(row['cache_dir'])
        if (out/'summary.json').exists():
            continue
        assert not out.exists(), 'Incomplete cache: inspect before retry'
        out.parent.mkdir(exist_ok=True)
        curve=read(TASK/row['annotation_path'])['progress']['dense_progress']
        videos=[TASK/row['video_views'][v]['video_path'] for v in ['main_view','left_view','right_view']]
        rng=random.Random(42+ep)
        # Pick pause anchors before decoding; all views share the same schedule.
        count=int((len(curve)-1)*5/row['fps'])+2
        pauses=Counter()
        if row['split']=='train' and rng.random()<.2:
            for _ in range(rng.randint(1,3)):
                pauses[rng.randrange(max(1,count-1))]+=rng.randint(5,20)
        mapping=[]
        def observations():
            inserted=0
            for number,item in enumerate(iter_observations(videos,curve,sample_fps=5,end_frame=len(curve))):
                source=item['frame_index']
                for repeat in range(1+pauses[number]):
                    # Keep strictly increasing virtual timestamps during pauses.
                    if repeat:
                        inserted+=1
                    frame=source+round(inserted*row['fps']/5)
                    mapping.append(dict(frame_index=frame,source_frame=source,pause_copy=repeat>0))
                    yield dict(frame_index=frame,images=item['images'],pt=curve[source])
        result=build_episode(observations(),stream,out,row['task_prompt'],row['sample_id'],origin,
            source=dict(dataset=str(DATA),episode=ep,annotation_sha256=row['annotation_sha256'],split=row['split']))
        save(out/'source_frames.json',mapping)
        save(status,dict(completed=i+1,total=len(records),episode=ep,samples=result['num_samples'],seconds=time.time()-start))
        print(json.dumps(read(status)),flush=True)


def validate():
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset
    index=read(CACHE/'index.json');counts=Counter();pauses=0;frames=0
    for row in index['records']:
        out=Path(row['cache_dir']);summary=read(out/'summary.json')
        samples=[json.loads(s) for s in (out/'samples.jsonl').read_text().splitlines()]
        mapping=read(out/'source_frames.json');curve=read(TASK/row['annotation_path'])['progress']['dense_progress']
        assert len(mapping)==len(samples)==summary['num_samples']
        previous=None
        for sample,source in zip(samples,mapping):
            assert sample['frame_index']==source['frame_index']
            assert sample['pt']==curve[source['source_frame']]
            assert all((out/f).is_file() for f in sample['images'].values())
            if previous is None:assert sample['history'] is None
            else:assert sample['frame_index']>previous and (out/sample['history']).is_file()
            previous=sample['frame_index']
        assert mapping[-1]['source_frame']==len(curve)-1
        if row['split']=='validation':assert not any(m['pause_copy'] for m in mapping)
        pauses+=sum(m['pause_copy'] for m in mapping);frames+=len(samples);counts[row['split']]+=1
        ds=M3Dataset([out/'samples.jsonl'])
        for i in {0,len(ds)//2,len(ds)-1}:ds[i]
    config=read(ROOT/'config.json')
    train=M3Dataset(config['sample_files'],load_history=False)
    val=M3Dataset(config['validation_sample_files'],load_history=False)
    assert train.history_source==val.history_source==index['history_source']
    report=dict(trajectories=dict(counts),frames=frames,pause_frames=pauses,train_samples=len(train),validation_samples=len(val),
        source_frame_labels_exact=True,causal_history_sample_checks=3*len(index['records']),passed=True)
    save(ROOT/'cache-validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','validate']);p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.mode=='worker':worker(a.worker,a.workers)
    else:validate()
