"""Selected R05 supplemental PT and frozen FoREST memory, on idle GPUs only."""
import argparse
import csv
import os
import time

from .common import *
from experiments.prm_five_groups_20261114.prepare_cache import progress_values


def labels():
    import numpy as np
    import torch
    from experiments.prm_five_groups_20261114.recover_pour_labels import (
        load_encoder, progress_from_deltas, REVISION, MODEL)
    from mem_lite.datasets.prepare_progress import extract_deltas
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    selection=read(ROOT/'selection.json'); source_checks(selection)
    missing=[r for r in selection['records'] if r['generate_missing_progress']]
    model,processor,digest=load_encoder('cuda')
    save(ROOT/'label-initialization.json',dict(vision_sha256=digest,model=str(MODEL),revision=REVISION,
         frozen=not any(p.requires_grad for p in model.parameters()),pid=os.getpid(),batch=16,max_pixels=262144))
    for row in missing:
        p=ROOT/'labels'/row['task_id']/(row['sample_id']+'.progress.csv')
        assert not p.exists(),f'Preserve prior labels: {p}'
        save(ROOT/'label-status.json',dict(status='running',key=row['key'],updated=time.time(),pid=os.getpid()))
        raw,info=extract_deltas(row['videos'][0],model,processor,16,262144)
        assert info['frame_count']==row['frames'] and info['fps']==row['fps']
        annotation=read(row['annotation_path'])
        effective,pt,events=progress_from_deltas(raw,annotation)
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w') as f:
            w=csv.writer(f);w.writerow(['frame_index','delta','effective_delta','pt'])
            w.writerows((i,float(a),float(b),float(c)) for i,(a,b,c) in enumerate(zip(raw,effective,pt)))
        # Verify saved precision and normalization independently through CSV parsing.
        with p.open() as f: restored=list(csv.DictReader(f))
        assert np.array_equal(np.array([float(x['pt']) for x in restored]),pt)
        assert pt[0]==0 and pt[-1]==1
        save(p.with_suffix('.json'),dict(video_sha256=row['video_hashes'][0],
            annotation_sha256=row['annotation_sha256'],vision_sha256=digest,revision=REVISION,
            csv_sha256=sha(p),events=events,frames=len(pt),original_source_unchanged=True,
            policy='first-effective-pour-and-associated-putdown-cumulative-visual-change-v1'))
        row.update(progress_path=str(p),progress_sha256=sha(p))
    for row in selection['records']:
        values=progress_values(row); assert values[0]==0
        obs=rows(epdir(row)/'observations.jsonl')
        for r in obs: r['pt']=values[r['frame_index']]
        write_rows(epdir(row)/'samples.jsonl',obs)
    source_checks(selection)
    save(ROOT/'resolved-selection.json',selection)
    save(ROOT/'label-status.json',dict(status='completed',supplemented_episodes=len(missing),
         original_labels_retained=10-len(missing),updated=time.time()))


def forest(worker):
    import torch
    from PIL import Image
    from mem_lite.datasets.prepare_stream import build_episode
    from mem_lite.datasets.views import VIEW_POLICY
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import load_vision_encoder,QwenFrameEncoder
    from experiments.prm_five_groups_20261114.prepare_cache import VISION_SHA
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    selection=read(ROOT/'resolved-selection.json');source_checks(selection)
    visual,processor,digest=load_vision_encoder('/cache/cvpzhr/models/Qwen3.5-4B','cuda')
    assert digest==VISION_SHA
    memory=read(ROOT/'code/mem_lite/configs/streamforest_clip_depth.json')
    stream=StreamMemory(QwenFrameEncoder(visual),processor,memory,65536,4096)
    origin=stream.bind_history_source(dict(method='streamforest-v1',vision_sha256=digest,
        memory_config=memory,sample_fps=5.,include_episode_last_frame=True,
        max_pixels=65536,min_pixels=4096,view_policy=VIEW_POLICY))
    save(ROOT/f'forest-initialization-{worker}.json',dict(history_source=origin,pid=os.getpid(),
        frozen=not any(p.requires_grad for p in visual.parameters())))
    completed=[]
    for row in selection['records'][worker::2]:
        target=OUT/'forest'/'train'/row['task_id']/row['sample_id'];observed=rows(epdir(row)/'samples.jsonl')
        def observations():
            for i,r in enumerate(observed):
                if i%50==0: save(ROOT/f'forest-worker-{worker}.json',dict(status='running',key=row['key'],
                    observations=i,completed_episodes=len(completed),pid=os.getpid(),updated=time.time()))
                images=[]
                for p in r['images']:
                    with Image.open(p) as image:images.append(image.convert('RGB'))
                yield dict(frame_index=r['frame_index'],pt=r['pt'],images=images)
        summary=build_episode(observations(),stream,target,row['task_prompt'],row['key'],origin,source=row)
        actual=rows(target/'samples.jsonl')
        assert len(actual)==len(observed)==summary['num_samples']
        for i,(a,b) in enumerate(zip(actual,observed)):
            assert a['frame_index']==b['frame_index'] and a['pt']==b['pt']
            for v,p in enumerate(b['images']):assert sha(target/a['images'][VIEWS[v]])==sha(p)
            if i==0:assert a['history'] is None
            elif a['history']:
                h=torch.load(target/a['history'],map_location='cpu',weights_only=True)
                assert h['through_frame']==observed[i-1]['frame_index']
                assert h['tokens'].ndim==2 and torch.isfinite(h['tokens']).all()
        receipt=dict(key=row['key'],samples=len(actual),samples_sha256=sha(target/'samples.jsonl'),
                     source_pt_unchanged=True,all_history_causal=True,all_png_sha_equal=True)
        save(target/'verification.json',receipt);completed.append(receipt)
    save(ROOT/f'forest-worker-{worker}.json',dict(status='completed',records=completed,pid=os.getpid(),updated=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['labels','forest']);p.add_argument('--worker',type=int)
    a=p.parse_args();labels() if a.action=='labels' else forest(a.worker)
