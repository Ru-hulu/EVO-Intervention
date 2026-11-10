"""Rebuild every temporal window; hard-link immutable per-frame RGB/pixel data.

Robometer pixel caches contain independent preprocessed frames, not temporal
embeddings. Only window indices change. Old caches/results remain untouched.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from robometer_training.uniform import POLICY, prefix_indices

VIEWS = ['cam_high']*8+['cam_left_wrist','cam_right_wrist']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def link(source,target):
    source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(source,target)
    assert os.path.samefile(source,target)
    return str(target)


def episode(job):
    split,rows,old_root,new_root,model=job
    import torch
    from safetensors import safe_open
    from robometer_training.data import ProgressCollator, ObservationWindow, _episode_cache
    from robometer_training.model import load_processor
    torch.set_num_threads(1)
    root,old=Path(new_root),Path(old_root)
    eid=rows[0]['episode_id']
    assert [r['sample_index'] for r in rows]==list(range(len(rows)))
    assert all(r['episode_id']==eid and r['kind']=='original' and r['pt_policy']=='none' for r in rows)
    assert all(rows[i]['frame_index']<rows[i+1]['frame_index'] for i in range(len(rows)-1))
    images=[]
    for r in rows:
        assert r['input_views']==VIEWS and len(r['image'])==10
        assert r['observation_indices'][-3:]==[r['sample_index']]*3
        current=[]
        for path in r['image'][-3:]:
            rel=Path(path).relative_to(old);current.append(link(path,root/rel))
        images.append(current)
    pixel=link(rows[0]['pixel_cache'],root/Path(rows[0]['pixel_cache']).relative_to(old))
    with safe_open(pixel,framework='pt',device='cpu') as f:
        assert f.metadata()['input_contract']=='history7-main-current3-v1'
        for view in ['cam_high','cam_left_wrist','cam_right_wrist']:
            assert f.get_slice('pixels_'+view).get_shape()[0]==len(rows)
    window=ObservationWindow(8)
    prepared=[]
    for i,old_row in enumerate(rows):
        r=dict(old_row);time_slots=prefix_indices(i);past=time_slots[:-1]
        selected=[images[j][0] for j in past]+images[i]
        assert window.push(eid,images[i][0])[:-1]+images[i]==selected
        r.update(image=selected, history_sample_indices=past,
                 observation_indices=past+[i]*3, window_indices=past+[i]*3,
                 history_frame_indices=[rows[j]['frame_index'] for j in past],
                 history_source_frame_indices=[rows[j]['source_frame_index'] for j in past],
                 pixel_cache=pixel, temporal_sampling=POLICY,
                 temporal_indices=time_slots, padding='repeat_current_frame_on_right')
        assert time_slots[0]==0 and time_slots[-1]==i and max(time_slots)<=i
        for key in ['id','episode_id','frame_index','source_frame_index','pt','task','kind','pt_policy']:
            assert r[key]==old_row[key]
        prepared.append(r)
    collator=ProgressCollator(load_processor(model,min_pixels=4096,max_pixels=65536),4096)
    chosen=sorted({0,min(1,len(rows)-1),min(6,len(rows)-1),min(7,len(rows)-1),min(8,len(rows)-1),len(rows)//2,len(rows)-1})
    examples=[prepared[i] for i in chosen]
    cached=collator.encode(examples,supervised=True)
    online=collator.encode([{k:v for k,v in r.items() if k!='pixel_cache'} for r in examples],supervised=True)
    assert cached.keys()==online.keys()
    for key in cached:
        a,b=cached[key],online[key]
        if key=='pixel_values':a,b=a.to(torch.bfloat16),b.to(torch.bfloat16)
        assert torch.equal(a,b),(eid,key)
    _episode_cache.cache_clear()
    dest=root/'episodes'/split/(eid+'.jsonl');dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared));temp.replace(dest)
    return dict(episode_id=eid,split=split,samples=len(rows),episode_jsonl=str(dest),pixel_cache=pixel,
                pixel_bytes=Path(pixel).stat().st_size,frame_data_reused_by_hardlink=True,
                cached_online_examples=len(chosen),pt_unchanged=True,temporal_sampling=POLICY)


def build(config,workers):
    old=Path(config['source_cache']);out=Path(config['train_data']).parent
    if (out/'cache-ready.json').exists():
        receipt=json.loads((out/'cache-ready.json').read_text())
        assert receipt['temporal_sampling']==POLICY
        for split in ['train','validation']:assert sha(out/f'{split}.jsonl')==receipt['sha256'][split]
        return receipt
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((old/'manifest.json').read_text());jobs=[];source_hashes={};ids={}
    for split in ['train','validation']:
        src=old/f'{split}.jsonl';source_hashes[split]=sha(src)
        assert source_hashes[split]==manifest['splits'][split]['sha256']
        grouped=defaultdict(list)
        with src.open() as f:
            for line in f:
                r=json.loads(line);assert r['task_id']==config['task'];grouped[r['episode_id']].append(r)
        assert sum(map(len,grouped.values()))==config[f'expected_{split}_samples']
        ids[split]=set(grouped)
        jobs.extend((split,rows,str(old),str(out),config['model_path']) for _,rows in sorted(grouped.items()))
    assert not ids['train']&ids['validation']
    started=time.time();results=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(episode,job) for job in jobs]):
            results.append(future.result())
            write(out/'status.json',dict(phase='building',completed_episodes=len(results),total_episodes=len(jobs),updated_at=time.time()))
    hashes={}
    for split in ['train','validation']:
        with (out/f'{split}.jsonl').open('wb') as f:
            for info in sorted((r for r in results if r['split']==split),key=lambda r:r['episode_id']):
                f.write(Path(info['episode_jsonl']).read_bytes())
        hashes[split]=sha(out/f'{split}.jsonl')
        assert sha(old/f'{split}.jsonl')==source_hashes[split]
    receipt=dict(status='passed',temporal_sampling=POLICY,padding='repeat_current_frame_on_right',
                 source_cache=str(old),source_sha256=source_hashes,sha256=hashes,
                 episodes=len(results),samples=sum(r['samples'] for r in results),
                 all_pt_and_episode_splits_unchanged=True,all_windows_causal=True,
                 all_cached_online_inputs_equal=True,cached_online_examples=sum(r['cached_online_examples'] for r in results),
                 source_pixels_immutable_hardlinks=True,elapsed_seconds=time.time()-started)
    write(out/'manifest.json',dict(**receipt,sample_fps=5,window_size=8,images_per_sample=10,
                                 input_contract='history7-main-current3-v1',episode_records=results))
    write(out/'cache-ready.json',receipt);write(out/'status.json',dict(phase='completed',**receipt))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();print(json.dumps(build(json.loads(Path(args.config).read_text()),args.workers),indent=2))
