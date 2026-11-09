"""Prepare disjoint S01 trajectory splits and causal Forest caches for fine-tuning."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

ROOT = Path('/cache/cvpzhr/setup/forest-s01-decline-20261106')
DATA = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline')
TASK = DATA/'State/01-cover_uncover_blocks'
CACHE = Path('/cache/cvpzhr/datasets/forest-s01-decline-5fps-20261106')
CHECKPOINT = '/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt'
OUT = '/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106'


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    CACHE.mkdir(exist_ok=True)
    assert not any(CACHE.iterdir()), 'Preparation requires an empty output directory'
    records = [json.loads(s) for s in (TASK/'metadata.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for row in records:
        ep = row['episode_index']
        if row['trajectory_source']=='vla':
            seed = row['actual_seed']
            category = ('fa' if '完整注意力' in row['model_name'] else 'rec') + ('_success' if row['task_success'] else '_failure')
        else:
            seed = read(TASK/row['evidence_path']/'result.json')['seed']
            category = 'script_recovery' if row['task_success'] else 'script_failure'
        row.update(scene_seed=seed,category=category)
        groups[seed].append(row)
    # Stratify whole seed groups. Mixed-outcome groups stay in train; homogeneous
    # groups can be held out as a unit, including repeated attempts at one seed.
    targets = dict(fa_success=4,rec_success=4,fa_failure=8,rec_failure=4,script_recovery=6,script_failure=2)
    rng = random.Random(42)
    val = []
    for category,count in targets.items():
        candidates = [g for g in groups.values() if all(r['category']==category and r['episode_index']!=83 for r in g)]
        rng.shuffle(candidates)
        choices={0:[]}
        for group in candidates:
            for size,selected in list(choices.items()):
                total=size+len(group)
                if total<=count and (total not in choices or len(selected)+1>len(choices[total])):
                    choices[total]=selected+[group]
        assert count in choices,(category,count)
        val.extend(r['episode_index'] for group in choices[count] for r in group)
    validation = set(val)
    for row in records:
        row['split']='validation' if row['episode_index'] in validation else 'train'
        row['cache_dir']=str(CACHE/row['split']/f"episode_{row['episode_index']:06d}")
        row['annotation_sha256']=sha(TASK/row['annotation_path'])
    train_seeds={r['scene_seed'] for r in records if r['split']=='train'}
    val_seeds={r['scene_seed'] for r in records if r['split']=='validation'}
    assert not train_seeds & val_seeds
    for key in ['video_sha256','action_fingerprint']:
        left={r[key] for r in records if r['split']=='train' and r.get(key)}
        right={r[key] for r in records if r['split']=='validation' and r.get(key)}
        assert not left & right, key
    config=read(Path(CHECKPOINT).parent/'config.json')
    for key in ['data_signature','validation_data_signature','preparation_hashes','best_validation_mae_pp','best_validation_step']:
        config.pop(key,None)
    positive=[r for r in records if r['split']=='train' and r['mixture_group']=='imprint_success']
    abnormal=[r for r in records if r['split']=='train' and r['mixture_group']=='abnormal']
    assert (len(positive),len(abnormal))==(36,74)
    # Exact trajectory visits 1:2: all train episodes once, one extra success.
    visits=positive+[rng.choice(positive)]+abnormal
    rng.shuffle(visits)
    config.update(initialize_from=CHECKPOINT,output_dir=OUT,epochs=3,max_steps=None,
        learning_rate=1e-5,batch_size=4,gradient_accumulation_steps=1,num_workers=4,
        log_every_steps=10,save_every_steps=200,save_every_epochs=1,keep_only_latest=False,
        sample_files=[str(Path(r['cache_dir'])/'samples.jsonl') for r in visits],
        validation_sample_files=[str(Path(r['cache_dir'])/'samples.jsonl') for r in records if r['split']=='validation'],
        validation_suite_by_episode={r['cache_dir']:'State' for r in records if r['split']=='validation'},
        validation_interval_epochs=.5,validation_batch_size=4,validation_num_workers=2)
    config['model']['gradient_checkpointing']=False
    save(ROOT/'config.json',config)
    index=dict(records=records,history_source=config['history_source'],model_path=config['model_path'],
        initial_checkpoint=CHECKPOINT,seed=42,source_dataset=str(DATA),
        train_trajectories=110,validation_trajectories=28,train_visits={'imprint_success':37,'abnormal':74},
        grouping='Same actual scene seed across models and scripts kept together; mixed-outcome seed groups in train',
        validation_scope='Unseen trajectories and seeds; shared task/color permutations, not unseen-task validation',
        validation_by_category=targets,pause_augmentation={'probability':.2,'segments':[1,3],'frames_at_5fps':[5,20],'validation':False})
    save(CACHE/'index.json',index)
    save(ROOT/'split.json',{k:v for k,v in index.items() if k!='history_source'})
    print(json.dumps({k:v for k,v in index.items() if k not in ['records','history_source']},ensure_ascii=False),flush=True)


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
        source_frame_labels_exact=True,causal_history_sample_checks=414,passed=True)
    save(ROOT/'cache-validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','validate']);p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='worker':worker(a.worker,a.workers)
    else:validate()
