"""Prepare v4 fine-tuning with unchanged old caches and validation assignments."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

BASE=Path('/cache/cvpzhr/setup/forest-s01-critical-sampling-20261106')
WORK=Path('/cache/cvpzhr/setup/forest-s01-history-pairs-20261106')
CACHE=Path('/cache/cvpzhr/datasets/forest-s01-history-pairs-5fps-20261106')
OUT=Path('/cache/cvpzhr/outputs/forest-s01-history-pairs-20261106')
DATA=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v4-history-pairs')
TASK=DATA/'State/01-cover_uncover_blocks'
REGISTRY=Path('/cache/cvpzhr/experiments/forest-s01-decline')
ARCHIVE=REGISTRY/'03-history-pairs-20261106'
GROUPS=['wrong_approach','failure_transition','post_failure_motion','recovery']


def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def setup():
    assert not CACHE.exists() and not OUT.exists() and not (WORK/'config.json').exists()
    assert read(DATA/'verification/history-pairs.json')['complete']
    cfg=read(BASE/'config.json');split=read(DATA/'split.json');old=read(BASE/'split.json')
    assert split['records'][:138]==old['records']
    assert Counter(r['split'] for r in split['records'])=={'train':122,'validation':28}
    CACHE.mkdir();(CACHE/'train').mkdir()
    for row in split['records']:
        assert sha(TASK/row['annotation_path'])==row['annotation_sha256']
        if row['episode_index']>=138:
            assert row['split']=='train' and row['category'] in ['script_success','script_history_failure']
            row['cache_dir']=str(CACHE/'train'/f"episode_{row['episode_index']:06d}")
            cfg['sample_files'].append(str(Path(row['cache_dir'])/'samples.jsonl'))
    assert len(set(cfg['sample_files']))==122 and len(cfg['sample_files'])==123
    assert cfg['validation_sample_files']==read(BASE/'config.json')['validation_sample_files']
    cfg['output_dir']=str(OUT)
    cfg['validation_event_groups']=str(WORK/'validation-event-groups.json')
    cfg.pop('weighted_sampling')
    assert cfg['model']['gradient_checkpointing'] is False and cfg['batch_size']==2
    assert cfg['initialize_from']=='/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt'
    shutil.copytree(BASE/'code',WORK/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(BASE/'validation-event-groups.json',WORK/'validation-event-groups.json')
    save(WORK/'config.json',cfg);save(WORK/'split.json',split)
    save(CACHE/'index.json',dict(split,history_source=cfg['history_source']))
    save(WORK/'status.json',dict(phase='cache_preparation',new_trajectories=12,reused_trajectories=138))
    print('Prepared 12 new caches, preserving 138 old caches and exact validation files.',flush=True)


def sampling():
    import torch
    sys.path.insert(0,str(WORK/'code'))
    from mem_lite.datasets.m3_dataset import M3Dataset
    from mem_lite.weighted_sampler import DistributedWeightedSampler
    cfg=read(WORK/'config.json');split=read(WORK/'split.json')
    rows={r['cache_dir']:r for r in split['records']};records=[]
    for filename in cfg['sample_files']:
        cache=Path(filename).parent;row=rows[str(cache)];assert row['split']=='train'
        ann=read(TASK/row['annotation_path'])['progress']
        mapping=read(cache/'source_frames.json')
        samples=[json.loads(l) for l in Path(filename).read_text().splitlines()]
        assert len(mapping)==len(samples)
        deltas=read(TASK/'deltas'/f"episode_{row['episode_index']:06d}.json")['deltas']
        window=max(1,round(row['fps']/5))
        movement=[sum(deltas[max(0,m['source_frame']-window+1):m['source_frame']+1])/window for m in mapping]
        positive=sorted(v for m,v in zip(mapping,movement) if not m['pause_copy'] and v>1e-6)
        threshold=positive[len(positive)//2] if positive else float('inf')
        intervals=ann.get('penalty_intervals',[])
        if row['category']=='script_history_failure':
            boundary=read(TASK/row['subtask_boundary_path'])
            intervals=[dict(native_failure_frame=boundary['failure_frame'],
                error_range=[boundary['error_frame'],boundary['failure_frame']],recovery_range=None)]
        for sample,m,motion in zip(samples,mapping,movement):
            f=m['source_frame'];assert sample['frame_index']==m['frame_index']
            assert abs(sample['pt']-ann['dense_progress'][f])<1e-6
            normal=row['category'].endswith('_success')
            group='normal' if normal else 'other_abnormal'
            if not normal and not m['pause_copy']:
                for interval in intervals:
                    fail,error,recovery=interval['native_failure_frame'],interval['error_range'],interval['recovery_range']
                    if fail is not None and abs(f-fail)<=round(row['fps']):group='failure_transition';break
                    if error[0]<=f<error[1]:group='wrong_approach';break
                    if recovery is not None and recovery[0]<=f<=recovery[1]:group='recovery';break
                    if fail is not None and f>fail+round(row['fps']) and motion>=threshold:group='post_failure_motion';break
            records.append(dict(index=len(records),episode=row['episode_index'],category=row['category'],
                source_frame=f,frame_index=sample['frame_index'],pause_copy=m['pause_copy'],group=group))
    n=len(records);counts=Counter(r['group'] for r in records);abnormal=n-counts['normal']
    prior_records=read(BASE/'sampling-index.json')
    assert records[:len(prior_records)]==prior_records, 'Existing sample grouping changed'
    assert all(counts[g]>0 for g in GROUPS)
    for ep in range(139,150,2):
        found=Counter(r['group'] for r in records if r['episode']==ep)
        assert all(found[g]>0 for g in ['wrong_approach','failure_transition','post_failure_motion']), (ep,found)
    weights=torch.tensor([1/n if r['group']=='normal' else .5/n for r in records],dtype=torch.float64)
    for r in records:
        if r['group'] in GROUPS:weights[r['index']]+=.5*(abnormal/n)/len(GROUPS)/counts[r['group']]
    assert abs(float(weights.sum())-1)<1e-12
    ds=M3Dataset(cfg['sample_files'],cfg['model']['history_dim'],load_history=False)
    assert len(ds)==n
    torch.save(dict(weights=weights,data_signature=ds.signature),WORK/'sampling-weights.pt')
    cfg['weighted_sampling']=dict(weights_file=str(WORK/'sampling-weights.pt'),weights_sha256=sha(WORK/'sampling-weights.pt'),
        policy='abnormal_half_uniform_half_critical_v1',normal_probability=counts['normal']/n,
        abnormal_probability=abnormal/n,critical_groups=GROUPS,replacement=True)
    samplers=[DistributedWeightedSampler(ds,cfg['weighted_sampling'],rank,4,42) for rank in range(4)]
    shards=[list(s) for s in samplers];merged=[shards[i%4][i//4] for i in range(4*len(shards[0]))]
    reference=torch.multinomial(weights,n,replacement=True,generator=torch.Generator().manual_seed(42)).tolist()
    assert merged[:n]==reference
    assert list(samplers[0])==shards[0]
    samplers[0].set_epoch(1);assert list(samplers[0])!=shards[0]
    updates=math.ceil(len(shards[0])/cfg['batch_size'])
    audit=dict(num_samples=n,unique_train_trajectories=122,val_trajectories=28,source_counts=dict(counts),
        expected_probabilities={g:sum(float(weights[r['index']]) for r in records if r['group']==g) for g in counts},
        new_samples=sum(r['episode']>=138 for r in records),new_failure_critical_counts=dict(Counter(r['group'] for r in records if r['category']=='script_history_failure')),
        updates_per_epoch=updates,total_updates=updates*cfg['epochs'],val_unchanged=True,
        rule='Same abnormal-half-uniform/half-critical policy as run 02. New native failure intervals enter existing groups. No added loss.',
        source_labels_exact=True,distributed_draw_verified=True)
    save(WORK/'config.json',cfg);save(WORK/'sampling-index.json',records);save(WORK/'sampling-audit.json',audit)
    hashes={str(p.relative_to(WORK/'code')):sha(p) for p in (WORK/'code').rglob('*.py')}
    assert hashes==read(BASE/'code-sha256.json'), 'Training code changed'
    save(WORK/'code-sha256.json',hashes)
    ARCHIVE.mkdir()
    for name in ['config.json','split.json','sampling-audit.json','sampling-index.json','code-sha256.json','cache-validation.json']:
        shutil.copy2(WORK/name,ARCHIVE/name)
    save(ARCHIVE/'manifest.json',dict(run_id=ARCHIVE.name,status='prepared',setup_dir=str(WORK),output_dir=str(OUT),
        initialize_from=cfg['initialize_from'],initialize_sha256=sha(Path(cfg['initialize_from'])),dataset=str(DATA),
        sampling=cfg['weighted_sampling'],baseline=str(REGISTRY/'02-critical-sampling-20261106'),
        comparison_note='Same base initialization, training code, loss, LR, global batch 8, sampling algorithm and three epochs. Dataset adds 12 training trajectories; epoch size and total updates increase. Validation unchanged.'))
    registry=read(REGISTRY/'index.json');assert ARCHIVE.name not in [r['id'] for r in registry['runs']]
    registry['runs'].append(dict(id=ARCHIVE.name,manifest=str(ARCHIVE/'manifest.json')))
    save(REGISTRY/'index.json',registry)
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'setup':setup,'sampling':sampling}[sys.argv[1]]()
