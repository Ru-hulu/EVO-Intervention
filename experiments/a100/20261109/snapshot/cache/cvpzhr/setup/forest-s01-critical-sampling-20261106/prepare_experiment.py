"""Freeze the baseline and create a training-only critical-sampling experiment."""
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

BASE = Path('/cache/cvpzhr/setup/forest-s01-decline-20261106')
OLD = Path('/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106-4gpu')
WORK = Path('/cache/cvpzhr/setup/forest-s01-critical-sampling-20261106')
OUT = Path('/cache/cvpzhr/outputs/forest-s01-critical-sampling-20261106')
REGISTRY = Path('/cache/cvpzhr/experiments/forest-s01-decline')
DATA = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline/State/01-cover_uncover_blocks')
GROUPS = ['wrong_approach', 'failure_transition', 'post_failure_motion', 'recovery']


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def main():
    assert not OUT.exists() and not (WORK/'config.json').exists(), 'Do not overwrite an experiment'
    cfg=json.loads((BASE/'config.json').read_text())
    rows={r['cache_dir']:r for r in json.loads((BASE/'split.json').read_text())['records']}
    records=[]
    for filename in cfg['sample_files']:
        cache=Path(filename).parent;row=rows[str(cache)];assert row['split']=='train'
        ann=json.loads((DATA/row['annotation_path']).read_text())['progress']
        mapping=json.loads((cache/'source_frames.json').read_text())
        samples=[json.loads(l) for l in Path(filename).read_text().splitlines()]
        assert len(mapping)==len(samples)
        deltas=json.loads((DATA/'deltas'/f"episode_{row['episode_index']:06d}.json").read_text())['deltas']
        window=max(1,round(row['fps']/5))
        movement=[sum(deltas[max(0,m['source_frame']-window+1):m['source_frame']+1])/window for m in mapping]
        positive=sorted(v for m,v in zip(mapping,movement) if not m['pause_copy'] and v>1e-6)
        threshold=positive[len(positive)//2] if positive else float('inf')
        intervals=ann.get('penalty_intervals',[])
        for sample,mapping_row,motion in zip(samples,mapping,movement):
            assert sample['frame_index']==mapping_row['frame_index']
            f=mapping_row['source_frame'];assert abs(sample['pt']-ann['dense_progress'][f])<1e-6
            normal=row['category'].endswith('_success')
            group='normal' if normal else 'other_abnormal'
            if not normal and not mapping_row['pause_copy']:
                for interval in intervals:
                    fail=interval['native_failure_frame'];error=interval['error_range'];recovery=interval['recovery_range']
                    if fail is not None and abs(f-fail)<=round(row['fps']): group='failure_transition';break
                    if error[0]<=f<error[1]:group='wrong_approach';break
                    if recovery is not None and recovery[0]<=f<=recovery[1]:group='recovery';break
                    if fail is not None and f>fail+round(row['fps']) and motion>=threshold:
                        group='post_failure_motion';break
            records.append(dict(index=len(records),episode=row['episode_index'],category=row['category'],
                source_frame=f,frame_index=sample['frame_index'],pause_copy=mapping_row['pause_copy'],group=group))
    n=len(records);counts=collections.Counter(r['group'] for r in records)
    assert n==35258 and counts['normal']==9253 and all(counts[g]>0 for g in GROUPS)
    abnormal=n-counts['normal']
    weights=torch.tensor([1/n if r['group']=='normal' else .5/n for r in records],dtype=torch.float64)
    for r in records:
        if r['group'] in GROUPS:weights[r['index']]+=.5*(abnormal/n)/len(GROUPS)/counts[r['group']]
    assert abs(float(weights.sum())-1)<1e-12
    code=WORK/'code';shutil.copytree(BASE.with_name(BASE.name+'-4gpu')/'code',code,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(WORK/'weighted_sampler.py',code/'mem_lite/weighted_sampler.py')
    p=code/'mem_lite/train_stage3.py';s=p.read_text()
    s=s.replace('from mem_lite.periodic_validation import','from mem_lite.weighted_sampler import DistributedWeightedSampler\nfrom mem_lite.periodic_validation import')
    s=s.replace('    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=config["seed"], shuffle=True)',
        '    sampler = (DistributedWeightedSampler(dataset, config["weighted_sampling"], rank, world_size, config["seed"])\n'
        '               if config.get("weighted_sampling") else\n'
        '               DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=config["seed"], shuffle=True))')
    s=s.replace('            if batch_index < next_batch:\n                started = perf_counter()\n                continue\n','')
    p.write_text(s)
    sys.path.insert(0,str(code))
    from mem_lite.datasets.m3_dataset import M3Dataset
    ds=M3Dataset(cfg['sample_files'],cfg['model']['history_dim'])
    assert len(ds)==n
    torch.save(dict(weights=weights,data_signature=ds.signature),WORK/'sampling-weights.pt')
    save(WORK/'sampling-index.json',records)
    cfg.update(output_dir=str(OUT),batch_size=2,num_workers=2,epochs=3,max_steps=None)
    cfg['weighted_sampling']=dict(weights_file=str(WORK/'sampling-weights.pt'),weights_sha256=sha(WORK/'sampling-weights.pt'),
        policy='abnormal_half_uniform_half_critical_v1',normal_probability=counts['normal']/n,
        abnormal_probability=abnormal/n,critical_groups=GROUPS,replacement=True)
    save(WORK/'config.json',cfg);shutil.copyfile(BASE/'split.json',WORK/'split.json')
    probabilities={g:sum(float(weights[r['index']]) for r in records if r['group']==g) for g in counts}
    # Verify deterministic draws, rank partitioning, epoch changes and loss-masked tail.
    from mem_lite.weighted_sampler import DistributedWeightedSampler
    a=[DistributedWeightedSampler(ds,cfg['weighted_sampling'],r,4,42) for r in range(4)]
    shards=[list(x) for x in a]
    merged=[shards[i%4][i//4] for i in range(len(shards[0])*4)]
    reference=torch.multinomial(weights,n,replacement=True,generator=torch.Generator().manual_seed(42)).tolist()
    assert merged[:n]==reference and len(merged)==35260 and len(shards[0])==8815
    assert list(a[0])==shards[0];a[0].set_epoch(1);assert list(a[0])!=shards[0]
    drawn=collections.Counter(records[i]['group'] for i in reference)
    audit=dict(num_samples=n,unique_train_trajectories=110,val_trajectories=28,source_counts=dict(counts),
        expected_probabilities=probabilities,epoch0_draw_counts=dict(drawn),data_signature=ds.signature,
        val_unchanged=cfg['validation_sample_files']==json.loads((BASE/'config.json').read_text())['validation_sample_files'],
        updates_per_epoch=4408,total_updates=13224,tests='all four shards reconstruct identical seeded global draw; padding excluded by train loss mask; epoch seed changes; all targets match v3 source labels',
        rule='Normal frame mass unchanged. Abnormal mass: 50% original uniform, 50% equally split across four critical groups. Transition +/-1s. Post-failure motion above per-trajectory median positive 0.2s visual delta. Pause copies only in original-distribution component. No val predictions used.')
    save(WORK/'sampling-audit.json',audit)
    save(WORK/'code-sha256.json',{str(p.relative_to(code)):sha(p) for p in sorted(code.rglob('*.py'))})
    # Archive this completed task's baseline without moving or modifying its originals.
    baseline=REGISTRY/'01-uniform-20261106';baseline.mkdir(parents=True,exist_ok=True)
    for f in list(OLD.glob('*.pt'))+[OLD/'train.jsonl',OLD/'validation.jsonl',OLD/'config.json',BASE/'split.json']:
        dst=baseline/f.name
        if not dst.exists():
            if f.suffix=='.pt':os.link(f,dst)
            else:shutil.copyfile(f,dst)
    baseline_report=dict(run_id='01-uniform-20261106',status='completed',output_dir=str(OLD),
        code_dirs=[str(BASE/'code'),str(BASE.with_name(BASE.name+'-4gpu')/'code')],
        final_validation=json.loads((OLD/'validation.jsonl').read_text().splitlines()[-1]),
        evaluations=['/cache/cvpzhr/evaluations/forest-s01-val5-step13224-20261106','/cache/cvpzhr/evaluations/forest-s01-val10-step13224-20261106'],
        artifacts={p.name:sha(p) for p in baseline.iterdir() if p.is_file() and p.name!='manifest.json'})
    save(baseline/'manifest.json',baseline_report)
    run=REGISTRY/'02-critical-sampling-20261106';run.mkdir(exist_ok=True)
    for f in ['config.json','split.json','sampling-audit.json','sampling-index.json','code-sha256.json']:
        shutil.copyfile(WORK/f,run/f)
    save(run/'manifest.json',dict(run_id=run.name,status='prepared',setup_dir=str(WORK),output_dir=str(OUT),
        initialize_from=cfg['initialize_from'],initialize_sha256=sha(Path(cfg['initialize_from'])),
        dataset=str(DATA.parent.parent),sampling=cfg['weighted_sampling'],baseline=str(baseline),
        comparison_note='Same initialization, labels, split, global batch, LR and update budget. Baseline used 2 GPUs then 4; this run uses 4 throughout, so RNG allocation and backend details are not bitwise controlled.'))
    save(REGISTRY/'index.json',dict(runs=[dict(id=baseline.name,manifest=str(baseline/'manifest.json')),dict(id=run.name,manifest=str(run/'manifest.json'))]))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
