"""Create an isolated hard native-failure loss ablation from the completed v4 run."""
import hashlib
import json
import math
from pathlib import Path
import shutil

BASE=Path('/cache/cvpzhr/setup/forest-s01-history-pairs-20261106')
WORK=Path('/cache/cvpzhr/setup/forest-s01-hard-failure5x-20261107')
OUT=Path('/cache/cvpzhr/outputs/forest-s01-hard-failure5x-20261107')
REG=Path('/cache/cvpzhr/experiments/forest-s01-decline')
ARCHIVE=REG/'05-hard-failure5x-20261107'


def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def replace_once(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new)


def main():
    assert not OUT.exists() and not (WORK/'config.json').exists()
    cfg=read(BASE/'config.json')
    assert cfg['batch_size']==2 and cfg['gradient_accumulation_steps']==1
    cfg.update(output_dir=str(OUT),batch_size=4,hard_failure_loss_weight=5.,hard_failure_threshold=.2)
    assert cfg['model']['gradient_checkpointing'] is False
    assert cfg['initialize_from']=='/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt'
    shutil.copytree(BASE/'code',WORK/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(WORK/'hard_failure_loss.py',WORK/'code/mem_lite/hard_failure_loss.py')
    trainer=WORK/'code/mem_lite/train_stage3.py';s=trainer.read_text()
    s=replace_once(s,'from mem_lite.weighted_sampler import DistributedWeightedSampler',
        'from mem_lite.weighted_sampler import DistributedWeightedSampler\nfrom mem_lite.hard_failure_loss import FailureCollator, hard_failure_weights')
    s=replace_once(s,'    return config\n',
        '    if not math.isfinite(config.get("hard_failure_loss_weight", 1.)) or config.get("hard_failure_loss_weight", 1.) < 1:\n'
        '        raise ValueError("hard_failure_loss_weight must be finite and >= 1")\n    return config\n')
    s=replace_once(s,'totals = torch.zeros(5, device=device, dtype=torch.float64)',
        'totals = torch.zeros(8, device=device, dtype=torch.float64)')
    s=replace_once(s,'                loss_sum = (per_sample_loss * valid).sum()',
        '                target_weights, hard_mask = hard_failure_weights(target_pt, result["pt"], batch["native_failure"].to(device), config["hard_failure_loss_weight"], config["hard_failure_threshold"])\n'
        '                loss_sum = (per_sample_loss * target_weights * valid).sum()')
    s=replace_once(s,'            totals[4] += error.sum()',
        '            totals[4] += error.sum()\n'
        '            totals[5] += (per_sample_loss.detach().double() * valid).sum()\n'
        '            zero_valid = valid & hard_mask\n'
        '            totals[6] += zero_valid.sum()\n'
        '            totals[7] += (per_sample_loss.detach().double() * zero_valid).sum()')
    s=replace_once(s,'                    "pt_mae_pp": (100 * totals[2] / totals[1]).item(),',
        '                    "loss_unweighted": (totals[5] / totals[1]).item(),\n'
        '                    "hard_failure_samples": int(totals[6].item()),\n'
        '                    "hard_failure_ce": (totals[7] / totals[6]).item() if totals[6] > 0 else None,\n'
        '                    "hard_failure_loss_weight": config.get("hard_failure_loss_weight", 1.),\n'
        '                    "pt_mae_pp": (100 * totals[2] / totals[1]).item(),')
    s=replace_once(s, 'collate_fn=Stage3Collator(processor, dataset.history_dim, config["max_sequence_length"],\n                                 config["history_capacity"]),',
        'collate_fn=FailureCollator(Stage3Collator(processor, dataset.history_dim, config["max_sequence_length"],\n                                 config["history_capacity"]), json.loads(Path(config["native_failure_frames"]).read_text())),')
    trainer.write_text(s)
    # No modifications to model inference, soft targets, or validation reduction.
    for rel in ['mem_lite/models/m3_predictor.py','mem_lite/periodic_validation.py','mem_lite/weighted_sampler.py']:
        assert sha(WORK/'code'/rel)==sha(BASE/'code'/rel)
    for name in ['split.json','sampling-audit.json','sampling-index.json','sampling-weights.pt','validation-event-groups.json','cache-validation.json']:
        shutil.copy2(BASE/name,WORK/name)
    cfg['weighted_sampling']['weights_file']=str(WORK/'sampling-weights.pt')
    cfg['validation_event_groups']=str(WORK/'validation-event-groups.json')
    assert sha(WORK/'sampling-weights.pt')==cfg['weighted_sampling']['weights_sha256']
    assert math.ceil(math.ceil(read(WORK/'sampling-audit.json')['num_samples']/2)/4)*3==14604
    data=Path(read(BASE/'split.json')['source_dataset'])/'State/01-cover_uncover_blocks'
    failure_frames={};counts={'eligible':0,'initial_or_other_zero':0,'nonzero':0};episodes=[]
    for row in read(BASE/'split.json')['records']:
        if row['split']!='train':continue
        cache=Path(row['cache_dir']);mapping=read(cache/'source_frames.json')
        states=[json.loads(line) for line in (data/row['states_path']).read_text().splitlines()]
        failure=next((i for i,state in enumerate(states) if state['fail_flag']),None)
        if failure is not None:
            assert all(state['fail_flag'] for state in states[failure:])
        if row['task_success']:assert failure is None
        curve=read(data/row['annotation_path'])['progress']['dense_progress']
        flags={}
        for m in mapping:
            f=m['source_frame'];failed=failure is not None and f>=failure
            if failed:assert curve[f]==0
            flags[str(m['frame_index'])]=failed
            counts['eligible' if failed else 'initial_or_other_zero' if curve[f]==0 else 'nonzero']+=1
        failure_frames[str(cache.resolve())]=flags
        episodes.append(dict(episode=row['episode_index'],failure_frame=failure,eligible=sum(flags.values())))
    assert len(failure_frames)==122 and counts['eligible']>0 and counts['initial_or_other_zero']>0
    save(WORK/'native-failure-frames.json',failure_frames)
    save(WORK/'failure-mask-audit.json',dict(counts=counts,episodes=episodes,train_only=True,
        source='first native fail_flag=True, mapped through source_frames including pause copies',threshold=.2,weight=5))
    cfg['native_failure_frames']=str(WORK/'native-failure-frames.json')
    cfg['native_failure_frames_sha256']=sha(WORK/'native-failure-frames.json')
    save(WORK/'config.json',cfg)
    save(WORK/'code-sha256.json',{str(p.relative_to(WORK/'code')):sha(p) for p in (WORK/'code').rglob('*.py')})
    ARCHIVE.mkdir()
    for name in ['config.json','split.json','sampling-audit.json','code-sha256.json','cache-validation.json','failure-mask-audit.json']:
        shutil.copy2(WORK/name,ARCHIVE/name)
    save(ARCHIVE/'manifest.json',dict(run_id=ARCHIVE.name,status='prepared',setup_dir=str(WORK),output_dir=str(OUT),
        baseline=str(REG/'03-history-pairs-20261106'),initialize_from=cfg['initialize_from'],
        initialize_sha256=sha(Path(cfg['initialize_from'])),hard_failure_loss_weight=5,
        loss='mean(valid_i * (5 if native_failed and GT_i==0 and detached_pred_i>0.2 else 1) * soft-bin CE_i); denominator is valid sample count',
        validation='unchanged, unweighted CE and errors',gpus=[2,3],global_batch_size=8,total_steps=14604,
        comparison_note='Same v4 data, caches, initialization, sampler, global batch 8 (now two GPUs x4 due to other jobs; per-rank RNG differs), LR and three epochs as run 03. Only native-failed GT-zero predictions above 20% get extra CE. Initial zero, normal success, recovery and easy failures retain base CE.'))
    registry=read(REG/'index.json');assert ARCHIVE.name not in [r['id'] for r in registry['runs']]
    registry['runs'].append(dict(id=ARCHIVE.name,manifest=str(ARCHIVE/'manifest.json')));save(REG/'index.json',registry)
    print('Prepared hard native-failure x5: GPUs 2,3, global batch 8, 14604 steps, unweighted validation.',flush=True)


if __name__=='__main__':main()
