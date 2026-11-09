"""Create an isolated zero-GT loss ablation from the completed v4 run."""
import hashlib
import json
import math
from pathlib import Path
import shutil

BASE=Path('/cache/cvpzhr/setup/forest-s01-history-pairs-20261106')
WORK=Path('/cache/cvpzhr/setup/forest-s01-zero-gt5x-20261107')
OUT=Path('/cache/cvpzhr/outputs/forest-s01-zero-gt5x-20261107')
REG=Path('/cache/cvpzhr/experiments/forest-s01-decline')
ARCHIVE=REG/'04-zero-gt5x-20261107'


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
    cfg.update(output_dir=str(OUT),batch_size=4,zero_gt_loss_weight=5.)
    assert cfg['model']['gradient_checkpointing'] is False
    assert cfg['initialize_from']=='/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt'
    shutil.copytree(BASE/'code',WORK/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(WORK/'zero_target_loss.py',WORK/'code/mem_lite/zero_target_loss.py')
    trainer=WORK/'code/mem_lite/train_stage3.py';s=trainer.read_text()
    s=replace_once(s,'from mem_lite.weighted_sampler import DistributedWeightedSampler',
        'from mem_lite.weighted_sampler import DistributedWeightedSampler\nfrom mem_lite.zero_target_loss import zero_target_weights')
    s=replace_once(s,'    return config\n',
        '    if not math.isfinite(config.get("zero_gt_loss_weight", 1.)) or config.get("zero_gt_loss_weight", 1.) < 1:\n'
        '        raise ValueError("zero_gt_loss_weight must be finite and >= 1")\n    return config\n')
    s=replace_once(s,'totals = torch.zeros(5, device=device, dtype=torch.float64)',
        'totals = torch.zeros(8, device=device, dtype=torch.float64)')
    s=replace_once(s,'                loss_sum = (per_sample_loss * valid).sum()',
        '                target_weights = zero_target_weights(target_pt, config.get("zero_gt_loss_weight", 1.))\n'
        '                loss_sum = (per_sample_loss * target_weights * valid).sum()')
    s=replace_once(s,'            totals[4] += error.sum()',
        '            totals[4] += error.sum()\n'
        '            totals[5] += (per_sample_loss.detach().double() * valid).sum()\n'
        '            zero_valid = valid & (target_pt == 0)\n'
        '            totals[6] += zero_valid.sum()\n'
        '            totals[7] += (per_sample_loss.detach().double() * zero_valid).sum()')
    s=replace_once(s,'                    "pt_mae_pp": (100 * totals[2] / totals[1]).item(),',
        '                    "loss_unweighted": (totals[5] / totals[1]).item(),\n'
        '                    "zero_gt_samples": int(totals[6].item()),\n'
        '                    "zero_gt_ce": (totals[7] / totals[6]).item() if totals[6] > 0 else None,\n'
        '                    "zero_gt_loss_weight": config.get("zero_gt_loss_weight", 1.),\n'
        '                    "pt_mae_pp": (100 * totals[2] / totals[1]).item(),')
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
    save(WORK/'config.json',cfg)
    save(WORK/'code-sha256.json',{str(p.relative_to(WORK/'code')):sha(p) for p in (WORK/'code').rglob('*.py')})
    ARCHIVE.mkdir()
    for name in ['config.json','split.json','sampling-audit.json','code-sha256.json','cache-validation.json']:
        shutil.copy2(WORK/name,ARCHIVE/name)
    save(ARCHIVE/'manifest.json',dict(run_id=ARCHIVE.name,status='prepared',setup_dir=str(WORK),output_dir=str(OUT),
        baseline=str(REG/'03-history-pairs-20261106'),initialize_from=cfg['initialize_from'],
        initialize_sha256=sha(Path(cfg['initialize_from'])),zero_gt_loss_weight=5,
        loss='mean(valid_i * (5 if GT_i == 0 else 1) * soft-bin CE_i); denominator is actual valid sample count',
        validation='unchanged, unweighted CE and errors',gpus=[2,3],global_batch_size=8,total_steps=14604,
        comparison_note='Same v4 data, caches, base initialization, weighted sampling, LR and 3 epochs. Two GPUs with batch 4 instead of four with batch 2; same global draws/batch and update count, but backend RNG is not bitwise controlled. Initial GT-zero frames are also weighted.'))
    registry=read(REG/'index.json');assert ARCHIVE.name not in [r['id'] for r in registry['runs']]
    registry['runs'].append(dict(id=ARCHIVE.name,manifest=str(ARCHIVE/'manifest.json')));save(REG/'index.json',registry)
    print('Prepared zero GT x5 ablation: GPUs 2,3, global batch 8, 14604 steps, unweighted validation.',flush=True)


if __name__=='__main__':main()
