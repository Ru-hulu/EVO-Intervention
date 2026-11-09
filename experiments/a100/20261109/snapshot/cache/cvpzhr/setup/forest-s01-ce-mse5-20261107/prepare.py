"""Create an isolated CE plus MSE loss ablation from the completed v4 run."""
import hashlib
import json
import math
from pathlib import Path
import shutil

BASE=Path('/cache/cvpzhr/setup/forest-s01-history-pairs-20261106')
WORK=Path('/cache/cvpzhr/setup/forest-s01-ce-mse5-20261107')
OUT=Path('/cache/cvpzhr/outputs/forest-s01-ce-mse5-20261107')
REG=Path('/cache/cvpzhr/experiments/forest-s01-decline')
ARCHIVE=REG/'06-ce-mse5-20261107'


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
    cfg.update(output_dir=str(OUT),batch_size=4,progress_mse_weight=5.)
    assert cfg['model']['gradient_checkpointing'] is False
    assert cfg['initialize_from']=='/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt'
    shutil.copytree(BASE/'code',WORK/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(WORK/'progress_loss.py',WORK/'code/mem_lite/progress_loss.py')
    trainer=WORK/'code/mem_lite/train_stage3.py';s=trainer.read_text()
    s=replace_once(s,'from mem_lite.weighted_sampler import DistributedWeightedSampler',
        'from mem_lite.weighted_sampler import DistributedWeightedSampler\nfrom mem_lite.progress_loss import progress_objective')
    s=replace_once(s,'    return config\n',
        '    if not math.isfinite(config.get("progress_mse_weight", 0.)) or config.get("progress_mse_weight", 0.) < 0:\n'
        '        raise ValueError("progress_mse_weight must be finite and >= 0")\n    return config\n')
    s=replace_once(s,'totals = torch.zeros(5, device=device, dtype=torch.float64)',
        'totals = torch.zeros(6, device=device, dtype=torch.float64)')
    s=replace_once(s,'                loss_sum = (per_sample_loss * valid).sum()',
        '                objective = progress_objective(per_sample_loss, result["pt"], target_pt, config.get("progress_mse_weight", 0.))\n'
        '                loss_sum = (objective * valid).sum()')
    s=replace_once(s,'            totals[4] += error.sum()',
        '            totals[4] += error.sum()\n'
        '            totals[5] += (per_sample_loss.detach().double() * valid).sum()')
    s=replace_once(s,'                    "pt_mae_pp": (100 * totals[2] / totals[1]).item(),',
        '                    "loss_ce": (totals[5] / totals[1]).item(),\n'
        '                    "loss_mse": (totals[3] / totals[1]).item(),\n'
        '                    "progress_mse_weight": config.get("progress_mse_weight", 0.),\n'
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
        initialize_sha256=sha(Path(cfg['initialize_from'])),progress_mse_weight=5,
        loss='mean(valid_i * (soft-bin CE_i + 5*(prediction_i-GT_i)^2)); denominator is actual valid sample count',
        validation='unchanged, unweighted CE and errors',gpus=[2,3],global_batch_size=8,total_steps=14604,
        comparison_note='Same v4 data, caches, base initialization, sampling, LR, 3 epochs and global batch 8. Two GPUs x4, matching run 04/05. No zero-GT or native-failure loss weighting; all valid samples use CE+5*MSE. Prediction is the differentiable expectation over 101 bins.'))
    registry=read(REG/'index.json');assert ARCHIVE.name not in [r['id'] for r in registry['runs']]
    registry['runs'].append(dict(id=ARCHIVE.name,manifest=str(ARCHIVE/'manifest.json')));save(REG/'index.json',registry)
    print('Prepared CE+5*MSE ablation: GPUs 2,3, global batch 8, 14604 steps, unweighted validation.',flush=True)


if __name__=='__main__':main()
