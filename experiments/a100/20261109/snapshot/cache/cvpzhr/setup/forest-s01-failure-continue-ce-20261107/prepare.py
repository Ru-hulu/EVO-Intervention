"""Prepare uniform, unweighted CE training after v5 data verification."""
import json,hashlib,shutil
from pathlib import Path
BASE=Path('/cache/cvpzhr/setup/forest-s01-ce-mse5-20261107')
TEMPLATE=Path('/cache/cvpzhr/setup/forest-s01-ce-mse5-early-stopping-20261107')
WORK=Path('/cache/cvpzhr/setup/forest-s01-failure-continue-ce-20261107')
DATA=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261107-v5-failure-continue')
CACHE=Path('/cache/cvpzhr/datasets/forest-s01-failure-continue-5fps-20261107')
OUT=Path('/cache/cvpzhr/outputs/forest-s01-failure-continue-ce-20261107')
REG=Path('/cache/cvpzhr/experiments/forest-s01-decline')
ARCHIVE=REG/'07-failure-continue-ce-20261107'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    assert read(DATA/'verification/history-pairs.json')['complete']
    assert not OUT.exists() and not CACHE.exists() and not ARCHIVE.exists()
    split=read(DATA/'split.json');old=read(BASE/'split.json')
    assert split['records'][:150]==old['records']
    assert len(split['records'])==174
    cfg=read(TEMPLATE/'config.json')
    cfg.update(output_dir=str(OUT),batch_size=4,epochs=3,progress_mse_weight=0.,early_stopping_patience=3,early_stopping_min_delta_pp=0.)
    cfg.pop('weighted_sampling',None)
    for key in ['early_stopping_state','best_validation_mae_pp','best_validation_step']:cfg.pop(key,None)
    cfg['sample_files']=list(dict.fromkeys(cfg['sample_files']))  # remove old extra success visit
    assert len(cfg['sample_files'])==122
    assert cfg['model']['gradient_checkpointing'] is False
    assert cfg['initialize_from']=='/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt'
    assert cfg['validation_sample_files']==read(BASE/'config.json')['validation_sample_files']
    CACHE.mkdir();(CACHE/'train').mkdir()
    for row in split['records'][150:]:
        assert row['split']=='train' and row['category']=='script_history_failure'
        row['cache_dir']=str(CACHE/'train'/f"episode_{row['episode_index']:06d}")
        cfg['sample_files'].append(str(Path(row['cache_dir'])/'samples.jsonl'))
    assert len(cfg['sample_files'])==len(set(cfg['sample_files']))==146
    shutil.copytree(TEMPLATE/'code',WORK/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(BASE/'validation-event-groups.json',WORK/'validation-event-groups.json')
    cfg['validation_event_groups']=str(WORK/'validation-event-groups.json')
    save(WORK/'config.json',cfg);save(WORK/'split.json',split)
    save(CACHE/'index.json',dict(split,history_source=cfg['history_source'],model_path=cfg['model_path']))
    ARCHIVE.mkdir()
    for f in ['config.json','split.json']:shutil.copy2(WORK/f,ARCHIVE/f)
    save(ARCHIVE/'manifest.json',dict(status='prepared',setup_dir=str(WORK),output_dir=str(OUT),data=str(DATA),
        loss='unweighted soft-bin CE',sampling='uniform without replacement; no duplicate sample-file visits',
        initialize_from=cfg['initialize_from'],gpus=[2,3],global_batch_size=8,epochs=3,
        early_stopping=dict(metric='val_mae_pp',patience=3,min_delta_pp=0),validation_unchanged=True))
    registry=read(REG/'index.json');assert ARCHIVE.name not in [r['id'] for r in registry['runs']]
    registry['runs'].append(dict(id=ARCHIVE.name,manifest=str(ARCHIVE/'manifest.json')));save(REG/'index.json',registry)
    save(WORK/'code-sha256.json',{str(f.relative_to(WORK/'code')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (WORK/'code').rglob('*.py')})
if __name__=='__main__':main()
