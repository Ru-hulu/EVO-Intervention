"""Bind the completed caches to explicit training configs; never alter original caches."""
import copy
import time
from .common import *
from .data import grm_answer


def main():
    assert not (ROOT/'plan.json').exists(), 'Do not replace an existing plan'
    assert read(CACHE_SETUP/'final-verification.json')['status'] == 'passed'
    assert read(CACHE_SETUP/'proc-progress-only-verification.json')['status'] == 'passed'
    assert not OUTPUT.exists(), 'Preserve any existing training outputs'
    config_files = {}; supervision = {}
    for name in ORDER:
        receipt = read(CACHE/name/'cache-ready.json')
        provenance = read(CACHE_SETUP/f'{name}-native-provenance.json')
        assert sha(CACHE/name/'train.jsonl') == receipt['train_sha256']
        train = CACHE/name/('progress-only/train.jsonl' if name=='procvlm-2b' else 'train.jsonl')
        records = rows(train)
        assert len(records)==2550 and len({r['episode_id'] for r in records})==10
        assert all(r['split']=='train' for r in records)
        if name in ('robobrain25-4b','dopamine-3b'):
            targets = [dict(id=r['id'],original_pt=r['pt'],answer=grm_answer(r)) for r in records]
            save(ROOT/name/'text-targets.json', targets)
            supervision[name] = dict(policy='native integer percent relative score from initial BEFORE',
                maximum_rounding_error_pt=max(abs(int(t['answer'].split('>')[1].split('%')[0])/100-t['original_pt']) for t in targets),
                original_pt_unchanged=True,targets_sha256=sha(ROOT/name/'text-targets.json'))
        checkpoint = Path(receipt['checkpoint'])
        # Recompute source and official-weight fingerprints before any training begins.
        for p,digest in provenance['source_sha256'].items(): assert sha(p)==digest, p
        weights = {str(p):sha(p) for p in sorted(checkpoint.glob('*.safetensors'))}
        assert weights
        if name=='robometer-4b':
            micro, accumulation, rate, schedule, warmup = 8,1,1e-4,'cosine',.03
            env = '/cache/cvpzhr/envs/robometer/bin/python'
        elif name=='procvlm-2b':
            micro, accumulation, rate, schedule, warmup = 2,4,1e-5,'linear',.05
            env = PYTHON
        else:
            micro, accumulation, rate, schedule, warmup = 4,2,1e-4,'cosine',.03
            env = PYTHON
        config = dict(name=name,model_path=str(checkpoint),official_revision=provenance['official_revision'],
            python=env,train_data=str(train),train_sha256=sha(train),weights_sha256=weights,
            selection_sha256=sha(CACHE_SETUP/'selection.json'),samples=2550,episodes=10,
            epochs=1,seed=42,world_size=2,gpus=[0,1],microbatch=micro,accumulation=accumulation,
            effective_batch=micro*accumulation*2,learning_rate=rate,schedule=schedule,warmup=warmup,
            weight_decay=.01,lora_rank=16,lora_alpha=32,lora_dropout=.05,
            gradient_checkpointing=False,attention='sdpa',bf16=True,allocator_fraction=.78,
            max_length=16384 if name=='procvlm-2b' else 4096,
            num_workers=4,output_dir=str(OUTPUT/name),setup_dir=str(ROOT/name),
            initialization='official weights, fresh language LoRA; no previous fine-tuned adapter',
            supervision=('official continuous-PT C51 progress-head loss' if name=='robometer-4b' else
                         'progress-only text answer and EOS; no CoT/remaining-action/auxiliary-head loss' if name=='procvlm-2b' else
                         'native integer-percent <score> answer and EOS; initial BEFORE references'),
            future_endpoint_reference=name in ('robobrain25-4b','dopamine-3b'))
        path=ROOT/name/'config.json'; save(path,config);config_files[name]=str(path)
    save(ROOT/'plan.json',dict(order=ORDER,gpus=[0,1],epochs=1,sample_fps=5,samples=2550,episodes=10,
        task_counts={k:2 for k in ['R01','R02','R03','R04','R05']},configs=config_files,
        config_sha256={k:sha(v) for k,v in config_files.items()},supervision=supervision,
        cache_audit_sha256=sha(CACHE_SETUP/'final-verification.json'),created=time.time(),
        evaluations_enabled=False,vlac_training_enabled=False,forest_owned_by_other_workspace=True,
        safety='one two-rank model at a time; 16GiB GPU reserve; finite-loss/gradient canary gate'))
    print('Prepared GPU0/1 official-model queue; no training launched',flush=True)


if __name__=='__main__': main()
