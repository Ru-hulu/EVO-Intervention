"""Audit all six immutable caches and bind independent one-epoch runs."""
from collections import Counter
from safetensors import safe_open
from .common import *


def main():
    assert not (ROOT/'plan.json').exists() and not OUTPUT.exists()
    audit=read(CACHE/'final-audit.json');assert audit['status']=='passed'
    selection=read(CACHE_SETUP/'selection.json');selection_sha=sha(CACHE_SETUP/'selection.json')
    assert selection_sha==audit['selection_sha256']
    for p,h in audit['code_sha256'].items():assert sha(p)==h,p
    configs={};lengths={}
    for name,short in [('robometer-4b','robometer'),('procvlm-2b','procvlm')]:
        reference=read('/cache/cvpzhr/setup/recover-five-finetune-20261114/robometer-4b/config.json' if short=='robometer' else
                       '/cache/cvpzhr/setup/procvlm-recover-10-progress-only-e1-20261114/reference.config.json')
        for p,h in reference['weights_sha256'].items():assert sha(p)==h,p
        provenance=audit['model_provenance'][short]
        for p,h in provenance['processor_tokenizer_config_sha256'].items():assert sha(p)==h,p
        configs[name]={};cases=[]
        for suite in SUITES:
            folder=CACHE/short/suite;receipt=read(folder/'cache-ready.json');binding=read(folder/'data-binding.json')
            path=folder/'train.jsonl';records=rows(path)
            assert sha(path)==receipt['train_sha256'] and receipt['selection_sha256']==selection_sha
            assert len(records)==selection['suites'][suite]['observations']
            expected={r['episode_id'] for r in selection['episodes'] if r['suite']==suite}
            assert len(expected)==35 and {r['episode_id'] for r in records}==expected
            source={r['episode_id']:rows(r['source_samples']) for r in selection['episodes'] if r['suite']==suite}
            for row in records:
                ref=source[row['episode_id']][row['sample_index']]
                assert row['split']=='train' and row['suite']==suite and row['augmentation']=='none'
                assert row['pt']==ref['pt'] and row['frame_index']==ref['frame_index']
            for item in receipt['receipts']:
                assert sha(item['pixel_cache'])==item['pixel_sha256']
                assert sha(item['episode_jsonl'])==item['episode_jsonl_sha256']
                with safe_open(item['pixel_cache'],framework='pt',device='cpu') as f:
                    lengths[item['pixel_cache']]=f.get_tensor('input_input_ids').numel()
                episode=[r for r in records if r['episode_id']==item['episode_id']]
                cases.append(episode[len(episode)//2])
            cfg=dict(reference)
            cfg.update(name=name,suite=suite,model_path=binding['model_directory'],official_revision=binding['model_revision'],
                train_data=str(path),train_sha256=sha(path),selection_sha256=selection_sha,
                samples=len(records),episodes=35,epochs=1,gpus=GPUS[name],
                setup_dir=str(ROOT/name/suite),output_dir=str(OUTPUT/name/suite),
                microbatch=8 if short=='robometer' else 16,accumulation=1,
                effective_batch=16 if short=='robometer' else 32,
                gradient_checkpointing=False,allocator_fraction=.88,min_free_gib=8,
                num_workers=4,max_length=binding['max_length'],
                processor_config={k:binding[k] for k in ('min_pixels','max_pixels')},
                pilot_steps=4,save_limit=2,training_fps=5,augmentation='none',
                initialization='Independent official model and fresh LoRA per suite; never resume pilot or another suite',
                cache_binding_sha256=sha(folder/'data-binding.json'),cache_audit_sha256=sha(CACHE/'final-audit.json'))
            assert cfg['effective_batch']==cfg['microbatch']*cfg['accumulation']*2
            configs[name][suite]=cfg
        cases.sort(key=lambda r:lengths[r['pixel_cache']])
        for cfg in configs[name].values():
            cfg['pilot_cases']=[cases[-1],cases[0]]
            cfg['pilot_scope']='Longest and shortest native cached prompts across this model\'s three authorized suites; pilot weights discarded'
            cfg['maximum_cached_prompt_tokens']=lengths[cases[-1]['pixel_cache']]
            save(Path(cfg['setup_dir'])/'reference-config.json',cfg)
    save(ROOT/'plan.json',dict(schema='soc35-independent-six-finetunes-v1',configs=configs,
        suite_order=SUITES,gpus=GPUS,models_initialize_independently_per_suite=True,
        training_samples_by_suite={s:selection['suites'][s]['observations'] for s in SUITES},
        episodes_per_model_per_suite=35,epochs=1,augmentation='none',sample_fps=5,
        selection_sha256=selection_sha,cache_audit_sha256=sha(CACHE/'final-audit.json'),
        evaluations_enabled=False,new_automations_created=False))
    save(ROOT/'preflight.json',dict(status='passed',caches=6,rows=38958,source_frames_pt_unchanged=True,
        all_pixel_hashes_verified=True,all_base_weight_hashes_verified=True,
        max_prompt_tokens={m:configs[m]['State']['maximum_cached_prompt_tokens'] for m in configs}))


if __name__=='__main__':main()
