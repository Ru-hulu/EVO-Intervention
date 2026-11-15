"""Merge verified input bindings; do not alter the existing FoREST process."""
from collections import Counter
from .common import *


def prepare():
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ROOT.mkdir(parents=True,exist_ok=True)
    source=read(SOURCE);forest=read(FOREST/'inputs.json');grm=read(GRM/'inputs.json');proof=read(GRM/'verification.json')
    assert proof['status']=='passed' and proof['inputs_sha256']==sha(GRM/'inputs.json')
    assert source['episodes']==forest['episodes']==grm['episodes']
    eps=source['episodes'];assert Counter(r['suite'] for r in eps)==EXPECTED
    assert set(Counter((r['suite'],r['task_id']) for r in eps).values())=={2}
    protected={};registry=dict(source['registry']);resolved={}
    def protect(p,h=None):
        h=h or sha(p);actual=resolve(p,h);protected[str(actual)]=h;resolved[str(p)]=str(actual)
    for p in (SOURCE,SELECTION,FOREST/'inputs.json',FOREST/'preparation.json',GRM/'inputs.json',GRM/'verification.json'):
        protect(p)
    for p,h in {**grm['protected_files'],**grm['artifacts_sha256']}.items():protect(p,h)
    # Source hashes that changed only through the authorized dataset alias swap
    # resolve to the exact previous bytes; no new labels/splits are substituted.
    for p,h in source['frozen_files'].items():protect(p,h)
    for p,h in read(GRM/'source-images.json').items():assert sha(p)==h,p
    for key,info in grm['registry'].items():
        item=dict(info,config=read(info['config_path']))
        assert item['config']['suite']==item['suite'] and item['config']['name']==item['model']
        registry[key]=item
    assert len(registry)==15
    for key,info in registry.items():
        assert key==f"{info['model']}-{info['suite']}" and info['model'] in MODELS
        protect(info['proof'],info['proof_sha256'])
        checkpoint=Path(info['checkpoint']);state=checkpoint/'trainable-state.pt' if checkpoint.is_dir() else checkpoint
        protect(state,info['checkpoint_sha256'])
        if info['model']!='forest':
            proof=read(info['proof']);cfg=info['config']
            assert proof['status']=='passed' and not proof['pilot'] and proof['epoch']==1
            assert proof['expected_episodes']==35 and proof['suite']==info['suite']
            assert cfg['epochs']==1 and cfg['episodes']==35 and cfg['training_fps']==5
            for p,h in cfg['weights_sha256'].items():protect(p,h)
    tasks={}
    for suite in SUITES:
        selected=[r for r in eps if r['suite']==suite]
        for model in MODELS[:-1]:
            key=f'{model}-{suite}';parts=partition(selected,2*REPLICAS[model])
            tasks[key]=dict(model=model,suite=suite,replicas_per_gpu=REPLICAS[model],shards=parts,
                expected_episodes=EXPECTED[suite],expected_observations=sum(len(r['frame_indices']) for r in selected))
    tuning=Path('/cache/cvpzhr/setup/group5-throughput-20261115/benchmark-results.json')
    benchmark=read(tuning);protect(tuning)
    assert benchmark['procvlm-2b']['choice']['workers_per_gpu']==4
    assert benchmark['robometer-4b']['choice']['workers_per_gpu']==2
    value=dict(schema='soc84-five-finetuned-baseline-v1',episodes=eps,registry=registry,tasks=tasks,
        task_order=list(tasks),external_forest_status=str(FOREST/'status.json'),external_forest_inputs_sha256=sha(FOREST/'inputs.json'),
        protected_files=protected,resolved_sources=resolved,selection_sha256=sha(SELECTION),
        protocol=dict(trajectories=EXPECTED,per_task=2,input_fps=5,include_last_frame=True,models=list(MODELS),
            weights='matching-suite SOC35 independent 1-epoch final checkpoints',memory='native per-method',
            perturbations=False,no_memory_ablation=False,multifps=False,metrics=['pearson','spearman','VS','ER','C'],
            aggregation='equal trajectories, undefined/invalid denominators retained',future_endpoint_models=list(MODELS[:2]),
            gpus_per_task=2,gpu_pairs=[[2,3],[0,1]],forest_pair_reserved_until_controller_completed=True,
            batch=1,replicas_per_gpu=REPLICAS,cpu_prefetch_depth=3,retained_gpu_headroom_gib=16,
            concurrency_policy='Proc/RoboMeter measured prior settings; GRM staged replicas after a first real episode per GPU',
            unknown_failure_policy='no blind restart; finish healthy workers and continue independent tasks',
            recurring_automation=False))
    freeze(ROOT/'inputs.json',value)
    save(ROOT/'preparation.json',dict(status='passed',models=5,suites=3,tasks=15,episodes_per_model=84,
        observations_per_model=15096,inputs_sha256=sha(ROOT/'inputs.json'),original_forest_untouched=True))
    print('Prepared 15 tasks: 3 external FoREST + 12 queued two-GPU tasks',flush=True)


if __name__=='__main__':prepare()
