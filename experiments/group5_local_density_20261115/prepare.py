"""Prepare only: no model loading, no CUDA work, no inference queue creation."""
import csv
import fcntl
import os
import time
from collections import Counter
from .common import *
from .protocol import make_entry, segment_order, model_cache, slots
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics


def validate_baseline(model,row,entry,info,source_sha):
    path=RESULTS/'fps6'/model/'episodes'/(row['key'].replace('/','__')+'.json')
    result=read(path)
    assert result['status']=='completed'
    assert result['identity']==dict(inputs_sha256=source_sha,model=model,suite=row['suite'],key=row['key'],
        checkpoint_sha256=info['checkpoint_sha256'],row_sha256=digest(row),entry_sha256=digest(entry),input_fps=6,memory_enabled=True)
    for k in ('source_frame_indices','timestamps','target_pt'):assert result[k]==entry[k]
    n=len(entry['target_pt'])
    assert len(result['predictions'])==len(result['raw_responses'])==len(result['input_audits'])==n
    assert result['source_video_sha256']==row['video_hashes'] and result['memory_enabled']
    audit=result['initialization'];assert digest(audit)==result['initialization_sha256']
    assert audit['strict_final_state'] and audit['memory_enabled'] and audit['epoch']==1
    assert not audit['future_endpoint_reference'] and not audit['previous_text_used']
    for k in ('model','suite','checkpoint_sha256'):assert audit[k]==result['identity'][k]
    for i,a in enumerate(result['input_audits']):
        assert a['playback_step']==i and a['playback_position']==entry['source_frame_indices'][i]
        assert [list(x) for x in a['input_playback_slots']]==slots(i,model)
    full=trajectory_metrics(result['predictions'],entry['target_pt'],entry['timestamps'])
    common=entry['common_scoring_indices']
    subset=trajectory_metrics([result['predictions'][i] for i in common],[entry['target_pt'][i] for i in common],entry['common_scoring_timestamps'])
    assert result['metrics']==dict(full_rate=full,common_6fps=subset)
    return dict(path=str(path),sha256=sha(path),observations=n,invalid_predictions=full['invalid_predictions'],metrics=result['metrics'])


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Preparation must be explicitly CPU-only'
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (DATA/'inputs.json').exists():
        from .verify import verify
        return verify()
    save(ROOT/'status.json',dict(status='preparing',pid=os.getpid(),evaluation_started=False,updated=time.time()))
    receipt=read(ROOT/'source-checksums.json')
    for p,h in receipt['files'].items():assert sha(ROOT/'code'/p)==h,p
    source=read(SOURCE);source_sha=sha(SOURCE);old_report=read(RESULTS/'metrics.json')
    assert old_report['status']=='completed' and old_report['inputs_sha256']==source_sha
    assert Counter(r['suite'] for r in source['episodes'])==dict(State=16,Order=30,Count=14)
    assert len(source['episodes'])==len({r['key'] for r in source['episodes']})==60
    assert set(Counter((r['suite'],r['task_id']) for r in source['episodes']).values())=={2}
    protected={str(SOURCE):source_sha,str(RESULTS/'metrics.json'):sha(RESULTS/'metrics.json')}
    streams={}
    for rate in RATES:
        item=source['streams'][str(rate)];assert sha(item['path'])==item['sha256']
        protected[item['path']]=item['sha256'];streams[rate]={e['key']:e for e in read(item['path'])['episodes']}
    for info in source['registry'].values():
        path=Path(info['checkpoint']);path=path/'trainable-state.pt' if path.is_dir() else path
        assert sha(path)==info['checkpoint_sha256'];protected[str(path)]=info['checkpoint_sha256']
        assert sha(info['proof'])==info['proof_sha256'];protected[info['proof']]=info['proof_sha256']
    baselines={};levels={k:[] for k in range(1,6)};images={};image_count=0
    for j,row in enumerate(source['episodes']):
        key=row['key'];assert row['split']=='val' and row['source_rate']=='30'
        for path,h in zip(row['videos'],row['video_hashes']):
            if path:assert sha(path)==h;protected[path]=h
        assert sha(row['progress_path'])==row['progress_sha256'];protected[row['progress_path']]=row['progress_sha256']
        with open(row['progress_path']) as f:truth=[float(r['pt']) for r in csv.DictReader(f)]
        assert len(truth)==row['frames']
        old={rate:streams[rate][key] for rate in RATES}
        for e in old.values():assert e['target_pt']==[truth[f] for f in e['source_frame_indices']]
        assert old[30]['source_frame_indices']==list(range(row['frames']))
        cache=Path(row['image_root']);proof_path=cache/'verification.json';proof=read(proof_path)
        assert proof['input']==dict(video_hashes=row['video_hashes'],frame_indices=row['frame_indices'])
        assert proof['decoded_native_frames']==row['frames'] and proof['lossless']
        expected={f'{f:08d}-{v}.png' for f in range(row['frames']) for v,p in enumerate(row['videos']) if p}
        assert set(proof['png_sha256'])==expected
        for name,h in proof['png_sha256'].items():assert sha(cache/name)==h
        protected[str(proof_path)]=sha(proof_path);images[key]=dict(path=str(cache),proof=str(proof_path),sha256=sha(proof_path),pngs=len(expected))
        image_count+=len(expected);order=segment_order(key)
        for level in levels:
            e=make_entry(row,old,level,order)
            if level==1:
                for k in ('source_frame_indices','timestamps','target_pt','common_scoring_indices'):assert e[k]==old[6][k]
            if level>1:assert set(levels[level-1][-1]['source_frame_indices'])<=set(e['source_frame_indices'])
            levels[level].append(e)
        for model in MODELS:
            b=validate_baseline(model,row,old[6],source['registry'][f'{model}-{row["suite"]}'],source_sha)
            baselines[f'{model}:{key}']=b;protected[b['path']]=b['sha256']
        save(ROOT/'status.json',dict(status='preparing',pid=os.getpid(),verified_episodes=j+1,verified_pngs=image_count,
            evaluation_started=False,updated=time.time()))
    shared={};jobs=[];outputs={};by_key={r['key']:r for r in source['episodes']}
    for level,episodes in levels.items():
        value=dict(schema='cumulative-local-density-stream-v1',level=level,seed=SEED,episodes=episodes,expected_episodes=60,
            expected_observations=sum(len(e['target_pt']) for e in episodes),common_scoring_fps=6)
        path=DATA/'streams'/f'level{level}.json';h=freeze(path,value);outputs[str(path)]=h
        shared[str(level)]=dict(path=str(path),sha256=h,observations=value['expected_observations'])
        for e in episodes:
            row=by_key[e['key']]
            for model in MODELS:
                info=source['registry'][f'{model}-{row["suite"]}'];b=baselines[f'{model}:{e["key"]}'] if level==1 else None
                cache=model_cache(row,e,info,path,h,b);p=model_path(model,level,e['key']);outputs[str(p)]=freeze(p,cache)
                jobs.append(dict(model=model,suite=row['suite'],key=e['key'],level=level,cache=str(p),cache_sha256=outputs[str(p)],
                    observations=len(e['target_pt']),status='reuse_verified_baseline' if level==1 else 'prepared_waiting_for_user_instruction'))
    protocol=dict(mode='cumulative',seed=SEED,segments=10,base_fps=6,new_segments_per_level=2,new_segment_fps=[12,18,24,30],
        segmentation='Ten equal half-open time intervals [j*N/300,(j+1)*N/300); native frame f belongs to floor(10*f/N)',
        sampling='Restrict each previously frozen global FPS grid to its assigned segments; include original final frame',
        boundary_phase='Global video time origin; sampling grid does not restart at segment boundaries',
        time_weighted_average_fps=[6,7.2,9.6,13.2,18],shared_randomization_across_models=True,
        per_trajectory_randomization='SHA256 canonical JSON of experiment/key/seed, then Python Random shuffle of 0..9',
        selection='Exact prior 60 native-30-FPS validation trajectories; two per task',
        weights='Exact prior nine matching-suite SOC35 final 1-epoch checkpoints',memory_configuration=source['protocol'],
        first_level='Reuse verified original 6 FPS predictions; no model calls',
        higher_levels='Recompute each entire trajectory with fresh initial memory; never splice old predictions',
        memory_reset_at_segment_boundary=False,common_scoring_fps=6,interpolation=False,label_recomputation=False,
        preparation='Shared verified lossless RGB cache plus separate native model input-index caches; no model forward or feature cache',
        evaluation_started=False,automatic_evaluation_enabled=False)
    outputs[str(DATA/'baseline-results.json')]=freeze(DATA/'baseline-results.json',baselines)
    outputs[str(DATA/'jobs.json')]=freeze(DATA/'jobs.json',dict(jobs=jobs,automatic_launch=False,requires_user_instruction=True))
    inputs=dict(schema='group5-local-density-v1',protocol=protocol,episodes=source['episodes'],registry=source['registry'],
        streams=shared,source_inputs_sha256=source_sha,baseline_predictions=len(baselines),image_caches=images,
        protected_source_files=protected,cache_files_sha256=outputs,source_checksums_sha256=sha(ROOT/'source-checksums.json'),
        expected_model_curves=900,reused_baseline_curves=180,new_inference_curves=720,
        expected_model_observations=3*sum(x['observations'] for x in shared.values()),evaluation_started=False)
    freeze(DATA/'inputs.json',inputs)
    from .verify import verify
    return verify()


if __name__=='__main__':
    try:print(json.dumps(prepare()))
    except BaseException:
        import traceback
        save(ROOT/'status.json',dict(status='preparation_failed',evaluation_started=False,error=traceback.format_exc(),updated=time.time()))
        raise
