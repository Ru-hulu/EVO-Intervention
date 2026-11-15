import random
from .common import digest, SEED, RATES
from experiments.prm_five_groups_20261114.eval_common import uniform_indices


def segment_order(key, seed=SEED):
    order=list(range(10))
    random.Random(int(digest(dict(experiment='cumulative-local-density-v1',key=key,seed=seed)),16)).shuffle(order)
    return order


def segment_of(frame, frames):
    assert type(frame) is int and 0<=frame<frames and frames>=10
    return frame*10//frames


def segment_rates(order, level):
    assert sorted(order)==list(range(10)) and level in range(1,6)
    rates=[6]*10
    for k in range(1,level):
        for segment in order[2*(k-1):2*k]:rates[segment]=RATES[k]
    return rates


def make_entry(row, old_entries, level, order):
    n=row['frames'];rates=segment_rates(order,level)
    frames=sorted(f for rate,entry in old_entries.items() for f in entry['source_frame_indices']
                  if rates[segment_of(f,n)]==rate)
    assert frames==sorted(set(frames)) and frames[0]==0 and frames[-1]==n-1
    truth=old_entries[30]['target_pt'];baseline=old_entries[6]
    lookup={f:i for i,f in enumerate(frames)}
    return dict(key=row['key'],suite=row['suite'],task_id=row['task_id'],level=level,seed=SEED,
        segment_order=order,segment_rates=rates,segment_ids=[segment_of(f,n) for f in frames],
        segment_frame_bounds=[(j*n+9)//10 for j in range(11)],
        segment_time_boundaries=[j*n/300 for j in range(11)],
        duration_seconds=n/30,nominal_time_weighted_fps=sum(rates)/10,
        source_frame_indices=frames,timestamps=[f/30 for f in frames],target_pt=[truth[f] for f in frames],
        progress_sha256=row['progress_sha256'],
        common_scoring_indices=[lookup[f] for f in baseline['common_scoring_native_frames']],
        common_scoring_native_frames=baseline['common_scoring_native_frames'],
        common_scoring_timestamps=baseline['common_scoring_timestamps'])


def slots(step, model):
    if model=='procvlm-2b':return [[max(0,i),0] for i in range(step-7,step+1)]
    if model=='robometer-4b':return [[i,0] for i in uniform_indices(step)[:7]]+[[step,v] for v in range(3)]
    assert model=='forest'
    return [[step,v] for v in range(3)]


def inference_episode(row, entry):
    from pathlib import Path
    image_root=Path(row['image_root'])
    return dict(key=row['key'],task=row['task_prompt'],positions=entry['source_frame_indices'],
        images=[[str(image_root/f'{frame:08d}-{v}.png') if path else None
                 for v,path in enumerate(row['videos'])] for frame in entry['source_frame_indices']])


def model_cache(row, entry, info, shared_path, shared_sha, baseline=None):
    model=info['model'];episode=inference_episode(row,entry)
    return dict(schema='local-density-native-input-index-cache-v1',model=model,suite=row['suite'],
        key=row['key'],level=entry['level'],checkpoint=info['checkpoint'],checkpoint_sha256=info['checkpoint_sha256'],
        shared_stream=str(shared_path),shared_stream_sha256=shared_sha,entry_sha256=digest(entry),
        inference=episode,input_playback_slots=[slots(i,model) for i in range(len(episode['positions']))],
        memory_enabled=True,memory_reset='once at trajectory start; never at segment boundaries',
        memory_execution='native CLIP/max_depth online state' if model=='forest' else 'native observed-prefix image history',
        uses_only_observed_prefix=True,targets_in_inference=False,future_endpoint_reference=False,
        baseline_result=baseline,reuse_predictions=entry['level']==1,evaluation_started=False)
