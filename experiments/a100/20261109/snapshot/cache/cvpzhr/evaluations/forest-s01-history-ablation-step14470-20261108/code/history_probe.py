"""Paired history intervention with fixed current views and instruction."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device

ROOT = Path('/cache/cvpzhr/evaluations/forest-s01-history-ablation-step14470-20261108')
SOURCE = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline/State/01-cover_uncover_blocks')
CHECKPOINT = Path('/cache/cvpzhr/evaluations/forest-s01-failure-continue-ce-episode55-step14470-20261108/checkpoint.pt')
PAIRS = [(55, 13), (54, 18), (84, 4)]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def geometry(state):
    # Covers dominate matching; wrist positions disambiguate action phase.
    covers = [v for pose in state['poses']['covers'] for v in pose['p']]
    wrists = [v * .5 for pose in state['tool_poses'] for v in pose[:3]]
    return np.array(covers+wrists)


def main():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'results.jsonl').exists(), 'Keep each diagnostic run immutable'
    all_rows = json.loads(Path('/cache/cvpzhr/setup/forest-s01-decline-20261106/split.json').read_text())['records']
    rows = {r['episode_index']: r for r in all_rows if r['split']=='validation'}
    used = sorted({ep for pair in PAIRS for ep in pair})
    datasets = {ep: M3Dataset([str(Path(rows[ep]['cache_dir'])/'samples.jsonl')]) for ep in used}
    indices = {ep: {s['frame_index']: i for i, (_,s,_) in enumerate(ds.samples)} for ep,ds in datasets.items()}
    states = {ep: {s['frame']: s for s in map(json.loads,(SOURCE/rows[ep]['states_path']).read_text().splitlines())} for ep in used}
    plan = []
    for fail, success in PAIRS:
        assert not rows[fail]['task_success'] and rows[success]['task_success']
        assert rows[fail]['task_prompt'] == rows[success]['task_prompt']
        assert states[fail][0]['target_state_transition'] == states[success][0]['target_state_transition']
        assert not any(s['fail_flag'] for s in states[success].values())
        first_failure = next(f for f,s in states[fail].items() if s['fail_flag'])
        wanted = [520,628,680,728,800,900,960,1000] if fail==55 else [first_failure+4,first_failure+80,first_failure+200,first_failure+320]
        for wanted_frame in wanted:
            frame = min(indices[fail], key=lambda f:abs(f-wanted_frame))
            candidate_frames = [f for f in indices[success] if f>=400]
            matched = min(candidate_frames,key=lambda f:np.linalg.norm(geometry(states[success][f])-geometry(states[fail][frame])))
            plan.append(dict(failure_episode=fail,failure_frame=frame,success_episode=success,success_frame=matched,
                failure_has_occurred=states[fail][frame]['fail_flag'],first_failure_frame=first_failure,
                open_order=states[fail][0]['open_order'],pose_distance=float(np.linalg.norm(geometry(states[success][matched])-geometry(states[fail][frame])))))
    checkpoint_sha = hashlib.file_digest(CHECKPOINT.open('rb'), 'sha256').hexdigest()
    dump(ROOT/'protocol.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_sha,step=14470,
        batch_size=1,training_modified=False,plan=plan,
        selection='Episode55 is a known failure case. Two additional same-layout pairs provide comparison; this is not an unbiased population estimate.',
        intervention='Identical current RGB views and instruction within each contrast. Only history changes. All sources are held-out validation episodes.',
        matching='Same required state transition/color order; donor frame minimizes cover and wrist pose distance, without looking at model predictions. Different scene seeds and motions remain.',
        caution='Swapped histories are synthetic inputs, not physically replayed counterfactual rollouts. They do not carry verified regression targets. Fixed-length resampling and zero tokens are auxiliary out-of-distribution controls.'))
    model, processor, cfg = load_checkpoint(str(CHECKPOINT))
    collator = Stage3Collator(processor,cfg['model']['history_dim'],cfg['max_sequence_length'],datasets[55].history_capacity)
    started = time.monotonic()
    records = []
    with torch.inference_mode(), (ROOT/'results.jsonl').open('w') as output:
        for pair_index,pair in enumerate(plan):
            fail,success = pair['failure_episode'],pair['success_episode']
            a = datasets[fail][indices[fail][pair['failure_frame']]]
            b = datasets[success][indices[success][pair['success_frame']]]
            for direction,current,donor in [('failure_view',a,b),('success_view',b,a)]:
                base = to_device(collator([current])['inputs'],'cuda')
                swapped = to_device(collator([dict(current,history_tokens=donor['history_tokens'])])['inputs'],'cuda')
                empty = to_device(collator([dict(current,history_tokens=None)])['inputs'],'cuda')
                for other in (swapped,empty):
                    for key in ('pixel_values','image_grid_thw'):
                        assert torch.equal(base[key],other[key]), key
                    def text_ids(inputs):
                        return inputs['input_ids'][inputs['attention_mask'].bool() & ~inputs['history_mask']]
                    assert torch.equal(text_ids(base),text_ids(other))
                n = len(current['history_tokens'])
                matched = dict(base,history_tokens=torch.zeros_like(base['history_tokens']))
                donor_tokens = donor['history_tokens'].to('cuda').float()
                resampled = F.adaptive_avg_pool1d(donor_tokens.T.unsqueeze(0),n)[0].T
                matched['history_tokens'][0,:n] = resampled.to(base['history_tokens'].dtype)
                zero = dict(base,history_tokens=torch.zeros_like(base['history_tokens']))
                for other in (matched,zero):
                    assert all(torch.equal(base[k],other[k]) for k in base if k!='history_tokens')
                variants = [('original',base),('original_repeat',base),('swapped',swapped),
                    ('empty',empty),('zero_same_length',zero),('swapped_same_length',matched)]
                for condition,inputs in variants:
                    result = model(inputs)
                    probabilities = result['bin_probs'][0].float().cpu().numpy()
                    prediction = float(result['pt'][0])
                    assert np.isfinite(probabilities).all() and (probabilities>=0).all()
                    assert abs(probabilities.sum()-1)<1e-6
                    assert abs(probabilities@np.linspace(0,1,101)-prediction)<1e-6
                    record = dict(pair_index=pair_index,**pair,direction=direction,condition=condition,
                        current_episode=fail if direction=='failure_view' else success,
                        current_frame=current['frame_index'],donor_episode=success if direction=='failure_view' else fail,
                        donor_frame=donor['frame_index'],original_target=current['target_pt'],
                        original_history_tokens=n,donor_history_tokens=len(donor['history_tokens']),
                        current_view_sha256=[hashlib.sha256(image.tobytes()).hexdigest() for image in current['images']],
                        predicted_pt=prediction,probabilities=probabilities.tolist(),
                        p_zero=float(probabilities[0]),p_low=float(probabilities[:11].sum()),p_high=float(probabilities[80:].sum()))
                    records.append(record);output.write(json.dumps(record)+'\n');output.flush()
                print(json.dumps(dict(pair=pair_index,direction=direction,done=len(records),total=len(plan)*12,seconds=time.monotonic()-started)),flush=True)
                del base,swapped,empty,matched,zero,variants,inputs,result
    repeats = []
    for i in range(len(plan)):
        for direction in ('failure_view','success_view'):
            group={r['condition']:r for r in records if r['pair_index']==i and r['direction']==direction}
            assert len({tuple(r['current_view_sha256']) for r in group.values()})==1
            repeats.append(abs(group['original']['predicted_pt']-group['original_repeat']['predicted_pt']))
    dump(ROOT/'status.json',dict(phase='completed',records=len(records),pairs=len(plan),seconds=time.monotonic()-started,
        max_repeat_delta_pp=max(repeats)*100,current_images_verified_identical=True,nonhistory_inputs_verified_identical_for_length_control=True))


if __name__=='__main__':
    main()
