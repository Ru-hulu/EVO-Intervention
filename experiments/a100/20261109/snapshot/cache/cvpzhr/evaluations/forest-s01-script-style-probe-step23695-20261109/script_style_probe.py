"""Audit source/label imbalance and test scripted failures/successes without training."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint,to_device
from history_swap_current import verify_inputs

OUT=Path('/cache/cvpzhr/evaluations/forest-s01-script-style-probe-step23695-20261109')
SETUP=Path('/cache/cvpzhr/setup/forest-s01-order102-expand120-20261109')
PY='/cache/cvpzhr/envs/forest/bin/python'


def read(p):return json.loads(p.read_text())


def save(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def load(row):
    cache=Path(row['cache_dir']);ev=Path(row['dataset_task_root'])/row['evidence_path']
    states={s['frame']:s for s in map(json.loads,(ev/'states.jsonl').read_text().splitlines())}
    order=states[0].get('open_order')
    if (ev/'geometry.json').exists():
        geometry_data=read(ev/'geometry.json')
        if order is None:order=geometry_data['open_order']
        for frame in geometry_data['frames']:
            state=states[frame['frame']]
            if 'covers' not in state['poses']:
                state['poses']=dict(covers=[dict(p=p) for p in frame['covers']],blocks=[dict(p=p) for p in frame['blocks']])
            if 'tool_poses' not in state:state['tool_poses']=frame['tool_poses']
    if order is None:order=read(ev/'result.json')['memory']['color_order_positions']
    def covered(s):
        return [abs(c['p'][0]-b['p'][0])<.03 and abs(c['p'][1]-b['p'][1])<.03 and c['p'][2]<.742
                for c,b in zip(s['poses']['covers'],s['poses']['blocks'])]
    covered_states={i:covered(s) for i,s in states.items()}
    start=next((i for i,c in covered_states.items() if all(c)),None)
    samples=[json.loads(line) for line in (cache/'samples.jsonl').read_text().splitlines()]
    mapping=read(cache/'source_frames.json');assert len(mapping)==len(samples)
    matched=[j for j,m in enumerate(mapping) if start is not None and m['source_frame']>start
             and covered_states[m['source_frame']]==[i==order[2] for i in range(3)]]
    return dict(states=states,samples=samples,mapping=mapping,matched=matched)


def geometry(state):
    wrists=state.get('tool_poses')
    if wrists is None:wrists=[state['left_ee'],state['right_ee']]
    return np.array([v for p in state['poses']['covers'] for v in p['p']]+[v*.5 for p in wrists for v in p[:3]])


def prepare():
    assert not OUT.exists();OUT.mkdir()
    os.link(Path('/cache/cvpzhr/evaluations/forest-s01-history-swap-step23695-20261109/checkpoint.pt'),OUT/'checkpoint.pt')
    assert torch.load(OUT/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)['step']==23695
    rows=read(SETUP/'split.json')['records'];cfg=read(SETUP/'config.json')
    normal=set(cfg['normal_half_sampling']);counts={};donors={};donor_data={}
    successes=[r for r in rows if r['trajectory_source']=='vla' and r['task_success']]
    for layout in ['012','021','102','120','201','210']:
        for row in sorted((r for r in successes if r['layout']==layout),
                          key=lambda r:(r['split']!='validation',r['category']!='fa_success',r['episode_index'])):
            data=load(row)
            if any(m['pause_copy'] for m in data['mapping']):continue
            candidates=[i for i in data['matched'] if i>0 and
                data['states'][data['mapping'][i-1]['source_frame']]['current_state_pointer']>=5]
            if candidates:
                data['candidates']=candidates;donors[layout]=row;donor_data[layout]=data;break
        assert layout in donors,layout
    plan=[];selected={str(r['episode_index']):r for r in donors.values()}
    for row in rows:
        if row['split']!='train' and row['category']!='script_success':continue
        data=load(row);matched=data['matched'];n=len(data['samples'])
        if row['split']=='train' and matched:
            fraction=(n//2)/n if row['cache_dir'] in normal else 1.
            for label in ['zero','nonzero']:
                m=sum((data['samples'][i]['pt']==0)==(label=='zero') for i in matched)
                if not m:continue
                key=(row['layout'],row['trajectory_source'],row['category'],label)
                c=counts.setdefault(key,dict(trajectories=0,cached_observations=0,expected_observations_after_sampling=0.))
                c['trajectories']+=1;c['cached_observations']+=m;c['expected_observations_after_sampling']+=m*fraction
        failure=row['split']=='train' and row['trajectory_source']=='scripted' and row['hard102']
        success=row['category']=='script_success'
        if not failure and not success:continue
        assert matched,(row['episode_index'],'no matched intermediate state')
        if failure:
            eligible=[i for i in matched if not data['mapping'][i]['pause_copy']]
            targets=sorted({eligible[round((len(eligible)-1)*f)] for f in [.2,.5,.8]})
            assert len(targets)==3
        else:targets=matched
        donor=donors[row['layout']];dd=donor_data[row['layout']]
        for index in targets:
            raw=data['mapping'][index]['source_frame'];state=data['states'][raw]
            if failure:assert state['fail_flag'] and data['samples'][index]['pt']==0
            else:assert not state['fail_flag']
            match=min(dd['candidates'],key=lambda i:np.linalg.norm(geometry(dd['states'][dd['mapping'][i]['source_frame']])-geometry(state)))
            plan.append(dict(case_id=len(plan),test='script_failure_history_swap' if failure else 'script_success_native',
                target_episode=row['episode_index'],target_index=index,source_frame=raw,
                virtual_frame=data['samples'][index]['frame_index'],layout=row['layout'],split=row['split'],
                donor_episode=donor['episode_index'],donor_index=match,
                geometry_distance=float(np.linalg.norm(geometry(dd['states'][dd['mapping'][match]['source_frame']])-geometry(state)))))
        selected[str(row['episode_index'])]=row
    assert len({p['target_episode'] for p in plan if p['test']=='script_failure_history_swap'})==104
    assert len({p['target_episode'] for p in plan if p['test']=='script_success_native'})==6
    save(OUT/'distribution.json',[dict(layout=k[0],source=k[1],category=k[2],label=k[3],**v) for k,v in sorted(counts.items())])
    save(OUT/'protocol.json',dict(rows=selected,plan=plan,checkpoint_step=23695,
        checkpoint_sha256=hashlib.sha256((OUT/'checkpoint.pt').read_bytes()).hexdigest(),batch_size=2,
        matched_state='After the first fully-covered state: red and green uncovered, only blue covered. Physical source frames mapped through pause augmentation.',
        target_selection='All 104 training script102 failures, 3 quantiles of matching non-pause frames each; all matching cached frames of all 6 v4 script successes (5 train,1 val).',
        correct_history='Same-layout VLA success, no pause augmentation, same cover state; history has native correct first-two opens completed (previous sample pointer>=5). Match nearest lid/wrist geometry without inspecting predictions.',
        distribution='Training observations in that intermediate state only; expected sampling retains floor(N/2)/N of each normal trajectory, all recovery/abnormal observations; not a calibrated model probability.',
        caveat='Synthetic history swaps have no counterfactual GT. Low swap sensitivity alone does not prove an image-only rule; scripted-success controls are required.',
        training_started=False))
    shutil.copy2(Path(__file__),OUT/'script_style_probe.py')
    shutil.copy2(Path(__file__).with_name('history_swap_current.py'),OUT/'history_swap_current.py')
    save(OUT/'status.json',dict(phase='running',cases=len(plan),script_failure_episodes=104,script_success_episodes=6,training_started=False,updated=time.time()))


def worker(rank):
    torch.set_num_threads(3);torch.manual_seed(42)
    protocol=read(OUT/'protocol.json');rows={int(k):v for k,v in protocol['rows'].items()}
    model,processor,cfg=load_checkpoint(str(OUT/'checkpoint.pt'))
    collator=Stage3Collator(processor,cfg['model']['history_dim'],cfg['max_sequence_length'],cfg['history_capacity'])
    datasets={};references={};done=0;started=time.monotonic()
    def ds(ep):
        if ep not in datasets:
            datasets[ep]=M3Dataset([str(Path(rows[ep]['cache_dir'])/'samples.jsonl')])
            assert datasets[ep].history_source==cfg['history_source']
        return datasets[ep]
    def inputs(ep,index,history=None):
        sample=ds(ep)[index];slot=index%2
        if index+1==len(ds(ep)) and slot==0:slot=1
        neighbor=ds(ep)[index-1 if slot else index+1]
        if history is not None:sample=dict(sample,history_tokens=history)
        batch=collator([neighbor,sample] if slot else [sample,neighbor])['inputs']
        return sample,slot,to_device(batch,'cuda')
    def forward(batch,slot):
        r=model(batch);q=r['bin_probs'][slot].float().cpu().numpy();p=float(r['pt'][slot])
        assert np.isfinite(q).all() and abs(q.sum()-1)<1e-6 and abs(q@np.linspace(0,1,101)-p)<1e-6
        return dict(predicted_pt=p,probabilities=q.tolist(),p_bin0=float(q[0]),p_high_80_100=float(q[80:].sum()))
    plan=protocol['plan'][rank::2]
    with torch.inference_mode(),(OUT/f'worker{rank}-results.jsonl').open('w') as f:
        for case in plan:
            ep,index=case['target_episode'],case['target_index']
            sample,slot,base=inputs(ep,index);native=forward(base,slot)
            records=[dict(condition='native_history',native_gt=sample['target_pt'],**native)]
            if case['test']=='script_failure_history_swap':
                de,di=case['donor_episode'],case['donor_index'];donor=ds(de)[di]
                assert donor['target_pt']>0 and sample['target_pt']==0
                _,_,swapped=inputs(ep,index,donor['history_tokens']);verify_inputs(base,swapped)
                records.append(dict(condition='vla_success_history',native_gt=sample['target_pt'],**forward(swapped,slot)))
                if (de,di) not in references:
                    _,dslot,dbatch=inputs(de,di);references[(de,di)]=forward(dbatch,dslot);del dbatch
                records.append(dict(condition='vla_success_reference',native_gt=donor['target_pt'],**references[(de,di)]))
                del swapped
            for record in records:
                images=donor['images'] if record['condition']=='vla_success_reference' else sample['images']
                record.update(case,current_image_sha256=[hashlib.sha256(image.tobytes()).hexdigest() for image in images])
                f.write(json.dumps(record)+'\n')
            f.flush();done+=1
            if done%20==0:print(json.dumps(dict(done=done,total=len(plan),seconds=time.monotonic()-started)),flush=True)
            del base
    save(OUT/f'worker{rank}-status.json',dict(phase='completed',cases=done,seconds=time.monotonic()-started))


def summarize():
    protocol=read(OUT/'protocol.json');rows={int(k):v for k,v in protocol['rows'].items()}
    records=[json.loads(line) for rank in range(2) for line in (OUT/f'worker{rank}-results.jsonl').read_text().splitlines()]
    assert len({r['case_id'] for r in records})==len(protocol['plan'])
    fails=[r for r in records if r['test']=='script_failure_history_swap']
    native={r['case_id']:r for r in fails if r['condition']=='native_history'}
    swapped=[r for r in fails if r['condition']=='vla_success_history']
    refs=[r for r in fails if r['condition']=='vla_success_reference']
    delta=np.array([r['predicted_pt']-native[r['case_id']]['predicted_pt'] for r in swapped])*100
    success=[]
    for ep in sorted({r['target_episode'] for r in records if r['test']=='script_success_native'}):
        selected=[r for r in records if r['test']=='script_success_native' and r['target_episode']==ep]
        success.append(dict(episode=ep,split=rows[ep]['split'],layout=rows[ep]['layout'],samples=len(selected),
            gt_mean_pct=float(np.mean([r['native_gt'] for r in selected])*100),
            pred_mean_pct=float(np.mean([r['predicted_pt'] for r in selected])*100),
            mae_pp=float(np.mean([abs(r['native_gt']-r['predicted_pt']) for r in selected])*100),
            under20_count=sum(r['predicted_pt']<.2 for r in selected),
            under50_count=sum(r['predicted_pt']<.5 for r in selected)))
    report=dict(checkpoint_step=23695,script_failures=dict(episodes=104,frames=len(native),
        native_mean_pct=float(np.mean([r['predicted_pt'] for r in native.values()])*100),
        swapped_mean_pct=float(np.mean([r['predicted_pt'] for r in swapped])*100),
        mean_absolute_delta_pp=float(abs(delta).mean()),max_absolute_delta_pp=float(abs(delta).max()),
        changes_over5pp=int((abs(delta)>5).sum()),swapped_under1pct=sum(r['predicted_pt']<.01 for r in swapped),
        swapped_under20pct=sum(r['predicted_pt']<.2 for r in swapped),
        donor_reference_mean_pct=float(np.mean([r['predicted_pt'] for r in refs])*100)),
        script_successes=success,training_started=False)
    save(OUT/'report.json',report)
    save(OUT/'status.json',dict(phase='completed',cases=len(protocol['plan']),records=len(records),training_started=False,updated=time.time()))
    print(json.dumps(report),flush=True)


def main():
    prepare();jobs=[]
    for rank,gpu in enumerate([2,3]):
        with (OUT/f'worker{rank}.log').open('w') as log:
            jobs.append(subprocess.Popen([PY,'-u',str(OUT/'script_style_probe.py'),'--worker',str(rank)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(SETUP/'code'),
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='3'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert codes==[0,0],codes
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        try:main()
        except Exception as exc:
            if OUT.exists():save(OUT/'status.json',dict(phase='failed',error=repr(exc),updated=time.time()))
            raise
