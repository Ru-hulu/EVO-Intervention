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

OUT=Path('/cache/cvpzhr/evaluations/forest-s01-script-style-probe-small-step23695-20261109')
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
    used=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert used[0]<1000, 'GPU0 is occupied; do not share with training'
    assert not OUT.exists();OUT.mkdir()
    source=Path('/cache/cvpzhr/evaluations/forest-s01-script-style-probe-step23695-20261109')
    original=read(source/'protocol.json');rows={int(k):v for k,v in original['rows'].items()}
    plan=[]
    for layout in ['201','021','012']:
        failed=[p for p in original['plan'] if p['test']=='script_failure_history_swap' and p['layout']==layout
                and rows[p['target_episode']]['sample_id'].startswith('expand120-')]
        ep=min(p['target_episode'] for p in failed)
        choices=sorted((p for p in failed if p['target_episode']==ep),key=lambda p:p['target_index'])
        plan.extend([choices[0],choices[-1]])
        successful=[p for p in original['plan'] if p['test']=='script_success_native' and p['layout']==layout]
        row=rows[successful[0]['target_episode']];data=load(row)
        choices=sorted((p for p in successful if data['states'][p['source_frame']]['current_state_pointer']>=5),key=lambda p:p['target_index'])
        assert len(choices)>=2
        plan.extend([choices[round((len(choices)-1)*.25)],choices[round((len(choices)-1)*.75)]])
    assert len(plan)==12 and len({p['target_episode'] for p in plan})==6
    chosen={p['target_episode'] for p in plan}|{p['donor_episode'] for p in plan}
    original.update(plan=plan,rows={str(ep):rows[ep] for ep in chosen},
        target_selection='SMALL diagnostic only: three newly-added script102 failures and three v4 script successes; two frames each. Layouts 201,021,012. Success points after native stage5. GPU0 exclusive, no training sharing.',
        requested_scope='User stopped bulk diagnostic; only this 12-point subset is evaluated.')
    save(OUT/'protocol.json',original)
    os.link(source/'checkpoint.pt',OUT/'checkpoint.pt')
    shutil.copy2(source/'distribution.json',OUT/'distribution.json')
    shutil.copy2(Path(__file__),OUT/'script_style_probe.py')
    shutil.copy2(Path(__file__).with_name('history_swap_current.py'),OUT/'history_swap_current.py')
    save(OUT/'status.json',dict(phase='running',cases=12,trajectories=6,gpu=0,training_started=False,updated=time.time()))


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
    plan=protocol['plan']
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
    records=[json.loads(line) for rank in range(1) for line in (OUT/f'worker{rank}-results.jsonl').read_text().splitlines()]
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
    report=dict(checkpoint_step=23695,script_failures=dict(episodes=len({r['target_episode'] for r in native.values()}),frames=len(native),
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
    for rank,gpu in enumerate([0]):
        with (OUT/f'worker{rank}.log').open('w') as log:
            jobs.append(subprocess.Popen([PY,'-u',str(OUT/'script_style_probe.py'),'--worker',str(rank)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(SETUP/'code'),
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='3'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert codes==[0],codes
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        try:main()
        except Exception as exc:
            if OUT.exists():save(OUT/'status.json',dict(phase='failed',error=repr(exc),updated=time.time()))
            raise
