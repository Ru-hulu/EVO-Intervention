"""Fixed RGB/instruction, correct versus wrong native histories: FOREST step23695."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device

OUT = Path('/cache/cvpzhr/evaluations/forest-s01-history-swap-step23695-20261109')
SETUP = Path('/cache/cvpzhr/setup/forest-s01-order102-expand120-20261109')
TARGETS = [55,203,205]
DONORS = [10,0]
TIMES = [36,38,40,42,44,45,46,48]
PY = '/cache/cvpzhr/envs/forest/bin/python'


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def states(row):
    path=Path(row['dataset_task_root'])/row['evidence_path']/'states.jsonl'
    return {s['frame']:s for s in map(json.loads,path.read_text().splitlines())}


def covered(state):
    return [abs(c['p'][0]-b['p'][0])<.03 and abs(c['p'][1]-b['p'][1])<.03 and c['p'][2]<.742
            for c,b in zip(state['poses']['covers'],state['poses']['blocks'])]


def geometry(state):
    return np.array([v for pose in state['poses']['covers'] for v in pose['p']]+
        [v*.5 for pose in state['tool_poses'] for v in pose[:3]])


def prepare():
    assert not OUT.exists(), 'Keep each diagnostic run immutable'
    OUT.mkdir()
    checkpoint=OUT/'checkpoint.pt'
    shutil.copy2(Path('/cache/cvpzhr/outputs/forest-s01-order102-expand120-20261109/best.pt'),checkpoint)
    assert torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['step']==23695
    rows=read(SETUP/'split.json')['records']
    selected={str(ep):rows[ep] for ep in TARGETS+DONORS}
    cached={ep:[json.loads(line) for line in (Path(rows[ep]['cache_dir'])/'samples.jsonl').read_text().splitlines()]
            for ep in TARGETS+DONORS}
    ss={ep:states(rows[ep]) for ep in TARGETS+DONORS}
    for ep in TARGETS+DONORS:
        assert rows[ep]['layout']=='201'
        mapping=read(Path(rows[ep]['cache_dir'])/'source_frames.json')
        assert not any(m['pause_copy'] for m in mapping)
        assert ss[ep][0]['open_order']==[1,2,0]
        assert len({s['frame_index'] for s in cached[ep]})==len(cached[ep])
    assert all(not rows[ep]['task_success'] for ep in TARGETS)
    assert all(rows[ep]['task_success'] and not any(s['fail_flag'] for s in ss[ep].values()) for ep in DONORS)
    plan=[]
    for target in TARGETS:
        for seconds in TIMES:
            frame=min((s['frame_index'] for s in cached[target]),key=lambda f:abs(f/rows[target]['fps']-seconds))
            current=ss[target][frame]
            assert current['fail_flag']
            donor_frames={};distances={}
            for donor in DONORS:
                assert rows[target]['task_prompt']==rows[donor]['task_prompt']
                candidates=[s['frame_index'] for j,s in enumerate(cached[donor]) if j>0 and s['history'] is not None
                    and ss[donor][cached[donor][j-1]['frame_index']]['current_state_pointer']>=5
                    and covered(ss[donor][s['frame_index']])==covered(current)]
                assert candidates, (target,frame,donor,covered(current))
                match=min(candidates,key=lambda f:np.linalg.norm(geometry(ss[donor][f])-geometry(current)))
                donor_frames[str(donor)]=match
                distances[str(donor)]=float(np.linalg.norm(geometry(ss[donor][match])-geometry(current)))
            plan.append(dict(target_episode=target,current_frame=frame,current_seconds=frame/rows[target]['fps'],
                donor_frames=donor_frames,geometry_distances=distances,covered=covered(current)))
    protocol=dict(checkpoint=str(checkpoint),checkpoint_step=23695,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        rows=selected,plan=plan,batch_size=2,training_started=False,
        primary_correct_history='episode10: held-out validation, recurrent VLA success, same layout and phase',
        auxiliary_correct_history='episode0: training-set full-attention VLA success, same layout and phase',
        selection='Fixed times 36,38,40,42,44,45,46,48 seconds; donor history must already contain native completion of the correct first two uncover actions (previous sample state pointer >=5). Minimize cover/wrist pose distance among identical cover states; predictions not used for matching.',
        current_inputs='Current three RGB views and task instruction held fixed within each direction. Both failure-view and matched-success-view contrasts. Batch2 uses the original adjacent cached observation as a fixed second example; only the target example history changes.',
        controls='Repeat the original history. Also replace correct history at the exact wrong-history token count using adaptive mean resampling; auxiliary OOD control only.',
        limitations='Spliced histories are synthetic counterfactual inputs, not replayed trajectories with validated GT. Natural history replacement can change token count/positions. Auxiliary resampling is not a natural memory snapshot. All targets are known difficult cases; this is not an unbiased population estimate.')
    save(OUT/'protocol.json',protocol)
    shutil.copy2(Path(__file__),OUT/'history_swap_current.py')
    save(OUT/'status.json',dict(phase='running',targets=TARGETS,pairs=len(plan),updated=time.time()))


def verify_inputs(base,other):
    for key in ['pixel_values','image_grid_thw']:
        assert torch.equal(base[key],other[key]),key
    def text(inputs):
        return inputs['input_ids'][inputs['attention_mask'].bool() & ~inputs['history_mask']]
    assert torch.equal(text(base),text(other))


def worker(target):
    torch.set_num_threads(3);torch.manual_seed(42)
    protocol=read(OUT/'protocol.json')
    rows={int(ep):row for ep,row in protocol['rows'].items()}
    datasets={ep:M3Dataset([str(Path(rows[ep]['cache_dir'])/'samples.jsonl')]) for ep in [target,*DONORS]}
    indices={ep:{s['frame_index']:i for i,(_,s,_) in enumerate(ds.samples)} for ep,ds in datasets.items()}
    model,processor,cfg=load_checkpoint(str(OUT/'checkpoint.pt'))
    assert not model.training
    assert all(ds.history_source==cfg['history_source'] and ds.history_capacity==cfg['history_capacity'] for ds in datasets.values())
    collator=Stage3Collator(processor,cfg['model']['history_dim'],cfg['max_sequence_length'],cfg['history_capacity'])
    plan=[p for p in protocol['plan'] if p['target_episode']==target]
    count=0;started=time.monotonic()
    with torch.inference_mode(),(OUT/f'ep{target}-results.jsonl').open('w') as output:
        for pair_index,pair in enumerate(plan):
            bad=datasets[target][indices[target][pair['current_frame']]]
            good={ep:datasets[ep][indices[ep][pair['donor_frames'][str(ep)]]] for ep in DONORS}
            assert bad['target_pt']==0 and all(s['target_pt']>0 for s in good.values())
            # The primary success donor is also the fixed success-image control.
            for direction,current in [('failure_view',bad),('success_view',good[10])]:
                current_episode=target if direction=='failure_view' else 10
                current_index=indices[current_episode][current['frame_index']]
                slot=current_index%2
                if current_index+1==len(datasets[current_episode]) and slot==0:
                    slot=1
                neighbor=datasets[current_episode][current_index-1 if slot else current_index+1]
                def batch(tokens):
                    sample=dict(current,history_tokens=tokens)
                    return to_device(collator([neighbor,sample] if slot else [sample,neighbor])['inputs'],'cuda')
                wrong=batch(bad['history_tokens'])
                right=batch(good[10]['history_tokens'])
                right_fa=batch(good[0]['history_tokens'])
                verify_inputs(wrong,right);verify_inputs(wrong,right_fa)
                n=len(bad['history_tokens'])
                fixed=dict(wrong,history_tokens=wrong['history_tokens'].clone())
                fixed['history_tokens'][slot]=0
                tokens=good[10]['history_tokens'].to('cuda').float()
                fixed['history_tokens'][slot,:n]=F.adaptive_avg_pool1d(tokens.T.unsqueeze(0),n)[0].T.to(fixed['history_tokens'].dtype)
                assert all(torch.equal(wrong[k],fixed[k]) for k in wrong if k!='history_tokens')
                baseline=wrong if direction=='failure_view' else right
                image_hashes=[hashlib.sha256(image.tobytes()).hexdigest() for image in current['images']]
                variants=[('wrong_history',wrong),('correct_history_val',right),('correct_history_fa_train',right_fa),
                          ('original_repeat',baseline),('correct_history_fixed_length_aux',fixed)]
                original_probabilities=None;repeat_probabilities=None
                for condition,inputs in variants:
                    result=model(inputs);probabilities=result['bin_probs'][slot].float().cpu().numpy();prediction=float(result['pt'][slot])
                    assert np.isfinite(probabilities).all() and (probabilities>=0).all()
                    assert abs(probabilities.sum()-1)<1e-6
                    assert abs(probabilities@np.linspace(0,1,101)-prediction)<1e-6
                    baseline_condition='wrong_history' if direction=='failure_view' else 'correct_history_val'
                    if condition==baseline_condition:original_probabilities=probabilities.copy()
                    if condition=='original_repeat':repeat_probabilities=probabilities.copy()
                    pair_info={k:v for k,v in pair.items() if k!='current_frame'}
                    record=dict(pair_index=pair_index,**pair_info,direction=direction,condition=condition,
                        current_episode=current_episode,current_frame=current['frame_index'],
                        fixed_neighbor_frame=neighbor['frame_index'],target_batch_slot=slot,
                        current_native_gt=current['target_pt'],current_view_sha256=image_hashes,
                        wrong_history_tokens=n,correct_history_tokens=len(good[10]['history_tokens']),
                        auxiliary_fa_history_tokens=len(good[0]['history_tokens']),predicted_pt=prediction,
                        probabilities=probabilities.tolist(),p_bin0=float(probabilities[0]),
                        p_low_0_10=float(probabilities[:11].sum()),p_high_80_100=float(probabilities[80:].sum()),
                        max_bin=int(np.argmax(probabilities)),max_bin_probability=float(probabilities.max()))
                    output.write(json.dumps(record)+'\n');output.flush();count+=1
                assert np.array_equal(original_probabilities,repeat_probabilities), 'Repeated original input is not deterministic'
                print(json.dumps(dict(target=target,pair=pair_index,direction=direction,done=count,total=len(plan)*10,seconds=time.monotonic()-started)),flush=True)
                del wrong,right,right_fa,fixed,baseline,variants,inputs,result
    save(OUT/f'ep{target}-status.json',dict(phase='completed',records=count,seconds=time.monotonic()-started,
        fixed_images_verified=True,nonhistory_text_verified=True,repeat_probabilities_exact=True))


def distance(a,b):
    a=np.asarray(a);b=np.asarray(b);m=(a+b)/2
    def kl(p,q):
        mask=p>0
        return float(np.sum(p[mask]*np.log2(p[mask]/q[mask])))
    return dict(total_variation=float(abs(a-b).sum()/2),js_bits=(kl(a,m)+kl(b,m))/2)


def summarize():
    records=[json.loads(line) for ep in TARGETS for line in (OUT/f'ep{ep}-results.jsonl').read_text().splitlines()]
    assert len(records)==240
    contrasts=[]
    for ep in TARGETS:
        for direction in ['failure_view','success_view']:
            for index in range(len(TIMES)):
                group={r['condition']:r for r in records if r['target_episode']==ep and r['direction']==direction and r['pair_index']==index}
                assert len({tuple(r['current_view_sha256']) for r in group.values()})==1
                wrong=group['wrong_history']
                for condition in ['correct_history_val','correct_history_fa_train','correct_history_fixed_length_aux']:
                    right=group[condition]
                    contrasts.append(dict(target_episode=ep,direction=direction,current_seconds=wrong['current_seconds'],
                        condition=condition,wrong_p=wrong['predicted_pt'],correct_p=right['predicted_pt'],
                        delta_pp=(right['predicted_pt']-wrong['predicted_pt'])*100,
                        wrong_bin0=wrong['p_bin0'],correct_bin0=right['p_bin0'],
                        wrong_high=wrong['p_high_80_100'],correct_high=right['p_high_80_100'],
                        **distance(wrong['probabilities'],right['probabilities'])))
    summaries=[]
    for ep in TARGETS:
        for direction in ['failure_view','success_view']:
            for condition in ['correct_history_val','correct_history_fa_train','correct_history_fixed_length_aux']:
                selected=[r for r in contrasts if r['target_episode']==ep and r['direction']==direction and r['condition']==condition]
                summaries.append(dict(target_episode=ep,direction=direction,condition=condition,n=len(selected),
                    mean_delta_pp=float(np.mean([r['delta_pp'] for r in selected])),
                    mean_absolute_delta_pp=float(np.mean([abs(r['delta_pp']) for r in selected])),
                    max_absolute_delta_pp=max(abs(r['delta_pp']) for r in selected),
                    changes_over5pp=sum(abs(r['delta_pp'])>5 for r in selected),
                    mean_total_variation=float(np.mean([r['total_variation'] for r in selected]))))
    save(OUT/'contrasts.json',contrasts);save(OUT/'summary.json',dict(groups=summaries,
        focus45s=[r for r in records if r['current_seconds']==45 and r['condition']!='original_repeat']))
    save(OUT/'status.json',dict(phase='completed',records=len(records),contrasts=len(contrasts),
        targets=TARGETS,training_started=False,updated=time.time()))


def launch(targets=None):
    if targets is None:
        prepare();targets=TARGETS
    jobs=[]
    for gpu,ep in enumerate(targets):
        with (OUT/f'ep{ep}.log').open('w') as log:
            jobs.append(subprocess.Popen([PY,'-u',str(OUT/'history_swap_current.py'),'--target',str(ep)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(SETUP/'code'),
                    OMP_NUM_THREADS='3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert codes==[0]*len(targets),codes
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=int);parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--retry-targets',nargs='+',type=int);args=parser.parse_args()
    if args.target is not None:
        worker(args.target)
    elif args.summarize_only:
        summarize()
    else:
        try:launch(args.retry_targets)
        except Exception as exc:
            if OUT.exists():save(OUT/'status.json',dict(phase='failed',error=repr(exc),updated=time.time()))
            raise
