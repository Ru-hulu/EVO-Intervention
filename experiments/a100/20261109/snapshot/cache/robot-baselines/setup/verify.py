"""Recompute every progress value from saved poses and native event evidence."""
import json
import math
import sys
from pathlib import Path


def near(p, target, x=.03, y=.03, z=.77):
    return abs(p[0]-target[0])<x and abs(p[1]-target[1])<y and p[2]<z


def multiply(a,b):
    w,x,y,z=a;v,i,j,k=b
    return [w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v]


def angle(a,b):
    dot=abs(sum(x*y for x,y in zip(a,b)))
    dot/=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return 2*math.acos(min(1,dot))


root=Path(sys.argv[1]);checks=[]
for row in map(json.loads,(root/'metadata.jsonl').read_text().splitlines()):
    folder=root/row['sample_id']
    result=json.loads((folder/'result.json').read_text())
    progress=json.loads((root/'progress'/(row['sample_id']+'.json')).read_text())
    states=[json.loads(x) for x in (folder/'states.jsonl').read_text().splitlines()]
    task=result['task'];memory=result['memory']
    for index,s in enumerate(states):
        poses=s['poses']
        if task=='rearrange_blocks':
            expected=[near(poses['block1']['p'],memory['A_target']),s['stage_id']>=1,
                      near(poses['block2']['p'],[.13,memory['middle'][1]])]
        elif task=='swap_blocks':
            expected=[near(poses['block1']['p'],memory['B_home'],.04,.06,.765),
                      near(poses['block2']['p'],memory['A_home'],.04,.06,.765),s['native_success']]
            assert s['flag'], 'Native irreversible collision flag'
        elif task=='swap_T':
            expected=[]
            for key,target in [('T_block1',memory['red_target']),('T_block2',memory['blue_target'])]:
                actual=poses[key]
                q=multiply(target[3:],[math.sqrt(.5),math.sqrt(.5),0,0])
                expected.append(math.dist(actual['p'][:2],target[:2])<.025 and actual['p'][2]<.77
                                and angle(actual['q'],q)<math.radians(15))
        elif task=='battery_try':
            expected=[]
            for i,key in enumerate(['battery1','battery2']):
                actual=poses[key];q=actual['q']
                orientation=int(abs(q[2]-q[3])>abs(q[0]-q[1]))
                expected.append(math.dist(actual['p'][:2],s[f'target_pose{i+1}_p'][:2])<.02
                                and orientation==s['correct_combination'][i] and actual['p'][2]<.79)
        else:
            expected=[s['flag'][i]=='True' and near(a['p'],memory['homes'][i])
                      for i,a in enumerate(poses['blocks_lst'])]
        assert expected==s['atoms'],(task,index,expected,s['atoms'])
        assert abs(sum(expected)/len(expected)-progress['dense_progress'][index])<1e-10
    assert len(states)==row['total_frames']==len(progress['dense_progress'])
    assert all(a['physics_step']<b['physics_step'] for a,b in zip(states,states[1:]))
    assert result['native_success'] and result['error'] is None and states[-1]['progress']==1
    checks.append(dict(sample_id=row['sample_id'],frames_checked=len(states),all_progress_recomputed=True,
                       native_success=True,recording_tail=result.get('recording_tail')))
report=dict(episodes=len(checks),frames_checked=sum(x['frames_checked'] for x in checks),checks=checks)
(root/'progress-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
