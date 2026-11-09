"""Shared-success-weight visual integral over evidenced subtask boundaries.

Generated labels remain review-required. Failed unfinished intervals are masked,
not normalized to success. Simulation predicates define boundaries, not slopes.
"""
import hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/cache/robot-baselines');OUT=ROOT/'annotations/boundary-integral-20261105'
DATASETS=['ProPriMem-VLA-Failures-20261104','RoboMME-FrameSamp-20261105','RoboMME-SimpleMem-20261105']
POLICY='boundary_visual_integral_v4_shared_success_reference'
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False));tmp.replace(p)
def inside(p,t):return max(abs(p[0]-t[0]),abs(p[1]-t[1]))<.03 and p[2]<.77

def rm_atoms(row,root):
 states=[json.loads(l) for l in (Path(row['evidence_path'])/'states.jsonl').read_text().splitlines()]
 assert len(states)==row['total_frames']
 task=row['task_id'];goals=[];types={};history=False;sets=[]
 if task=='C06':
  goals=[f'{side}:{i}' for side,num in [('left',states[0]['card_id_1']),('middle',states[0]['card_id_2']),('confirm',1)] for i in range(1,num+1)]
 elif task=='S01':goals=[f'phase:{i}' for i in range(1,len(states[0]['target_state_transition']))]
 elif task in ['S02','S06']:goals=['first_place','confirm','final_place']
 elif task=='S03':goals=[f'{kind}:{i}' for i in range(3) for kind in ['visit_green','return_home']]
 elif task=='S04':goals=['battery:0','battery:1','success_confirmation']
 elif task=='S05':goals=['order:01','order:12','success_confirmation']
 for g in goals:types[g]=g.split(':')[0] if task in ['C06','S03','S04'] else g
 frozen=False
 for s in states:
  p=s['poses'];done=set()
  if task=='C06':
   counts=[s['press_cnt_1'],s['press_cnt_2'],s['press_cnt_check_button']];targets=[s['card_id_1'],s['card_id_2'],1]
   frozen=frozen or any(a>b for a,b in zip(counts,targets))
   if not frozen:done={f'{kind}:{i}' for kind,c in zip(['left','middle','confirm'],counts) for i in range(1,c+1)}
  elif task=='S01':
   frozen=frozen or bool(s['fail_flag'])
   if not frozen:done=set(goals[:s['current_state_pointer']])
  elif task in ['S02','S06']:
   history=history or (inside(p['block'][0]['p'],[.1,-.1]) if task=='S02' else inside(p['block1'][0]['p'],p['final_occupy_mat'][0]['p']))
   if history:done.add('first_place')
   if s['stage_id']>=1:done.add('confirm')
   if s['success']:done.add('final_place')
  elif task=='S03':
   for i,flag in enumerate(s['flag']):
    if flag=='True':
     done.add(f'visit_green:{i}')
     if inside(p['blocks_lst'][i]['p'],p['mat_lst'][int(s['selected_idx'][i])]['p']):done.add(f'return_home:{i}')
  elif task=='S04':
   for i in range(2):
    b=p[f'battery{i+1}'][0];q=np.asarray(b['q']);orientation=int(abs(q@np.array([0,0,.707,-.707]))>abs(q@np.array([.707,-.707,0,0])))
    if np.linalg.norm(np.asarray(b['p'][:2])-s[f'target_pose{i+1}_p'][:2])<.02 and orientation==s['correct_combination'][i]:done.add(f'battery:{i}')
   if s['success']:done=set(goals)
  elif task=='S05':
   bs=[p[f'block{i}'][0]['p'] for i in range(1,4)]
   for a,b,key in [(0,1,'order:01'),(1,2,'order:12')]:
    if 0<bs[b][0]-bs[a][0]<.13 and abs(bs[b][1]-bs[a][1])<.04:done.add(key)
   if s['success']:done=set(goals)
  sets.append(done)
 # Confirm changes over 3 frames to suppress contact-threshold flicker.
 # Final native success is authoritative even if recording ends immediately.
 stable=set(sets[0]);filtered=[]
 for f,raw in enumerate(sets):
  if f+3<=len(sets) and all(raw==sets[j] for j in range(f,f+3)):stable=set(raw)
  if states[f]['success']:stable=set(goals)
  filtered.append(set(stable))
 events=[dict(frame=0,done=sorted(filtered[0]))]
 for f,done in enumerate(filtered[1:],1):
  if done!=filtered[f-1]:events.append(dict(frame=f,done=sorted(done)))
 return dict(goals=goals,types=types,events=events,start=0,source='recorded object poses and native task predicates; 3-frame change confirmation',source_sha256=hashlib.sha256((Path(row['evidence_path'])/'states.jsonl').read_bytes()).hexdigest())

def mme_atoms(row,replay):
 tasklist=replay['task_list'];start_index=replay['events'][0]['index']
 required=[t for t in tasklist if t['index']>=start_index and not t['demonstration']]
 goals=[str(t['index']) for t in required]
 # Exact subtask name; no guessed semantic equivalence across names.
 types={str(t['index']):t['name'] for t in required}
 events=[dict(frame=e['frame'],done=[g for g in goals if int(g)<e['index']]) for e in replay['events']]
 return dict(goals=goals,types=types,events=events,start=replay['execution_start'],source='saved-action replay; every joint state and terminal status verified',max_qpos_error=replay['max_qpos_error'])

def integrate(delta,boundary,weights,success):
 n=len(delta);events=boundary['events'];goals=boundary['goals'];assert goals and set(weights)==set(goals)
 assert all(math.isfinite(x) and x>=0 for x in delta)
 total=sum(weights.values());assert total>0
 w={g:weights[g]/total for g in goals};p=[0.]*n;stages=[];previous=set(events[0]['done']);start=events[0]['frame'];value=sum(w[g] for g in previous)
 p[:start+1]=[value]*(start+1)
 for event in events[1:]:
  end=event['frame'];done=set(event['done']);gained=done-previous;lost=previous-done
  assert end>start
  mass=sum(delta[start+1:end+1]);gain=sum(w[g] for g in gained);loss=sum(w[g] for g in lost)
  if gain and mass<=1e-8:raise ValueError('Completed interval has zero visual delta')
  cumulative=0.
  for f in range(start+1,end+1):
   cumulative+=delta[f];p[f]=value+(gain*cumulative/mass if gain else 0.)
  # Match lite-mem: remove invalidated contribution at evidenced reset frame.
  p[end]-=loss
  stages.append(dict(start_frame=start,end_frame=end+1,gained=sorted(gained),rollback_ids=sorted(lost),delta_sum=mass))
  value=sum(w[g] for g in done);assert abs(p[end]-value)<1e-8
  previous=done;start=end
 p[start+1:]=[value]*(n-start-1)
 if success:assert previous==set(goals) and abs(p[-1]-1)<1e-8
 if not success and p[-1]>=1-1e-8:raise ValueError('Failure has all goals credited; boundary review required')
 assert len(p)==n and all(math.isfinite(v) and -1e-8<=v<=1+1e-8 for v in p)
 return p,w,stages

def main():
 rows=[];pending=[];references=defaultdict(list)
 for ds in DATASETS:
  root=ROOT/'viewer-datasets'/ds
  for row in map(json.loads,(root/'metadata.jsonl').read_text().splitlines()):
   key=row['sample_id'];dp=OUT/ds/'deltas'/(key+'.json');rp=OUT/ds/'replay'/(key+'.json')
   if not dp.exists() or (ds!=DATASETS[0] and not rp.exists()):pending.append(dict(dataset=ds,sample=key,reason='waiting for deltas or verified replay'));continue
   delta=read(dp)
   try:
    boundary=rm_atoms(row,root) if ds==DATASETS[0] else mme_atoms(row,read(rp))
    assert boundary['goals'];assert delta['frames']==row['total_frames']
    write(OUT/ds/'boundaries'/(key+'.json'),dict(sample_id=key,**boundary))
    family='RMBench' if ds==DATASETS[0] else 'RoboMME'
    rows.append((ds,row,delta,boundary,family))
    if row['task_success'] and row.get('reference_eligible',True):
     # Last successful completion of each atom is the reference, as in lite-mem.
     values={};previous=set(boundary['events'][0]['done']);start=boundary['start']
     for event in boundary['events'][1:]:
      gained=set(event['done'])-previous;end=event['frame'];mass=math.fsum(delta['deltas'][start+1:end+1])
      for atom in gained:
       if mass>1e-8:values[atom]=mass/len(gained)
      previous=set(event['done']);start=end
     for atom,mass in values.items():references[(family,row['task_id'],boundary['types'][atom])].append(dict(mass=mass,dataset=ds,sample=key,atom=atom))
   except Exception as e:pending.append(dict(dataset=ds,sample=key,reason=str(e)))
 published=[]
 for ds,row,delta,b,family in rows:
  key=row['sample_id']
  try:
   refs={g:references[(family,row['task_id'],b['types'][g])] for g in b['goals']}
   missing=[g for g,r in refs.items() if not r]
   if missing:raise ValueError('No successful reference for '+str(missing))
   weights={g:statistics.median(r['mass'] for r in refs[g]) for g in b['goals']}
   curve,w,stages=integrate(delta['deltas'],b,weights,row['task_success'])
   unknown=[] if row['task_success'] else [[b['events'][-1]['frame'],row['total_frames']]]
   public_events=[]
   previous=set()
   for e in b['events']:
    done=set(e['done']);lost=previous-done
    public_events.append(dict(frame=e['frame'],kind='regression' if lost else 'complete',progress=curve[e['frame']],rollback_ids=sorted(lost)))
    previous=done
   p=dict(policy=POLICY,total=len(b['goals']),dense_progress=curve,weights=w,points=[[e['frame'],len(e['done'])] for e in b['events']],events=public_events,first_issue=next((e for e in public_events if e['kind']=='regression'),None),source='成功参考共享权重；子任务区间内 Qwen 视觉 delta 积分；失效帧扣回贡献',reference_samples=refs,unresolved_intervals=unknown,preview_only=True,training_eligible=False,normalization='median successful completion visual mass per task and exact subtask type; not per failed trajectory',boundary_source=b['source'],vision_sha256=delta['vision_sha256'],video_sha256=delta['video_sha256'],reference_commit='0a63f18eb35cfede460c29d56d870271b611a046')
   out=OUT/ds/'progress'/(key+'.json');write(out,p);published.append(dict(dataset=ds,sample=key,frames=len(curve),final=curve[-1],success=row['task_success'],unknown_intervals=unknown))
  except Exception as e:pending.append(dict(dataset=ds,sample=key,reason=str(e)))
 write(OUT/'integration-status.json',dict(prepared=len(published),pending=len(pending),results=published,issues=pending,publication='not published; boundary and reference review required'))
 print('prepared',len(published),'pending',len(pending),flush=True)
if __name__=='__main__':main()
