"""Count failed trajectories revisiting observed successful uncover configurations."""
import json,collections
from pathlib import Path
ROOT=Path('/cache/cvpzhr/evaluations/forest-s01-failure-state-statistics-20261108')
DATA=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261107-v5-failure-continue/State/01-cover_uncover_blocks')
SETUP=Path('/cache/cvpzhr/setup/forest-s01-failure-continue-ce-20261107')
def load(p):return json.loads(p.read_text())
def cov(covers,blocks):return tuple(abs(c[0]-b[0])<.03 and abs(c[1]-b[1])<.03 and c[2]<.742 for c,b in zip(covers,blocks))
def main():
 ROOT.mkdir(exist_ok=True,parents=True)
 records=load(SETUP/'split.json')['records'];episodes={};missing=[]
 for r in records:
  ep=r['episode_index'];ev=DATA/r['evidence_path'];g=ev/'geometry.json'
  if g.exists():
   g=load(g);order=g['open_order'];frames=[dict(frame=f['frame'],fail=f['fail'],phase=f['phase'],covered=cov(f['covers'],f['blocks'])) for f in g['frames']]
  else:
   ss=[json.loads(l) for l in (DATA/r['states_path']).read_text().splitlines()]
   result=load(ev/'result.json');order=result['memory']['color_order_positions']
   frames=[dict(frame=s['frame'],fail=s['fail_flag'],phase=s['current_state_pointer'],covered=cov([x['p'] for x in s['poses']['covers']],[x['p'] for x in s['poses']['blocks']])) for s in ss]
  assert len(frames)==r['total_frames'] and [f['frame'] for f in frames]==list(range(len(frames)))
  episodes[ep]=dict(row=r,order=order,frames=frames)
 # Reference states must actually appear in successful TRAIN trajectories after uncover starts.
 ref=collections.defaultdict(set)
 for e in episodes.values():
  r=e['row']
  if r['split']=='train' and r['task_success']:
   for f in e['frames']:
    if f['phase']>=4 and not f['fail']:ref[tuple(e['order'])].add(f['covered'])
 counts=collections.defaultdict(lambda:dict(episodes=set(),points=0,seconds=0.,original_frames=0,delays=[]))
 per=[];training_points=0;native_failed=[]
 for ep,e in episodes.items():
  r=e['row']
  if r['split']!='train':continue
  frames=e['frames'];fps=r['fps'];order=e['order'];cache=Path(r['cache_dir'])
  mapping=load(cache/'source_frames.json');samples=[json.loads(l) for l in (cache/'samples.jsonl').read_text().splitlines()]
  assert len(samples)==len(mapping);training_points+=len(samples)
  first=next((f['frame'] for f in frames if f['fail']),None)
  if first is not None:native_failed.append(ep)
  targets={'red_open':tuple(i!=order[0] for i in range(3)), 'only_blue_covered':tuple(i==order[2] for i in range(3)), 'all_open':(False,False,False)}
  er=dict(episode=ep,category=r['category'],success=r['task_success'],order=order,first_failure_frame=first,groups={})
  for name,target in targets.items():
   successful_reference=target in ref[tuple(order)]
   for delay in [0,5,10,20]:
    key=f'{name}_delay{delay}'
    ids={f['frame'] for f in frames if first is not None and f['fail'] and (f['frame']-first)/fps>=delay and f['covered']==target and successful_reference}
    pts=[i for i,m in enumerate(mapping) if m['source_frame'] in ids]
    assert all(samples[i]['pt']==0 for i in pts)
    if ids:
     x=counts[key];x['episodes'].add(ep);x['points']+=len(pts);x['original_frames']+=len(ids);x['seconds']+=len(ids)/fps;x['delays'].append((min(ids)-first)/fps)
    if delay==0:
     er['groups'][name]=dict(original_frames=len(ids),training_points=len(pts),seconds=len(ids)/fps,first_delay_seconds=(min(ids)-first)/fps if ids else None,source_range=[min(ids),max(ids)] if ids else None,success_reference=successful_reference)
  per.append(er)
 # Positive examples in the same states, restrict to achieved uncover phase.
 positives={}
 for name in ['red_open','only_blue_covered','all_open']:
  points=0;eps=[];same_layout_points=0;same_layout_eps=[]
  for ep,e in episodes.items():
   r=e['row']
   if r['split']!='train' or not r['task_success']:continue
   order=e['order'];target={'red_open':tuple(i!=order[0] for i in range(3)),'only_blue_covered':tuple(i==order[2] for i in range(3)),'all_open':(False,False,False)}[name]
   valid={f['frame'] for f in e['frames'] if f['phase']>=4 and not f['fail'] and f['covered']==target}
   mapping=load(Path(r['cache_dir'])/'source_frames.json');n=sum(m['source_frame'] in valid for m in mapping)
   if n:eps.append(ep);points+=n
   if n and order==[1,2,0]:same_layout_eps.append(ep);same_layout_points+=n
  positives[name]=dict(episodes=eps,points=points,ep55_layout_episodes=same_layout_eps,ep55_layout_points=same_layout_points)
 result=dict(total_train_trajectories=len(per),training_points=training_points,native_failure_episodes=native_failed,missing=missing,
  definition='Native geometry configuration, not pixel identity. Matched against same-layout successful train trajectories after uncover phase. All delay thresholds are measured from first native failure. Includes actual pause-augmented training inputs.',
  groups={k:dict(v,episodes=sorted(v['episodes']),trajectory_count=len(v['episodes']),training_share_pct=100*v['points']/training_points) for k,v in counts.items()},positive_reference=positives,episodes=per)
 assert training_points==46301
 (ROOT/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
