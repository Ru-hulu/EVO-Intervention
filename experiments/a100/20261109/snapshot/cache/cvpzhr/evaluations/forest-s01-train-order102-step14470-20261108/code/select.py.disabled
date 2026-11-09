import json
from pathlib import Path
ROOT=Path('/cache/cvpzhr/evaluations/forest-s01-train-order102-step14470-20261108')
DATA=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261107-v5-failure-continue/State/01-cover_uncover_blocks')
SPLIT=Path('/cache/cvpzhr/setup/forest-s01-failure-continue-ce-20261107/split.json')
def covered(s):
 p=s['poses'];return [abs(c['p'][0]-b['p'][0])<.03 and abs(c['p'][1]-b['p'][1])<.03 and c['p'][2]<.742 for c,b in zip(p['covers'],p['blocks'])]
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 selected=[];audit=[]
 for r in json.loads(SPLIT.read_text())['records']:
  if r['task_success'] or not r.get('states_path'):continue
  geometry=DATA/r['evidence_path']/'geometry.json'
  if geometry.exists():
   g=json.loads(geometry.read_text());expected=g['open_order']
   states=[dict(frame=f['frame'],fail_flag=f['fail'],poses=dict(covers=[dict(p=x) for x in f['covers']],blocks=[dict(p=x) for x in f['blocks']])) for f in g['frames']]
  else:
   states=[json.loads(l) for l in (DATA/r['states_path']).read_text().splitlines()]
   if not all('covers' in s.get('poses',{}) and 'blocks' in s.get('poses',{}) for s in states):
    audit.append(dict(episode=r['episode_index'],skipped='missing cover/block pose evidence'));continue
   if 'open_order' in states[0]:expected=states[0]['open_order']
   else:
    colors=r.get('color_ids_left_to_right')
    if colors is None:continue
    expected=[colors.index(i) for i in range(3)]
  cov=[covered(s) for s in states]
  start=next((i for i,c in enumerate(cov) if all(c)),None)
  if start is None:continue
  lift=[(next((i for i in range(start+1,len(states)-4) if all(not cov[t][k] for t in range(i,i+5))),len(states)),k) for k in range(3)]
  actual=[k for i,k in sorted(lift) if i<len(states)]
  rank_order=[expected.index(k) for k in actual]
  matches=[s['frame'] for s,c in zip(states,cov) if s['fail_flag'] and c==[k==expected[2] for k in range(3)]]
  record=dict(episode=r['episode_index'],split=r['split'],expected=expected,actual=actual,rank_order=rank_order,matched_original_frames=len(matches))
  audit.append(record)
  if rank_order[:2]==[1,0] and matches and r['split']=='train':
   mapping=json.loads((Path(r['cache_dir'])/'source_frames.json').read_text())
   selected.append(dict(row=r,**record,matched_frames=matches,training_pause_copies=sum(x['pause_copy'] for x in mapping)))
 (ROOT/'selection.json').write_text(json.dumps(dict(records=selected,audit=audit,rule='Native cover predicate; first uncover order begins 102 (five consecutive uncovered frames); native failure true and only blue covered; train only'),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps([{k:v for k,v in s.items() if k not in ['row','matched_frames']} for s in selected],indent=2))
 assert len(selected)==9, 'Audit discrepancy; inspect before inference'
if __name__=='__main__':main()
