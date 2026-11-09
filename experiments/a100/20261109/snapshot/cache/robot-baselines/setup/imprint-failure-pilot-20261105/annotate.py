"""Label unique failures and matched successes after direct transfer finishes."""
from pathlib import Path
import json,time,sys,os,subprocess,math
B=Path('/cache/robot-baselines');OUT=B/'annotations/imprint-failure-pilot-20261105';DS='IMPRINT-Natural-Failures-20261105'
S=B/'setup/model-trajectory-progress';sys.path.insert(0,str(S))
from extract import write
while True:
 counts={};selected=[]
 for model in ['full_attention','recurrent']:
  records=[json.loads(p.read_text()) for p in (B/'eval-results/imprint-failure-pilot-20261105'/model/'cover_blocks').glob('seed_*/result.json')];unique={r['fingerprint'] for r in records if not r['success'] and not r.get('duplicate_failure')}
  counts[model]=dict(attempts=len(records),unique_failures=len(unique))
 write(OUT/'status.json',dict(status='waiting_collection',counts=counts,time=time.time()))
 if all(c['unique_failures']>=30 or c['attempts']>=300 for c in counts.values()):
  metadata=B/'viewer-datasets'/DS/'metadata.jsonl'
  if metadata.exists() and len(metadata.read_text().splitlines())>=sum(c['attempts'] for c in counts.values()):break
 time.sleep(30)
rows=[json.loads(l) for l in (B/'viewer-datasets'/DS/'metadata.jsonl').read_text().splitlines()]
for model in ['full_attention','recurrent']:
 own=[r for r in rows if model in r['sample_id']];seen=set();wins=0
 for r in own:
  x=json.loads((Path(r['evidence_path']).parents[2]/'result.json').read_text())
  if x['success']:
   if wins>=10:continue
   wins+=1
  else:
   if x.get('duplicate_failure') or x['fingerprint'] in seen or len(seen)>=30:continue
   seen.add(x['fingerprint'])
  selected.append(r)
shadow=OUT/'workspace/viewer-datasets'/DS;shadow.mkdir(parents=True,exist_ok=True)
(shadow/'metadata.jsonl').write_text(''.join(json.dumps(dict(r,video_path=str(B/'viewer-datasets'/DS/r['video_path'])))+'\n' for r in selected))
import extract
extract.ROOT=OUT/'workspace';extract.OUT=OUT;extract.DATASETS=[DS];write(OUT/'status.json',dict(status='extracting_visual_deltas',selected=len(selected)));extract.main()
for r in selected:
 states=[json.loads(l) for l in (Path(r['evidence_path'])/'states.jsonl').read_text().splitlines()];frames=[]
 for s in states:
  d=[[math.dist(tcp[:3],c[:3]) for c in s['cover_contacts']] for tcp in s['tool_tcp']]
  frames.append(dict(frame=s['frame'],phase=s['current_state_pointer'],fail=s['fail_flag'],covers=[p['p'] for p in s['poses']['covers']],blocks=[p['p'] for p in s['poses']['blocks']],tool_poses=s['tool_poses'],tool_to_grasp_distance=d))
 write(B/'annotations/s01-wrong-lid-20261105'/DS/'geometry'/(r['sample_id']+'.json'),dict(sample_id=r['sample_id'],frames=frames,open_order=states[0]['open_order'],source='direct passive geometry capture'))
ids=OUT/'selected.json';write(ids,[r['sample_id'] for r in selected])
env=dict(os.environ,COVER_DATASETS=json.dumps([DS]),COVER_SELECTED_IDS=str(ids),COVER_DELTA_ROOT=str(OUT),COVER_ANNOTATION_ROOT=str(OUT/'fixed-labels'))
subprocess.run([sys.executable,str(S/'relabel_imprint_fixed.py'),'--publish'],env=env,check=True)
report=json.loads((OUT/'fixed-labels/status.json').read_text());assert report['status']=='published',report
write(OUT/'status.json',dict(status='complete',selected=len(selected),counts=counts,time=time.time()))
