"""Annotate the completed extra batch with the original shared reference pool."""
import json,sys,time,shutil
from pathlib import Path
BASE=Path('/cache/robot-baselines');DS='ProPriMem-IMPRINT-Extra-20261105';OLD='ProPriMem-VLA-Failures-20261104'
OUT=BASE/'annotations/imprint-extra-20261105';REF=BASE/'annotations/boundary-integral-20261105'
sys.path.insert(0,str(BASE/'setup/model-trajectory-progress'))
import extract,integrate,finalize

def rows(ds):
 p=BASE/'viewer-datasets'/ds/'metadata.jsonl'
 return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 while len(rows(DS))<80:
  integrate.write(OUT/'status.json',dict(status='waiting_collection',completed=len(rows(DS)),target=80,time=time.time()))
  time.sleep(30)
 new=rows(DS);old=rows(OLD)
 while not all((REF/OLD/'deltas'/(r['sample_id']+'.json')).exists() for r in old):time.sleep(30)
 integrate.write(OUT/'status.json',dict(status='extracting_visual_deltas',time=time.time()))
 extract.OUT=OUT;extract.DATASETS=[DS];extract.main()
 # Use exactly the original RMBench successful reference pool, shared by models.
 shadow=OUT/'reference-workspace';meta=shadow/'viewer-datasets'/OLD/'metadata.jsonl';meta.parent.mkdir(parents=True,exist_ok=True)
 meta.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old+[dict(r,reference_eligible=False) for r in new]))
 delta=OUT/OLD/'deltas';delta.mkdir(parents=True,exist_ok=True)
 for r in old+new:
  name=r['sample_id']+'.json';src=REF/OLD/'deltas'/name if r in old else OUT/DS/'deltas'/name;dest=delta/name
  if not dest.exists():dest.symlink_to(src)
 integrate.ROOT=shadow;integrate.OUT=OUT;integrate.DATASETS=[OLD];integrate.main()
 report=integrate.read(OUT/'integration-status.json');ids={r['sample_id'] for r in new}
 for kind in ['progress','boundaries']:
  for key in ids:
   src=OUT/OLD/kind/(key+'.json');dest=OUT/DS/kind/(key+'.json')
   if src.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 issues=[dict(x,dataset=DS) for x in report['issues'] if x['sample'] in ids]
 integrate.write(OUT/'integration-status.json',dict(prepared=80-len(issues),pending=len(issues),issues=issues))
 finalize.ROOT=BASE;finalize.OUT=OUT;finalize.DATASETS=[DS];finalize.publish()
 integrate.write(OUT/'status.json',dict(status='complete' if not issues else 'partial_needs_review',target=80,issues=issues,time=time.time()))
if __name__=='__main__':main()
