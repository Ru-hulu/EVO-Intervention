"""Refresh published S01 stage descriptions after geometry labeling ends."""
import time,json
from publish_s01_progress import OUT,publish,write
while True:
 path=OUT/'status.json'
 if path.exists():
  status=json.loads(path.read_text())
  if status['status']!='running':break
 time.sleep(10)
for p in sorted(OUT.glob('*/reports/*.json')):
 r=json.loads(p.read_text());publish(r['dataset'],r['sample'])
status['stage_descriptions_updated']=True;status['time']=time.time();write(OUT/'status.json',status)
print(json.dumps(status),flush=True)
