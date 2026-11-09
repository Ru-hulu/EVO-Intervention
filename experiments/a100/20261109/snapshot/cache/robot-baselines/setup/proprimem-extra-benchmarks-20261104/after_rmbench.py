import json,time,subprocess,traceback
from pathlib import Path
s=Path(__file__).parent;plan=json.loads((s/'plan.json').read_text())
def status(stage,**kw):(s/'status.json').write_text(json.dumps(dict(stage=stage,updated=time.time(),**kw),indent=2))
status('waiting_for_rmbench_210')
while not all(Path(p).exists() and json.loads(Path(p).read_text()).get('status')=='complete' for p in plan['depends_on']):time.sleep(30)
for name in json.loads((s/'plan.json').read_text())['scripts']:
 status('running',script=name)
 with (s/(name+'.log')).open('a') as log:
  code=subprocess.call(['bash',str(s/name)],stdout=log,stderr=subprocess.STDOUT)
 subprocess.run(['/cache/robot-baselines/envs/simplemem-libero/bin/python',str(s/'export.py')],check=True)
 if code:
  status('error',script=name,exit_code=code);raise SystemExit(code)
status('complete')
