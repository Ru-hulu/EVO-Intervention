import os,time,signal,subprocess
from pathlib import Path
s=Path(__file__).parent;r=Path('/cache/robot-baselines/eval-results/vla-render-validation-20261104/simplemem/put_back_block');pid=698130
while not (r/'rt32/result.json').exists():
 try:os.kill(pid,0)
 except ProcessLookupError:raise RuntimeError('Initial SimpleMem validation worker exited; inspect its error record')
 time.sleep(3)
os.kill(pid,signal.SIGSTOP)
parents={int(a):int(b) for a,b in (l.split() for l in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())};owned={pid}
while True:
 n=len(owned);owned|={p for p,q in parents.items() if q in owned}
 if len(owned)==n:break
for p in owned-{pid}:
 try:os.kill(p,signal.SIGTERM)
 except ProcessLookupError:pass
os.kill(pid,signal.SIGKILL)
p=r/'rt8'
if p.exists() and not (p/'result.json').exists():p.rename(p.with_name('rt8-worker-upgrade-'+str(int(time.time()))))
with Path('/cache/robot-baselines/logs/vla-failures-simplemem.log').open('a') as log:
 proc=subprocess.Popen(['bash',str(s/'launch.sh'),'simplemem','0'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(s/'simplemem-upgrade.pid').write_text(str(proc.pid))
