"""Download finished atomic prediction files, refresh comparison until all 40 finish."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
remote = 'ma-user@dev-modelarts-cnnorth9.huaweicloud.com'
source = '/cache/robot-baselines/eval-results/robometer-cover40-20261109'
key = str(Path.home()/'.ssh/zhr.pem')
while True:
    subprocess.run(['scp','-q','-r','-P','30312','-i',key,remote+':'+source+'/predictions',str(root)],check=True)
    subprocess.run([sys.executable,str(root/'code/compare.py')],check=True)
    report = json.loads((root/'comparison.json').read_text())
    if report['trajectories'] == 40:
        for filename in ('manifest-0.json','manifest-1.json','status-0.json','status-1.json'):
            subprocess.run(['scp','-q','-P','30312','-i',key,remote+':'+source+'/'+filename,str(root)],check=True)
        (root/'status.json').write_text(json.dumps(dict(status='complete',trajectories=40)))
        break
    time.sleep(30)
