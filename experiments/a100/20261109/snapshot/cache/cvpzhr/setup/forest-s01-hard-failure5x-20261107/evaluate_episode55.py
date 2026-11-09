import json,os,shutil,subprocess,time,hashlib
from pathlib import Path
import torch
BASE=Path('/cache/cvpzhr/evaluations')
CK=Path('/cache/cvpzhr/outputs/forest-s01-hard-failure5x-20261107/last.pt')
VIEW=BASE/'forest-s01-val5-20261106'
CODE=Path('/cache/robot-baselines/repos/memory-vla-RL/downstream/eval/forest_latest')
while not CK.exists():time.sleep(5)
# Atomic training checkpoints are replaced, never written in place.
snapshot=BASE/'forest-hard-failure-episode55-snapshot.pt'
assert not snapshot.exists()
shutil.copy2(CK,snapshot)
step=torch.load(snapshot,map_location='cpu',weights_only=True,mmap=True)['step']
out=BASE/f'forest-s01-hard-failure-episode55-step{step}-20261107'
out.mkdir(exist_ok=False);snapshot.rename(out/'source-checkpoint.pt')
shutil.copytree(CODE,out/'code',ignore=shutil.ignore_patterns('__pycache__'))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/cache/cvpzhr/setup/forest-s01-hard-failure5x-20261107/code',PYTHONUTF8='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (out/'eval.log').open('w') as log:
 subprocess.run(['/cache/cvpzhr/envs/forest/bin/python','-u',str(out/'code/evaluate.py'),'--checkpoint',str(out/'source-checkpoint.pt'),'--output-dir',str(out),'--episodes','55','--batch-size','4'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
m=json.loads((out/'metrics.json').read_text());target=out/'viewer-data'/m['dataset'];rows=[json.loads(s) for s in (target/'metadata.jsonl').read_text().splitlines()];assert len(rows)==1 and rows[0]['episode_index']==55
p=target/'progress'/f"{rows[0]['sample_id']}.json";d=json.loads(p.read_text());assert d['checkpoint_step']==step
sha=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest()
d['source']=f'困难失败帧加权 FOREST · 训练早期 step {step} · 仅 episode 55 · 5 Hz 因果记忆前向'
d['checkpoint_sha256']=sha;p.write_text(json.dumps(d,ensure_ascii=False)+'\n')
rows[0]['task_name']='覆盖与揭开积木 · 困难失败加权 · 训练早期';(target/'metadata.jsonl').write_text(json.dumps(rows[0],ensure_ascii=False)+'\n')
name='FOREST-S01-Latest';link=VIEW/'viewer-data'/name
assert link.is_symlink();(out/'previous-latest.json').write_text(json.dumps(dict(target=str(link.resolve()))))
nextlink=link.with_name(name+'.next');nextlink.symlink_to(target,target_is_directory=True);nextlink.replace(link)
namespath=VIEW/'val5-viewer/catalog_names.json';names=json.loads(namespath.read_text());names[name]=f'最新 FOREST · 困难失败加权 step{step} · 仅 episode 55（训练早期）';tmp=namespath.with_suffix('.tmp');tmp.write_text(json.dumps(names,ensure_ascii=False,indent=2)+'\n');tmp.replace(namespath)
(out/'status.json').write_text(json.dumps(dict(phase='completed',published=True,checkpoint_step=step,checkpoint_sha256=sha),indent=2))
print(json.dumps(dict(output=str(out),step=step,metrics=m),ensure_ascii=False),flush=True)
