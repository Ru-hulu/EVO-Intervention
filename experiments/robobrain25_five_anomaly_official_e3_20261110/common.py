from pathlib import Path
import hashlib,json,os,time
ROOT=Path(__file__).resolve().parent
PY='/cache/cvpzhr/envs/robometer/bin/python'
TASKS=['pencil','flipcup','spray','pour','drawer']
NAMES={'pencil':'笔袋恢复','flipcup':'翻杯恢复','spray':'喷瓶检查','pour':'重复倒水','drawer':'抽屉检查'}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def config(task):return read(ROOT/task/'config.json')
def status(task,phase,**kw):
 p=ROOT/task/'status.json';s=read(p) if p.exists() else {'task':task}
 s.update(phase=phase,updated_at=time.time(),**kw);save(p,s)
