"""Restricted SSH receiver for completed IMPRINT episodes."""
import hashlib,json,os,re,shutil,subprocess,sys,tarfile
from pathlib import Path
host=sys.argv[1]
assert host in ('4090C','4090D')
command=os.environ.get('SSH_ORIGINAL_COMMAND','')
if command=='ping':
 print('READY',host);sys.exit(0)
if command=='migration-status':
 p=Path('/cache/robot-baselines/setup/imprint-extra-20261105/recurrent-migration.json')
 print(p.read_text() if p.exists() else json.dumps({'status':'waiting_episode'}));sys.exit(0)
if command=='download-recurrent':
 subprocess.run(['tar','-C','/cache/robot-baselines/models/IMPRINT/simulation/rmbench/recurrent/stage2','-cf','-','step-100000'],check=True);sys.exit(0)
assert re.fullmatch(r'upload(?:-pilot)? (full_attention|recurrent)/(cover_blocks|blocks_ranking_try)/seed_[0-9]+',command),command
rel=command.split()[1];base=Path('/cache/robot-baselines');pilot=command.startswith('upload-pilot ');batch='imprint-failure-pilot-20261105' if pilot else 'imprint-extra-20261105';stage=base/'incoming-imprint-direct'/batch/host/rel
stage.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  assert not member.issym() and not member.islnk() and not member.isdev()
  path=Path(member.name)
  assert not path.is_absolute() and '..' not in path.parts
  archive.extract(member,stage)
manifest=json.loads((stage/'transfer-sha256.json').read_text())
for name,expected in manifest.items():
 p=stage/name;assert p.resolve().is_relative_to(stage.resolve())
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==expected,name
out=base/'eval-results'/batch/rel
for name in ('result.json','status.json'):
 p=stage/name;x=json.loads(p.read_text());assert x['status']=='complete'
 x['source_evidence_path']=x['evidence'];x['evidence']=str(out/'run/evidence/0');x['execution_host']=host;p.write_text(json.dumps(x,indent=2))
out.parent.mkdir(parents=True,exist_ok=True)
if not out.exists():stage.rename(out)
else:
 for name,expected in manifest.items():
  if name in ('result.json','status.json'):continue
  if name.endswith('.log'):
   temp=(out/name).with_suffix('.log.transfer');shutil.copyfile(stage/name,temp);temp.replace(out/name)
  h=hashlib.sha256()
  with (out/name).open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==expected,name
subprocess.run([str(base/'envs/imprint/bin/python'),str(base/('setup/imprint-failure-pilot-20261105/publish.py' if pilot else 'setup/imprint-extra-20261105/publish.py'))],check=True)
report=base/'setup'/batch/'direct-transfer-receipts'/host
report.mkdir(parents=True,exist_ok=True)
(report/(out.parent.name+'-'+out.name+'.json')).write_text(json.dumps(dict(host=host,trajectory=rel,sha256_verified=True,files=len(manifest))))
print('VERIFIED',host,rel)
