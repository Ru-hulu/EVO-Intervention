"""Rebuild current-three-view inputs, verify four GPUs, then launch the authorized fresh run."""
from pathlib import Path
import os,subprocess,signal,time,json,traceback,struct,hashlib,fcntl
SETUP=Path(__file__).resolve().parent
CODE=Path('/cache/cvpzhr/code/robometer-current3-w8-e1-20261104')
PY='/cache/cvpzhr/envs/robometer/bin/python'
lock=(SETUP/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ENV={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'),
'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONNOUSERSITE':'1'}
state=dict(pid=os.getpid(),started_at=time.time(),formal_training_started=False,
           input_contract='history7-main-current3-v1',training_authorized=True)
child=None
def write(path,data):
 p=Path(path);temp=p.with_suffix(p.suffix+'.tmp')
 temp.write_text(json.dumps(data,indent=2));temp.replace(p)
def save(phase,**kw):
 state.update(phase=phase,updated_at=time.time(),**kw);write(SETUP/'pipeline-status.json',state)
def run(stage,args,timeout):
 global child
 save(stage)
 with (SETUP/(stage+'.log')).open('a') as log:
  child=subprocess.Popen([PY,*args],cwd=SETUP,env=ENV,stdout=log,stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL,start_new_session=True)
  save(stage,child_pid=child.pid)
  try:code=child.wait(timeout=timeout)
  except BaseException:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
 if code:raise RuntimeError(f'{stage} exited {code}; see {stage}.log')
def make_benchmark():
 c=json.loads((SETUP/'config.json').read_text())
 cache=Path(c['train_data']).parent
 report=json.loads((cache/'report.json').read_text())
 assert report['status']=='completed'
 rows=[json.loads(line) for line in Path(c['validation_data']).read_text().splitlines()]
 selected=[rows[int((i+.5)*len(rows)/752)] for i in range(752)]
 assert len({r['task_id'] for r in selected})==47
 val=SETUP/'benchmark.validation.jsonl'
 with val.open('w') as f:
  for row in selected:f.write(json.dumps(row)+'\n')
 b=json.loads(json.dumps(c));b['validation_data']=str(val)
 b['output_dir']=str(SETUP/'benchmark-output');b['eval_steps']=20
 b['training'].update(max_steps=40,logging_steps=5)
 write(SETUP/'benchmark.config.json',b)
 manifest=json.loads((cache/'manifest.json').read_text())
 headers=[]
 for info in manifest['splits']['train']['episodes']:
  with Path(info['pixel_cache']).open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  patches=8*header['pixels_cam_high']['shape'][1]+header['pixels_cam_left_wrist']['shape'][1]+header['pixels_cam_right_wrist']['shape'][1]
  headers.append(dict(episode=info['episode_id'],tokens=header['input_input_ids']['shape'][0],patches=patches,samples=info['samples'],file=info['episode_jsonl']))
 worst=max(headers,key=lambda x:(x['tokens'],x['patches']))
 stress=json.loads(Path(worst['file']).read_text().splitlines()[-1])
 write(SETUP/'stress-row.json',stress)
 write(SETUP/'input-shapes.json',dict(episodes=len(headers),max=worst,
     min_tokens=min(x['tokens'] for x in headers),
     weighted_mean_tokens=sum(x['tokens']*x['samples'] for x in headers)/sum(x['samples'] for x in headers)))
try:
 run('cpu-tests',['-m','unittest','test_inputs','test_saving','-v'],180)
 write(SETUP/'cpu-tests.passed.json',dict(passed=True,time=time.time(),tests=8))
 run('cache',[str(SETUP/'prepare_cache.py'),'--workers','64'],7200)
 save('preparing_benchmark');make_benchmark()
 run('preflight',[str(SETUP/'run_preflight.py')],2400)
 assert json.loads((SETUP/'preflight-status.json').read_text())['phase']=='completed'
 save('launching_training',child_pid=None)
 with (SETUP/'training-runner.log').open('a') as log:
  process=subprocess.Popen([PY,'-u',str(SETUP/'run_training.py')],cwd=SETUP,
       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (SETUP/'training.pid').write_text(str(process.pid)+'\n')
 # Confirm the training controller accepted the configuration before exiting.
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  p=SETUP/'training-status.json'
  if p.exists():
   training=json.loads(p.read_text())
   if training['phase']=='failed':raise RuntimeError(training['error'])
   if training.get('formal_training_started'):
    save('training_launched',formal_training_started=True,training_pid=process.pid,
         training_status=str(p),output=training['output'])
    break
  if process.poll() is not None:raise RuntimeError('Training controller exited before accepting launch')
  time.sleep(1)
 else:raise RuntimeError('Training controller did not accept launch within 30 seconds')
except BaseException as exc:
 save('failed',error=repr(exc),traceback=traceback.format_exc())
 raise
