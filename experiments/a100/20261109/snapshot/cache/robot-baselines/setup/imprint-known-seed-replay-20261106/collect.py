"""Natural failures only; retain every attempt and exclude exact action repeats."""
import os,json,time,hashlib,socket,traceback,signal
import numpy as np
import worker
batch = os.environ.get('IMPRINT_RESULTS_BATCH', 'imprint-failure-pilot-20261105')
ROOT = worker.R / 'eval-results' / batch / worker.MODEL / 'cover_blocks'
plan_path = worker.S / 'seed-plan.json'
HOST=os.environ['IMPRINT_HOST'];START=int(os.environ['IMPRINT_SEED'])
def records():return [json.loads(p.read_text()) for p in ROOT.glob('seed_*/result.json')]
def command(seed):
 data=json.dumps({'cmd':'set_collection_seed','obs':seed}).encode()
 with socket.create_connection(('127.0.0.1', 8070 + int(worker.GPU)),timeout=60) as s:
  s.sendall(len(data).to_bytes(4,'big')+data)
  def read(n):
   b=b''
   while len(b)<n:
    v=s.recv(n-len(b));assert v;b+=v
   return b
  response=json.loads(read(int.from_bytes(read(4),'big')))
  assert response['res']['policy_seed']==seed,response
class ResidentServer:
 def __init__(self, pid):
  self.pid = pid
  self.returncode = None
 def poll(self):
  try:
   status = (worker.Path('/proc') / str(self.pid) / 'stat').read_text()
   if status.split(') ', 1)[1].startswith('Z '):
    self.returncode = 1
  except FileNotFoundError:
   self.returncode = 1
  return self.returncode
 def terminate(self):
  if self.poll() is None:
   os.kill(self.pid, signal.SIGTERM)
 def wait(self, timeout=30):
  deadline = time.monotonic() + timeout
  while self.poll() is None and time.monotonic() < deadline:
   time.sleep(.25)

if os.environ.get('IMPRINT_EXISTING_SERVER_PID'):
 worker.SERVER = ResidentServer(int(os.environ['IMPRINT_EXISTING_SERVER_PID']))

try:
 while True:
  done=records();unique={r['fingerprint'] for r in done if not r['success']}
  status=dict(status='running',model=worker.MODEL,completed=len(done),failures=sum(not r['success'] for r in done),unique_failures=len(unique),target=30,max_trials=300,time=time.time())
  if len(unique)>=30 or len(done)>=300:
   status['status']='complete' if len(unique)>=30 else 'insufficient_natural_failures';worker.save(worker.S/'status.json',status);break
  worker.save(worker.S/'status.json',status)
  previous = [int(p.name.split('_')[1]) for p in ROOT.glob('seed_*')]
  trial_id = max([START - 1] + previous) + 1
  plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
  seed = trial_id
  policy_seed = 1000000 + trial_id
  mode = 'scene_sweep'
  if plan:
   index = sum(r.get('seed_selection') in ('known_failure_replay', 'hard_scene_resample') for r in done)
   replays = plan['replays']
   if index < len(replays):
    seed, policy_seed = replays[index]
    mode = 'known_failure_replay'
   elif plan['scenes']:
    seed = plan['scenes'][(index - len(replays)) % len(plan['scenes'])]
    mode = 'hard_scene_resample'
   else:
    status['status'] = 'diagnostics_complete'
    worker.save(worker.S/'status.json', status)
    break
  os.environ['IMPRINT_POLICY_SEED'] = str(policy_seed)
  if worker.SERVER is not None:
   command(policy_seed)
  out = ROOT / f'seed_{trial_id}'
  worker.save(out/'status.json', dict(status='running', requested_seed=seed, trial_id=trial_id, policy_seed=policy_seed, model=worker.MODEL, start=time.time()))
  x = worker.episode('cover_blocks', seed, 8, out)
  e = out / 'run/evidence/0'
  states=[json.loads(l) for l in (e/'states.jsonl').read_text().splitlines()]
  actions=[json.loads(l)['args'][0] for l in (e/'actions.jsonl').read_text().splitlines()]
  signature=hashlib.sha256(np.round(np.asarray(actions),3).astype('<f4').tobytes()+json.dumps(states[0]['target_state_transition']).encode()).hexdigest()
  fail=next((s['frame'] for s in states if s['fail_flag']),None);phase=max(s['current_state_pointer'] for s in states)
  outcome='success' if x['success'] else 'wrong_order_irreversible' if fail is not None else 'timeout_no_completed_subtask' if phase==0 else 'timeout_partial_progress'
  x.update(execution_host=HOST,trial_id=trial_id,seed_selection=mode,policy_seed=policy_seed,fingerprint=signature,duplicate_failure=not x['success'] and signature in unique,failure_mode=outcome,completed_phase=phase,first_native_failure=fail,collection='natural_failure_pilot',perturbation=None)
  worker.save(out/'status.json',x);worker.save(out/'result.json',x);print('COMPLETE',json.dumps(x),flush=True)
except BaseException:
 worker.save(worker.S/'error.json',dict(error=traceback.format_exc()));raise
finally:
 if worker.SERVER is not None:worker.SERVER.terminate();worker.SERVER.wait(timeout=30)
