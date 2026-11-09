"""Rebuild original Forest caches from pencil videos, without any augmentation."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
S=Path(__file__).resolve().parent
OUT=Path('/cache/cvpzhr/datasets/forest-pencil-5fps-no-augmentation-20261109')
sys.path.insert(0,str(S/'code'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def get_plan():
 p=read(OUT/'plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()==p['plan_sha256']
 assert p['augmentation_policy']['enabled'] is False and len(p['records'])==35
 return p

def expected_indices(frames):
 ids=list(range(0,frames,6))
 if ids[-1]!=frames-1:ids.append(frames-1)
 return ids

def verify(record,directory,plan):
 import torch
 from mem_lite.datasets.m3_dataset import M3Dataset,VIEW_ORDER
 summary=read(directory/'summary.json');rows=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
 with Path(record['progress_csv']).open() as f:values=[float(r['pt']) for r in csv.DictReader(f)]
 assert sha(record['progress_csv'])==record['progress_sha256'] and len(values)==record['frames']
 assert summary['history_source']==plan['history_source'] and summary['history_capacity']==1008 and summary['history_dim']==2560
 assert summary['episode_id']==record['episode_id'] and summary['task_prompt']==record['task_prompt']
 assert summary['augmentation']=={'enabled':False,'inserted_observations':0,'stationary':False,'jitter':False}
 assert len(rows)==summary['num_samples']==record['source_samples']
 assert [r['frame_index'] for r in rows]==expected_indices(record['frames'])
 previous=None;checked=0
 for i,row in enumerate(rows):
  assert row['pt']==values[row['frame_index']] and 0<=row['pt']<=1
  assert set(row['images'])==set(VIEW_ORDER) and all(v is not None for v in row['images'].values())
  assert all((directory/v).is_file() for v in row['images'].values())
  assert (row['history'] is None)==(i==0)
  if row['history']:
   hp=directory/row['history'];assert hp.is_file()
   h=torch.load(hp,map_location='cpu',weights_only=True);t=h['tokens'];assert h['through_frame']==previous
   assert t.ndim==2 and t.shape[1]==2560 and 0<len(t)<=1008 and t.dtype==torch.bfloat16 and torch.isfinite(t).all().item();checked+=1
  previous=row['frame_index']
 assert len(list((directory/'history').glob('*.pt')))==len(rows)-1
 dataset=M3Dataset([str(directory)])
 for i in [0,len(rows)-1]:assert dataset[i]['target_pt']==rows[i]['pt'] and len(dataset[i]['images'])==3
 return {'episode_id':record['episode_id'],'samples':len(rows),'source_pt_unchanged':True,'all_three_views_present':True,'all_histories_verified':checked,'inserted_observations':0,'augmentation_enabled':False}

def build(record,visual,processor,plan):
 from mem_lite.memory.stream import StreamMemory
 from mem_lite.memory.vision import QwenFrameEncoder
 from mem_lite.datasets.prepare_stream import build_episode,iter_observations
 output=Path(record['output_dir'])
 if (output/'summary.json').exists():return verify(record,output,plan)
 assert not output.exists()
 part=output.with_name(output.name+'.partial');owner=part.with_suffix('.owner.json');ownership={'episode_id':record['episode_id'],'plan_sha256':plan['plan_sha256']}
 if part.exists():
  assert read(owner)==ownership and part.is_relative_to(OUT) and not part.is_symlink();shutil.rmtree(part)
 save(owner,ownership)
 assert sha(record['progress_csv'])==record['progress_sha256']
 with Path(record['progress_csv']).open() as f:values={int(row['frame_index']):float(row['pt']) for row in csv.DictReader(f)}
 assert set(values)==set(range(record['frames']))
 stream=StreamMemory(QwenFrameEncoder(visual),processor,plan['history_source']['memory_config'],65536,4096)
 observations=iter_observations(record['videos'],values,sample_fps=5,start_frame=0,end_frame=record['frames'])
 build_episode(observations,stream,part,record['task_prompt'],record['episode_id'],plan['history_source'],source=record)
 summary=read(part/'summary.json');summary['augmentation']={'enabled':False,'inserted_observations':0,'stationary':False,'jitter':False};save(part/'summary.json',summary)
 result=verify(record,part,plan);part.rename(output);owner.unlink();return result

def assigned(plan,count):
 groups=[[] for _ in range(count)];loads=[0]*count
 for r in sorted(plan['records'],key=lambda r:(-r['source_samples'],r['episode_id'])):
  w=min(range(count),key=loads.__getitem__);groups[w].append(r);loads[w]+=r['source_samples']
 return groups

def worker(worker_id,count,gpu):
 os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 import torch
 from mem_lite.memory.vision import load_vision_encoder
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.10)
 plan=get_plan();records=assigned(plan,count)[worker_id];path=S/f'worker-{worker_id}.json';state={'worker':worker_id,'gpu':gpu,'pid':os.getpid(),'total':len(records),'completed':0,'samples':0}
 def update(**kw):state.update(**kw,updated_at=time.time());save(path,state)
 try:
  update(phase='loading_vision');visual,processor,digest=load_vision_encoder(plan['model_path'],'cuda');assert digest==plan['history_source']['vision_sha256']
  for r in records:
   if shutil.disk_usage(OUT).free<100*2**30:raise RuntimeError('Less than 100 GiB free')
   update(phase='building',current=r['episode_id']);result=build(r,visual,processor,plan);directory=Path(r['output_dir'])
   save(OUT/'receipts'/f"{r['episode_id']}.json",dict(result,split=r['split'],output_dir=r['output_dir'],samples_sha256=sha(directory/'samples.jsonl'),summary_sha256=sha(directory/'summary.json')))
   state['completed']+=1;state['samples']+=r['source_samples'];update(phase='running',current=None)
  update(phase='completed')
 except Exception as e:update(phase='failed',error=repr(e),traceback=traceback.format_exc());raise

def idle(gpus):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 values={int(parts[0]):(int(parts[1]),int(parts[2])) for row in rows if (parts:=row.split(','))}
 return all(values[g][0]<2000 and values[g][1]<10 for g in gpus)

def controller(gpus,workers_per_gpu):
 lock=(S/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'pid':os.getpid(),'gpus':gpus,'workers_per_gpu':workers_per_gpu,'training_started':False,'automatic_training_start':False,'started_at':time.time(),'augmentation_enabled':False,'task':'zipper-pencil-case-inspection'}
 def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(S/'cache-status.json',state)
 try:
  update('preflight');plan=get_plan()
  for rel,digest in plan['code_sha256'].items():assert sha(S/rel)==digest,rel
  for path,digest in plan['sources'].items():assert sha(path)==digest,path
  update('waiting_for_idle_gpus')
  while True:
   if idle(gpus):
    time.sleep(5)
    if idle(gpus):break
   time.sleep(10)
  count=len(gpus)*workers_per_gpu;children=[]
  for i in range(count):
   with (S/f'worker-{i}.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(S/'run_cache.py'),'--worker',str(i),'--worker-count',str(count),'--gpu',str(gpus[i%len(gpus)])],cwd=S,stdout=log,stderr=subprocess.STDOUT)
   children.append(p)
  update('building_caches',worker_pids=[p.pid for p in children],completed_episodes=0,total_episodes=35)
  while any(p.poll() is None for p in children):
   statuses=[read(S/f'worker-{i}.json') for i in range(count) if (S/f'worker-{i}.json').exists()]
   update('building_caches',completed_episodes=sum(x.get('completed',0) for x in statuses),completed_samples=sum(x.get('samples',0) for x in statuses),failed_workers=[x['worker'] for x in statuses if x.get('phase')=='failed']);time.sleep(10)
  codes=[p.wait() for p in children]
  if any(codes):raise RuntimeError(f'Cache worker exit codes: {codes}')
  update('validating_outputs');import torch;torch.set_num_threads(2)
  from mem_lite.datasets.m3_dataset import M3Dataset
  counts={};history_count=0
  for split in ['train','validation']:
   records=[r for r in plan['records'] if r['split']==split];paths=[]
   for r in records:
    directory=Path(r['output_dir']);receipt=read(OUT/'receipts'/f"{r['episode_id']}.json")
    assert receipt['samples_sha256']==sha(directory/'samples.jsonl') and receipt['summary_sha256']==sha(directory/'summary.json')
    assert receipt['inserted_observations']==0;history_count+=receipt['all_histories_verified'];paths.append(str(directory))
   ds=M3Dataset(paths,load_history=False);expected=sum(r['source_samples'] for r in records);assert len(ds)==expected
   counts[split]={'episodes':len(records),'samples':len(ds)};save(OUT/f'{split}.sample_files.json',paths)
  assert counts=={'train':{'episodes':28,'samples':10789},'validation':{'episodes':7,'samples':2177}}
  for path,digest in plan['sources'].items():assert sha(path)==digest,path
  report={'status':'completed','task':'zipper-pencil-case-inspection','counts':counts,'all_histories_verified':history_count,'original_source_labels_preserved':True,'augmentation_enabled':False,'inserted_observations':0,'all_three_views_present':True,'generation_mode':'fresh visual encoding and causal memory from original videos','training_started':False,'automatic_training_start':False,'elapsed_seconds':time.time()-state['started_at'],'plan_sha256':plan['plan_sha256']}
  save(OUT/'cache-ready.json',report);update('completed',completed_episodes=35,total_episodes=35,completed_samples=12966,result=report)
 except Exception as e:update('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--worker-count',type=int);p.add_argument('--gpu',type=int);p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--workers-per-gpu',type=int,default=4);a=p.parse_args()
 if a.worker is None:controller(a.gpus,a.workers_per_gpu)
 else:worker(a.worker,a.worker_count,a.gpu)
