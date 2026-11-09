"""Original Forest base caches + exact approved static/jitter augmentation; no training launch."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,shutil,hashlib,csv
S=Path(__file__).resolve().parent
OUT=Path('/cache/cvpzhr/datasets/forest-five-anomaly-5fps-static-jitter-seed42-20261109')
sys.path[:0]=[str(S/'code'),str(S)]
import prepare_static_jitter as aug
aug.CONTRACT=dict(aug.CONTRACT,scope='124 independent training episodes across five anomaly tasks; one augmented version each',validation='33 unaugmented validation episodes from frozen HF split')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return aug.sha(Path(p))
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def get_plan():
 p=read(OUT/'plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()==p['plan_sha256'];return p

def verify_base(record,directory,plan):
 import torch
 from mem_lite.datasets.m3_dataset import M3Dataset,VIEW_ORDER
 m=read(directory/'summary.json');rows=aug.rows_at(directory)
 with Path(record['progress_csv']).open() as f:values=[float(r['pt']) for r in csv.DictReader(f)]
 expected=list(range(0,record['frames'],6))
 if expected[-1]!=record['frames']-1:expected.append(record['frames']-1)
 assert m['history_source']==plan['history_source'] and m['history_capacity']==1008 and m['history_dim']==2560
 assert m['episode_id']==record['episode_id'] and m['task_prompt']==record['task_prompt']
 assert len(rows)==m['num_samples']==record['source_samples'] and [r['frame_index'] for r in rows]==expected
 previous=None;selected={1,len(rows)//2,len(rows)-1}
 for i,row in enumerate(rows):
  assert row['pt']==values[row['frame_index']]
  assert set(row['images'])==set(VIEW_ORDER) and all(v is not None for v in row['images'].values())
  assert all((directory/v).is_file() for v in row['images'].values())
  assert (row['history'] is None)==(i==0)
  if row['history']:
   hp=directory/row['history'];assert hp.is_file() and hp.stat().st_size>100
   if i in selected:
    h=torch.load(hp,map_location='cpu',weights_only=True);t=h['tokens'];assert h['through_frame']==previous
    assert t.ndim==2 and t.shape[1]==2560 and 0<len(t)<=1008 and t.dtype==torch.bfloat16 and torch.isfinite(t).all().item()
  previous=row['frame_index']
 assert len(list((directory/'history').glob('*.pt')))==len(rows)-1
 dataset=M3Dataset([str(directory)])
 for i in [0,len(rows)-1]:assert dataset[i]['target_pt']==rows[i]['pt'] and len(dataset[i]['images'])==3
 return {'episode_id':record['episode_id'],'samples':len(rows),'source_pt_unchanged':True,'all_three_views_present':True,'causal_history_checked':True}

def build_base(r,visual,processor,plan):
 from mem_lite.memory.stream import StreamMemory
 from mem_lite.memory.vision import QwenFrameEncoder
 from mem_lite.datasets.prepare_stream import build_episode,iter_observations
 output=Path(r['base_dir'])
 if (output/'summary.json').exists():return verify_base(r,output,plan)
 assert not output.exists()
 part=output.with_name(output.name+'.partial');owner=part.with_suffix('.owner.json');ownership={'episode_id':r['episode_id'],'plan_sha256':plan['plan_sha256']}
 if part.exists():
  assert read(owner)==ownership and part.is_relative_to(OUT) and not part.is_symlink();shutil.rmtree(part)
 save(owner,ownership)
 assert sha(r['progress_csv'])==r['progress_sha256']
 with Path(r['progress_csv']).open() as f:values=[float(row['pt']) for row in csv.DictReader(f)]
 stream=StreamMemory(QwenFrameEncoder(visual),processor,plan['history_source']['memory_config'],65536,4096)
 observations=iter_observations(r['videos'],dict(enumerate(values)),sample_fps=5,start_frame=0,end_frame=r['frames'])
 build_episode(observations,stream,part,r['task_prompt'],r['episode_id'],plan['history_source'],source=r)
 result=verify_base(r,part,plan);part.rename(output);owner.unlink();return result

def assigned(plan,count):
 groups=[[] for _ in range(count)];loads=[0]*count
 for r in sorted(plan['records'],key=lambda r:(-(r['source_samples']+(r['augmented_samples'] if r['split']=='train' else 0)),r['episode_id'])):
  w=min(range(count),key=loads.__getitem__);groups[w].append(r);loads[w]+=r['source_samples']+(r['augmented_samples'] if r['split']=='train' else 0)
 return groups

def worker(worker_id,count,gpu):
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch
 from mem_lite.memory.vision import load_vision_encoder
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.22)
 plan=get_plan();records=assigned(plan,count)[worker_id];path=S/f'cache-worker-{worker_id}.json';state={'worker':worker_id,'gpu':gpu,'pid':os.getpid(),'total':len(records),'completed':0,'samples':0}
 def update(**kw):state.update(**kw,updated_at=time.time());save(path,state)
 try:
  update(phase='loading_vision');visual,processor,digest=load_vision_encoder(plan['model_path'],'cuda');assert digest==plan['history_source']['vision_sha256']
  for r in records:
   if shutil.disk_usage(OUT).free<250*2**30:raise RuntimeError('Less than 250 GiB free for cache generation')
   update(phase='base',current=r['episode_id']);base=build_base(r,visual,processor,plan)
   if r['split']=='train':
    update(phase='augmentation');d=Path(r['base_dir'])
    spec={'episode_id':r['episode_id'],'source_dir':str(d),'output_dir':r['output_dir'],'source_samples_sha256':sha(d/'samples.jsonl'),'source_summary_sha256':sha(d/'summary.json'),'source_samples':r['source_samples'],'augmented_samples':r['augmented_samples'],'added_samples':r['added_samples'],'events':r['events']}
    result=aug.build_episode({'plan_sha256':plan['plan_sha256']},spec,visual,processor)
   else:result=base
   save(OUT/'receipts'/f"{r['episode_id']}.json",{'episode_id':r['episode_id'],'split':r['split'],'base':base,'result':result,'output_dir':r['output_dir'],'samples_sha256':sha(Path(r['output_dir'])/'samples.jsonl'),'summary_sha256':sha(Path(r['output_dir'])/'summary.json')})
   state['completed']+=1;state['samples']+=r['augmented_samples'];update(phase='running',current=None)
  update(phase='completed')
 except Exception as e:update(phase='failed',error=repr(e));raise

def idle(gpus):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 values={int(parts[0]):(int(parts[1]),int(parts[2])) for row in rows if (parts:=row.split(','))}
 return all(values[g][0]<2000 and values[g][1]<10 for g in gpus)

def controller(gpus,workers_per_gpu,wait_idle):
 lock=(S/'cache-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=S/'cache-status.json';state={'pid':os.getpid(),'gpus':gpus,'workers_per_gpu':workers_per_gpu,'training_started':False,'automatic_training_start':False,'started_at':time.time()}
 def update(**kw):state.update(**kw,updated_at=time.time());save(status,state)
 try:
  update(phase='waiting_for_source_preparation')
  while not (OUT/'plan.json').exists():
   if (S/'preparation-status.json').exists() and read(S/'preparation-status.json')['phase']=='failed':raise RuntimeError('CPU source preparation failed; inspect preparation-status.json')
   time.sleep(15)
  plan=get_plan()
  for rel,d in plan['code_sha256'].items():assert sha(S/rel)==d,rel
  if wait_idle:
   update(phase='waiting_for_idle_gpus')
   while True:
    if idle(gpus):
     time.sleep(15)
     if idle(gpus):break
    time.sleep(30)
  update(phase='building_caches')
  count=len(gpus)*workers_per_gpu;children=[]
  for i in range(count):
   with (S/f'cache-worker-{i}.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(S/'run_cache.py'),'--worker',str(i),'--worker-count',str(count),'--gpu',str(gpus[i%len(gpus)])],cwd=S,stdout=log,stderr=subprocess.STDOUT)
   children.append(p)
  codes=[p.wait() for p in children]
  if any(codes):raise RuntimeError(f'Cache worker exit codes: {codes}')
  update(phase='validating_outputs')
  from mem_lite.datasets.m3_dataset import M3Dataset
  for split in ['train','validation']:
   records=[r for r in plan['records'] if r['split']==split];paths=[]
   for r in records:
    d=Path(r['output_dir']);receipt=read(OUT/'receipts'/f"{r['episode_id']}.json")
    assert receipt['samples_sha256']==sha(d/'samples.jsonl') and receipt['summary_sha256']==sha(d/'summary.json')
    paths.append(str(d))
   dataset=M3Dataset(paths,load_history=False);expected=sum(r['augmented_samples'] for r in records);assert len(dataset)==expected
   save(OUT/f'{split}.sample_files.json',paths)
  report={'status':'completed','episodes':157,'train_episodes':124,'validation_episodes':33,'samples':read(OUT/'source-preflight.json')['final_observations'],'original_normal_reference':str(S/'normal-reference-config.json'),'source_pt_preserved':True,'validation_augmented':False,'all_three_views_present':True,'inserted_pt_fixed_to_anchor':True,'training_started':False,'automatic_training_start':False,'elapsed_seconds':time.time()-state['started_at']}
  save(OUT/'cache-ready.json',report);update(phase='completed',result=report)
 except Exception as e:update(phase='failed',error=repr(e));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--worker-count',type=int);p.add_argument('--gpu',type=int);p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--workers-per-gpu',type=int,default=4);p.add_argument('--wait-idle',action='store_true');a=p.parse_args()
 if a.worker is None:controller(a.gpus,a.workers_per_gpu,a.wait_idle)
 else:worker(a.worker,a.worker_count,a.gpu)
