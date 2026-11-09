"""Four independent GPU workers; preserve the accepted pilot and its progress values."""
from pathlib import Path
import argparse,copy,csv,fcntl,json,os,subprocess,sys,time,traceback
from compute_pilot import calculate,frame_labels,sha,save
SETUP=Path(__file__).resolve().parent
SOURCE=SETUP/'batch_source_manifest.json'
MODEL=Path('/cache/cvpzhr/models/Qwen3-VL-4B-Instruct')
OLD=Path('/cache/cvpzhr/setup/flipcup-progress-20261105')

def verify_existing(out,record):
 ep=record['episode'];cp=out/f'{ep}.progress.csv';dp=out/f'{ep}-delta.txt';mp=out/f'{ep}.metadata.json'
 if not any(p.exists() for p in [cp,dp,mp]):return False
 if not all(p.exists() for p in [cp,dp,mp]):raise RuntimeError(f'Incomplete existing result: {ep}')
 m=json.loads(mp.read_text())
 assert m['video_sha256']==record['video_sha256'] and m['annotation_sha256']==record['annotation_sha256']
 assert m['progress_csv_sha256']==sha(cp) and m['delta_sha256']==sha(dp)
 assert m['schema_version']=='repeated-pouring-first-visit-progress-v1'
 assert m['total_frames']==record['frames']
 return True

def worker(gpu):
 source=json.loads(SOURCE.read_text());root=Path(source['dataset_root']);out=root/'progress_first_visit'
 assignments=json.loads((SETUP/'batch_assignments.json').read_text());records=assignments[str(gpu)]
 status=SETUP/f'worker_gpu{gpu}.json';started=time.perf_counter();completed=[];done_frames=0
 def update(**kwargs):save(status,{'gpu':gpu,'pid':os.getpid(),'completed_episodes':completed,'completed_frames':done_frames,'total_frames':sum(r['frames'] for r in records),'total_episodes':len(records),'elapsed_seconds':time.perf_counter()-started,**kwargs})
 try:
  os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['OMP_NUM_THREADS']='4';os.environ['TOKENIZERS_PARALLELISM']='false'
  sys.path[:0]=[str(OLD/'pydeps'),str(OLD)]
  import av,torch,importlib.metadata
  from original_video_to_delta import QwenDelta
  assert importlib.metadata.version('transformers')=='4.57.6'
  torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
  update(phase='loading_model');encoder=QwenDelta(MODEL,'cuda',262144)
  template=json.loads((out/'episode_000001.metadata.json').read_text())
  for record in records:
   ep=record['episode']
   if verify_existing(out,record):
    completed.append(ep);done_frames+=record['frames'];update(phase='running');continue
   vp=root/'videos/main_view'/f'{ep}.mp4';ap=root/'videos/main_view/annotation'/f'{ep}.json'
   assert sha(vp)==record['video_sha256'] and sha(ap)==record['annotation_sha256']
   ann=json.loads(ap.read_text());frame_labels(ann);epstart=time.perf_counter()
   encoder.last_image=encoder.last_phi=None;previous=None;deltas=[]
   with av.open(str(vp)) as container:
    stream=container.streams.video[0];stream.codec_context.thread_count=2
    assert float(stream.average_rate)==ann['fps']
    for i,frame in enumerate(container.decode(stream)):
     current=torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1)
     d=0. if previous is None else encoder.image2delta(previous,current)
     deltas.append(float(f'{d:.6f}'));previous=current
     if (i+1)%150==0:update(phase='computing_delta',current_episode=ep,current_frames=i+1)
   assert len(deltas)==record['frames'];labels,credited,pt,details=calculate(deltas,ann)
   dp=out/f'{ep}-delta.txt';tmp=dp.with_suffix('.txt.tmp');tmp.write_text(''.join(f'{d:.6f}\n' for d in deltas));tmp.replace(dp)
   cp=out/f'{ep}.progress.csv';tmp=cp.with_suffix('.csv.tmp')
   fields=['frame_index','time_seconds','stage_id','segment_index','stage_name','target_position','target_occurrence','mode','count_for_progress','delta','credited_delta','pt']
   with tmp.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for i,(r,d,c,p) in enumerate(zip(labels,deltas,credited,pt)):
     writer.writerow({'frame_index':i,'time_seconds':i/ann['fps'],**r,'delta':d,'credited_delta':c,'pt':p})
   tmp.replace(cp)
   meta=copy.deepcopy(template)
   meta.update(episode=ep,status='completed_with_user_approved_rule',rule_approved_date=source['rule_approved_date'],total_frames=len(pt),video_path=str(vp),video_sha256=record['video_sha256'],annotation_path=str(ap),annotation_sha256=record['annotation_sha256'],delta_sha256=sha(dp),progress_csv_sha256=sha(cp),implementation_sha256=sha(__file__),progress_function_sha256=sha(SETUP/'compute_pilot.py'),elapsed_seconds=time.perf_counter()-epstart,gpu=gpu,**details)
   save(out/f'{ep}.metadata.json',meta)
   completed.append(ep);done_frames+=record['frames'];update(phase='running',last_episode=ep)
   print(json.dumps({'gpu':gpu,'episode':ep,'frames':record['frames'],'seconds':meta['elapsed_seconds']}),flush=True)
  update(phase='completed')
 except Exception as exc:
  update(phase='failed',error=str(exc),traceback=traceback.format_exc());raise

def parent():
 lock=(SETUP/'batch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.perf_counter();source=json.loads(SOURCE.read_text());root=Path(source['dataset_root']);out=root/'progress_first_visit'
 status=SETUP/'batch_status.json'
 save(status,{'phase':'preflight','pid':os.getpid()})
 template=json.loads((out/'episode_000001.metadata.json').read_text())
 assert sha(OLD/'original_video_to_delta.py')==template['encoder_source_sha256']
 for name,record in template['model']['downloaded_files'].items():assert sha(MODEL/name)==record['sha256']
 pending=[];reused=[]
 for record in source['episodes']:
  if verify_existing(out,record):reused.append(record['episode'])
  else:pending.append(record)
 assignments={str(g):[] for g in range(4)};totals={str(g):0 for g in range(4)}
 for record in sorted(pending,key=lambda r:r['frames'],reverse=True):
  g=min(totals,key=totals.get);assignments[g].append(record);totals[g]+=record['frames']
 save(SETUP/'batch_assignments.json',assignments)
 children=[]
 for g in range(4):
  if not assignments[str(g)]:continue
  log=(SETUP/f'worker_gpu{g}.log').open('a')
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',str(g)],stdout=log,stderr=subprocess.STDOUT);log.close();children.append((g,p))
 save(status,{'phase':'running','pid':os.getpid(),'workers':{str(g):p.pid for g,p in children},'reused_episodes':reused,'pending_episodes':len(pending),'frames_by_gpu':totals})
 codes={str(g):p.wait() for g,p in children}
 if any(codes.values()):
  save(status,{'phase':'failed','worker_exit_codes':codes,'elapsed_seconds':time.perf_counter()-started});raise RuntimeError(f'Workers failed: {codes}')
 assert all(verify_existing(out,r) for r in source['episodes'])
 assert sha(out/'episode_000001.progress.csv')==source['pilot_csv_sha256']
 assert sha(out/'episode_000001-delta.txt')==source['pilot_delta_sha256']
 save(status,{'phase':'computed_pending_final_validation','worker_exit_codes':codes,'total_episodes':len(source['episodes']),'total_frames':source['total_frames'],'new_episodes':len(pending),'reused_episodes':reused,'elapsed_seconds':time.perf_counter()-started})
 print('All requested episodes computed; final validation and synchronization remain.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);a=p.parse_args()
 if a.worker is None:parent()
 else:worker(a.worker)
