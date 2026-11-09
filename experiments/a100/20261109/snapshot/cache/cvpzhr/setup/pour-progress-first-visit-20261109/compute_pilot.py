"""First pour and its put-down earn raw visual-delta credit once per target cup."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,sys,time,traceback
from collections import Counter

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def save(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def frame_labels(ann):
 n=ann['total_frames'];labels=[None]*n;counts=Counter();previous=None
 assert ann['fps']==30 and ann['frame_index_base']==0 and ann['frame_interval']=='inclusive'
 for j,s in enumerate(ann['stages'],1):
  assert s['segment_index']==j
  if s['action']=='pour':
   target=s['target_position'];assert target in ['left','middle','right']
   counts[target]+=1;occ=counts[target];previous=(target,occ)
  else:
   assert s['action']=='put_down' and previous is not None
   target,occ=previous;assert s['after_pour_target']==target
   assert ann['stages'][j-2]['action']=='pour'
  assert s['target_occurrence']==occ
  first=occ==1
  data={'stage_id':s['stage_id'],'segment_index':j,'stage_name':s['stage_name'],'target_position':target,'target_occurrence':occ,'count_for_progress':int(first),'mode':'first_visit' if first else 'repeat_hold'}
  a,b=s['source_start_frame'],s['source_end_frame'];assert 0<=a<=b<n
  for i in range(a,b+1):assert labels[i] is None;labels[i]=data
 for s in ann['idle_intervals']:
  for i in range(s['source_start_frame'],s['source_end_frame']+1):
   assert labels[i] is None
   labels[i]={'stage_id':0,'segment_index':0,'stage_name':'空闲／等待','target_position':'','target_occurrence':0,'count_for_progress':0,'mode':'idle_hold'}
 assert all(x is not None for x in labels)
 assert dict(counts)==ann['target_pour_counts'] and set(counts)=={'left','middle','right'}
 for target in counts:
  stages=[s for s in ann['stages'] if (s['target_position'] or s['after_pour_target'])==target and s['target_occurrence']==1]
  assert [s['action'] for s in stages]==['pour','put_down'], 'Pilot requires an explicit first pour/put-down pair per cup'
 return labels

def calculate(deltas,ann):
 labels=frame_labels(ann)
 assert len(deltas)==len(labels) and deltas[0]==0 and all(math.isfinite(d) and d>=0 for d in deltas)
 credited=[d*row['count_for_progress'] for d,row in zip(deltas,labels)]
 denominator=math.fsum(credited);assert denominator>0
 pt=[];cumulative=0.
 for d in credited:cumulative+=d;pt.append(cumulative/denominator)
 last=max(i for i,r in enumerate(labels) if r['count_for_progress'])
 assert math.isclose(pt[last],1,abs_tol=1e-12)
 pt[last:]=[1.]*(len(pt)-last)
 assert pt[0]==0 and all(0<=p<=1 for p in pt)
 for i in range(1,len(pt)):
  assert pt[i]>=pt[i-1]
  if not labels[i]['count_for_progress']:assert pt[i]==pt[i-1]
 reports=[]
 for s in ann['stages']:
  a,b=s['source_start_frame'],s['source_end_frame'];row=labels[a]
  raw=math.fsum(deltas[a:b+1]);credit=math.fsum(credited[a:b+1])
  if row['count_for_progress']:assert credit>0
  reports.append({**s,'count_for_progress':row['count_for_progress'],'mode':row['mode'],'raw_delta_sum':raw,'credited_delta_sum':credit,'pt_before':pt[a-1] if a else 0.,'pt_end':pt[b]})
 return labels,credited,pt,{'normalization_denominator':denominator,'completion_frame':last,'completion_time_seconds':last/ann['fps'],'stages':reports,'cup_weights':{t:math.fsum(d for d,r in zip(credited,labels) if r['target_position']==t)/denominator for t in ['left','middle','right']}}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset-root',type=Path,required=True);parser.add_argument('--episode',required=True);parser.add_argument('--gpu',default='0');args=parser.parse_args()
 setup=Path(__file__).resolve().parent;status=setup/'status.json';started=time.perf_counter()
 def update(**kw):save(status,{'episode':args.episode,'elapsed_seconds':time.perf_counter()-started,**kw})
 try:
  root=args.dataset_root;video=root/'videos/main_view'/f'{args.episode}.mp4';ap=root/'videos/main_view/annotation'/f'{args.episode}.json';ann=json.loads(ap.read_text());labels=frame_labels(ann)
  assert video.name==ann['video_file'] and sha(video)==ann['video_sha256']
  out=root/'progress_first_visit';out.mkdir(exist_ok=True)
  assert not (out/f'{args.episode}.progress.csv').exists(), 'Do not overwrite an existing result'
  old=Path('/cache/cvpzhr/setup/flipcup-progress-20261105');model=Path('/cache/cvpzhr/models/Qwen3-VL-4B-Instruct')
  sys.path[:0]=[str(old/'pydeps'),str(old)]
  os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['OMP_NUM_THREADS']='4';os.environ['TOKENIZERS_PARALLELISM']='false'
  import av,torch,importlib.metadata
  from original_video_to_delta import QwenDelta
  assert importlib.metadata.version('transformers')=='4.57.6'
  torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
  update(phase='loading_model',total_frames=ann['total_frames'])
  model_manifest=json.loads((model/'download_manifest.json').read_text())
  for name,spec in model_manifest['downloaded_files'].items():assert sha(model/name)==spec['sha256'],name
  encoder=QwenDelta(model,'cuda',262144);deltas=[];previous=None
  with av.open(str(video)) as container:
   stream=container.streams.video[0];stream.codec_context.thread_count=2
   assert float(stream.average_rate)==ann['fps']
   for i,frame in enumerate(container.decode(stream)):
    current=torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1)
    d=0. if previous is None else encoder.image2delta(previous,current)
    assert math.isfinite(d) and d>=0
    deltas.append(float(f'{d:.6f}'));previous=current
    if i%150==0:update(phase='computing_delta',frames=i+1,total_frames=ann['total_frames'])
  assert len(deltas)==ann['total_frames']
  labels,credited,pt,details=calculate(deltas,ann)
  delta_path=out/f'{args.episode}-delta.txt';delta_path.write_text(''.join(f'{d:.6f}\n' for d in deltas))
  csvpath=out/f'{args.episode}.progress.csv';tmp=csvpath.with_suffix('.csv.tmp')
  fields=['frame_index','time_seconds','stage_id','segment_index','stage_name','target_position','target_occurrence','mode','count_for_progress','delta','credited_delta','pt']
  with tmp.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for i,(label,d,c,p) in enumerate(zip(labels,deltas,credited,pt)):
    writer.writerow({'frame_index':i,'time_seconds':i/ann['fps'],**label,'delta':d,'credited_delta':c,'pt':p})
  tmp.replace(csvpath)
  metadata={'schema_version':'repeated-pouring-first-visit-progress-v1','episode':args.episode,'status':'pilot_pending_user_review','fps':ann['fps'],'total_frames':len(pt),'video_path':str(video),'video_sha256':sha(video),'annotation_path':str(ap),'annotation_sha256':sha(ap),'model':model_manifest,'feature':'Qwen3-VL-4B-Instruct frozen post-merger spatial tokens, FP32 mean','delta_rule':'L2 between adjacent original-frame features; BF16 encoder; FP32 pooling/distance; first delta=0; six-decimal raw values','progress_rule':'PT[t] = sum(delta[i] * first_visit_mask[i], i<=t) / sum(delta[i] * first_visit_mask[i], all i). First pour AND its put-down count once per target cup; subsequent visits and idle earn zero credit. Cups are weighted by accepted visual change, not equally.','boundary_rule':'Existing inclusive 30FPS intervals mapped from 5FPS annotation; delta at stage first frame belongs to that stage. Boundary uncertainty remains +/-0.2 seconds.','raw_delta_retained_for_repeat_and_idle':True,'min_pixels':4096,'max_pixels':262144,'delta_sha256':sha(delta_path),'progress_csv_sha256':sha(csvpath),'encoder_source_sha256':sha(old/'original_video_to_delta.py'),'implementation_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-started,'checks':{'all_original_frames_present':True,'monotonic':True,'repeated_pour_and_putdown_hold_exactly':True,'idle_holds_exactly':True,'final_progress':pt[-1]},**details}
  save(out/f'{args.episode}.metadata.json',metadata)
  update(phase='completed',frames=len(pt),output_dir=str(out),cup_weights=details['cup_weights'])
  print(json.dumps({'status':'completed','frames':len(pt),'output_dir':str(out),'cup_weights':details['cup_weights']},ensure_ascii=False),flush=True)
 except Exception as e:
  update(phase='failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
