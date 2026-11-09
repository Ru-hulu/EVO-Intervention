from pathlib import Path
import json,os,sys,subprocess,time,fcntl,re
R=Path('/cache/robot-baselines'); S=R/'setup/proprimem-rmbench-20261104'; OUT=R/'eval-results/proprimem-rmbench-20261104'; DS=R/'viewer-datasets/ProPriMem-RMBench-3Models-20261104'
PLAN=json.loads((S/'plan.json').read_text());MODELS={'simplemem':'SimpleMemVLA','full_attention':'IMPRINT 完整注意力','recurrent':'IMPRINT 循环记忆'}
NAMES={'C06':'按数字按键','S01':'盖住再揭开','S02':'放回原位','S03':'移到绿垫再放回','S04':'电池方向试错','S05':'积木排列试错','S06':'重新排列'}
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def export():
 import av
 DS.mkdir(parents=True,exist_ok=True)
 with (DS/'.export.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);rows=[]
  for task in PLAN['tasks']:
   for idx,model in enumerate(MODELS):
    d=OUT/model/task['sim_task'];status=d/'status.json'
    if not status.exists():continue
    state=json.loads(status.read_text())
    if state['status']!='complete':continue
    key=task['task_id']+'-'+model
    if model=='simplemem':
     videos=list((d/'videos'/task['sim_task']).glob('*.mp4'))
     if len(videos)!=1:continue
     sources={'main_view':videos[0]}
     success='__succ' in videos[0].name
    else:
     cameras={'main_view':'head_camera','left_view':'left_camera','right_view':'right_camera','third_view':'third_person','mosaic':'mosaic'}
     sources={v:d/'videos'/('trial_000_'+c+'.mp4') for v,c in cameras.items()}
     if not all(p.exists() for p in sources.values()):continue
     success=json.loads((d/'summary.json').read_text())['successes']==1
    folder=DS/key;folder.mkdir(exist_ok=True);views={}
    for view,src in sources.items():
     dest=folder/(view+'.mp4')
     if not dest.exists():os.link(src,dest)
     views[view]={'video_path':str(dest.relative_to(DS))}
    with av.open(str(sources['main_view'])) as c:
     st=c.streams.video[0];fps=float(st.average_rate);frames=st.frames or sum(1 for _ in c.decode(video=0))
    rows.append(dict(sample_id=key,task_id=task['task_id'],task_name=task['task_id']+'｜'+NAMES[task['task_id']],model_name=MODELS[model],episode_index=idx,video_path=views['main_view']['video_path'],video_views=views,fps=fps,total_frames=frames,task_success=success,annotation_path=None,failure_reason=MODELS[model],task_prompt=task['source_instruction']+'\n模型：'+MODELS[model]+'；起始 seed=40，实际接受 seed 见评估记录。'+(' S03 超出 IMPRINT 官方九任务范围。' if task['task_id']=='S03' and model!='simplemem' else ''),source='HF ProPriMem '+PLAN['hf_revision'],eval_output=str(d),requested_start_seed=40))
  tmp=DS/'metadata.jsonl.tmp';tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));tmp.replace(DS/'metadata.jsonl')
  print('VIEWER_EXPORTED',len(rows),'of 21',flush=True)
def run(model,gpu):
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME=str(R/'.cache/openpi'),HF_HOME=str(R/'.cache/huggingface'),HF_HUB_CACHE=str(R/'.cache/huggingface/hub'))
 tasks=sorted(PLAN['tasks'],key=lambda t:['S02','S06','S04','C06','S01','S03','S05'].index(t['task_id']))
 for task in tasks:
  d=OUT/model/task['sim_task'];status=d/'status.json'
  if status.exists() and json.loads(status.read_text())['status']=='complete':continue
  d.mkdir(parents=True,exist_ok=True)
  if model=='simplemem':
   cwd=R/'repos/SimpleMemVLA';env.update(PYBIN=str(R/'envs/simplemem-rmbench/bin/python'),CHECKPOINT=str(R/'models/SimpleMemVLA/RMBench'),GPUS=str(gpu),NUM_GPUS='1',SEEDS_PER_TASK='1',GROUP_SIZE='1',TASKS=task['sim_task'],RUN_DIR=str(d),VIDEO_MAX_PER_TASK='1',ATTN_IMPLEMENTATION='flash_attention_2',INSTRUCTION_TYPE='unseen')
   cmd=['bash','scripts/eval_rmbench.sh','--base_seed','40','--no_seed_cache']
  else:
   cwd=R/'repos/imprint';cmd=[str(R/'envs/imprint/bin/python'),'benchmarks/rmbench/eval/run_local.py','--rmbench-root',str(R/'dependencies/EventVLA/RoboTwin-Mem'),'--checkpoint',str(R/f'models/IMPRINT/simulation/rmbench/{model}/stage2/step-100000'),'--dinov3-model',str(R/'models/DINOv3-ViT-B16'),'--architecture',model,'--task',task['sim_task'],'--trials','1','--start-seed','40','--output',str(d/'run'),'--policy-python',str(R/'envs/imprint/bin/python'),'--sim-python',str(R/'envs/rmbench/bin/python'),'--policy-gpu',str(gpu),'--sim-gpu',str(gpu),'--port',str(8050+int(gpu))]
  started=time.time();write(status,dict(status='running',model=model,task=task['sim_task'],gpu=gpu,start=started,command=cmd));print('START',model,task['sim_task'],flush=True)
  with (d/'launch.log').open('w') as log:
   result=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
  if model!='simplemem' and (d/'run').exists():
   for p in (d/'run').iterdir():
    target=d/p.name
    if not target.exists():target.symlink_to(p.relative_to(d))
  complete=result.returncode==0
  if model=='simplemem':complete=complete and bool(list((d/'videos'/task['sim_task']).glob('*.mp4')))
  else:complete=complete and (d/'summary.json').exists() and (d/'videos/trial_000_head_camera.mp4').exists()
  write(status,dict(status='complete' if complete else 'failed',exit_code=result.returncode,model=model,task=task['sim_task'],gpu=gpu,start=started,end=time.time(),seconds=time.time()-started))
  print('END',model,task['sim_task'],'complete' if complete else 'failed',time.time()-started,flush=True);export()
if __name__=='__main__':
 if sys.argv[1]=='export':export()
 else:run(sys.argv[1],sys.argv[2])
