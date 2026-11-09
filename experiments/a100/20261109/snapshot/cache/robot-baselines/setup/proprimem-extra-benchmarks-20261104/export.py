from pathlib import Path
import json,os,re,av
r=Path('/cache/robot-baselines');s=Path(__file__).parent;plan=json.loads((s/'plan.json').read_text());ds=r/'viewer-datasets/ProPriMem-Extra-VLA-20261104';ds.mkdir(exist_ok=True);rows=[]
def add(task,key,episode,success,sources,note):
 folder=ds/key;folder.mkdir(exist_ok=True);views={}
 for view,src in sources.items():
  if not src.exists():continue
  p=folder/(view+'.mp4')
  if not p.exists():os.link(src,p)
  views[view]={'video_path':str(p.relative_to(ds))}
 with av.open(str(sources['main_view'])) as c:
  st=c.streams.video[0];fps=float(st.average_rate);n=st.frames or sum(1 for _ in c.decode(video=0))
 rows.append(dict(source_repo=task.get("source_repo"),environment=task.get("environment"),sim_task=task.get("sim_task"),sample_id=key,task_id=task['task_id'],task_name=task['task_id']+'｜'+task['task_name'],model_name='SimpleMemVLA',episode_index=episode,task_success=success,failure_reason='SimpleMemVLA｜'+('成功' if success else '失败'),annotation_path=None,annotation_status='pending_native_event_annotation',task_prompt=(task.get('source_instruction') or task['task_name'])+'。'+note,fps=fps,total_frames=n,video_path=views['main_view']['video_path'],video_views=views))
for task in [t for t in plan['tasks'] if t['environment']=='RoboDojo']:
 run_id='proprimem-'+('xylophone' if task['sim_task']=='play_Xylophone' else task['sim_task'])+'-10-20261104'
 for f in (r/'repos/RoboDojo/eval_result/RoboDojo'/task['sim_task']).rglob('*_cam_head_*.mp4'):
  if run_id not in str(f):continue
  episode=int(re.search(r'episode_(\d+)',f.name).group(1));add(task,task['task_id']+'-simplemem-'+str(episode),episode,'_success' in f.stem,{'main_view':f,'left_view':f.with_name(f.name.replace('cam_head','cam_left_wrist')),'right_view':f.with_name(f.name.replace('cam_head','cam_right_wrist'))},'RoboDojo 对应权重；原生任务判定。')
for f in (r/'eval-results/proprimem-extra-20261104/simplemem-libero-mem/videos').rglob('*.mp4'):
 match=re.search(r'libero_mem_t(\d+)__ep(\d+)__(succ|fail)',f.stem)
 if not match:continue
 task=next(t for t in plan['tasks'] if t.get('sim_task')=='libero_mem/'+match[1]);episode=int(match[2]);add(task,task['task_id']+'-simplemem-'+str(episode),episode,match[3]=='succ',{'main_view':f},'普通 LIBERO 权重迁移到 LIBERO-Mem，零样本测试；使用新生成的原生随机初始状态。')
p=ds/'metadata.jsonl.tmp';p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));p.replace(ds/'metadata.jsonl');print('EXPORTED',len(rows))
