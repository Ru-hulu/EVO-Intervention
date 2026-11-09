from pathlib import Path
import json,os,av,fcntl
r=Path('/cache/robot-baselines');base=r/'eval-results/vla-render-validation-20261104';ds=r/'viewer-datasets/ProPriMem-RenderAB-20261104';ds.mkdir(exist_ok=True)
tasks={t['sim_task']:t for t in json.loads((r/'setup/proprimem-rmbench-20261104/plan.json').read_text())['tasks']};models={'simplemem':'SimpleMemVLA','full_attention':'IMPRINT 完整注意力','recurrent':'IMPRINT 循环记忆'}
with (ds/'.export.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);rows=[]
 for result in sorted(base.glob('*/*/rt*/result.json')):
  x=json.loads(result.read_text());profile=result.parent.name
  if profile not in ['rt32','rt8']:continue
  t=tasks[x['task']];e=Path(x['evidence']);meta=json.loads((e/'evidence.json').read_text());key=t['task_id']+'-'+x['model']+'-'+profile;folder=ds/key;folder.mkdir(exist_ok=True);views={}
  for view,camera in zip(['main_view','left_view','right_view'],['head_camera','left_camera','right_camera']):
   src=e/(camera+'.mp4');p=folder/(view+'.mp4');assert src.is_file()
   if not p.exists():os.link(src,p)
   with av.open(str(p)) as c:assert c.streams.video[0].frames==meta['frames']
   views[view]={'video_path':str(p.relative_to(ds))}
  label=models[x['model']]+' · '+str(x['spp'])+' 次光追'
  rows.append(dict(sample_id=key,task_id=t['task_id'],task_name=t['task_name'],model_name=label,episode_index=list(models).index(x['model'])*2+int(x['spp']==8),fps=meta['fps'],total_frames=meta['frames'],video_path=views['main_view']['video_path'],video_views=views,task_success=x['success'],failure_reason=label+'｜'+('成功' if x['success'] else '失败'),annotation_path=None,task_prompt=t['source_instruction']+'\n32/8 次采样对照验证；不计入 210 次正式采集，尚无训练标注。'+f" 场景 seed={x['seed']}，动作步数={x['steps']}。",renderer_spp=x['spp'],actual_seed=x['seed'],eval_output=str(result.parent),evidence_path=str(e),source='Huawei paired renderer validation'))
 p=ds/'metadata.jsonl.tmp';p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));p.replace(ds/'metadata.jsonl');print('EXPORTED',len(rows),'rollouts',len(rows)*3,'views')
p=r/'viewer-app/catalog_names.json';x=json.loads(p.read_text());x[ds.name]='渲染对照｜RMBench｜32次 / 8次光追';p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
