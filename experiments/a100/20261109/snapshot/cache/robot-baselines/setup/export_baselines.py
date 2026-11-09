from pathlib import Path
import json,av
r=Path('/cache/robot-baselines');out=r/'viewer-datasets/Baseline-Huawei-20261104';out.mkdir(parents=True,exist_ok=True)
e=r/'eval-results'
items=[('01-simplemem-libero','SimpleMemVLA｜LIBERO Spatial/0｜成功',True,e/'simplemem-libero/videos/libero_spatial/libero_spatial_t0__ep0__succ.mp4',None),('02-simplemem-rmbench','SimpleMemVLA｜RMBench 观察后抓取｜未成功',False,e/'simplemem-rmbench/videos/observe_and_pickup/observe_and_pickup__seed100000__fail.mp4',None),('03-simplemem-robodojo','SimpleMemVLA｜RoboDojo 叠碗｜成功',True,e/'simplemem-robodojo/videos/episode_0000000_cam_head_success.mp4',{'left_view':e/'simplemem-robodojo/videos/episode_0000000_cam_left_wrist_success.mp4','right_view':e/'simplemem-robodojo/videos/episode_0000000_cam_right_wrist_success.mp4'}),('04-imprint-full-attention','IMPRINT 完整注意力｜RMBench 观察后抓取｜未成功',False,e/'imprint-full_attention/videos/trial_000_head_camera.mp4',{'left_view':e/'imprint-full_attention/videos/trial_000_left_camera.mp4','right_view':e/'imprint-full_attention/videos/trial_000_right_camera.mp4','third_view':e/'imprint-full_attention/videos/trial_000_third_person.mp4','mosaic':e/'imprint-full_attention/videos/trial_000_mosaic.mp4'}),('05-imprint-recurrent','IMPRINT 循环记忆｜RMBench 观察后抓取｜未成功',False,e/'imprint-recurrent/videos/trial_000_head_camera.mp4',{'left_view':e/'imprint-recurrent/videos/trial_000_left_camera.mp4','right_view':e/'imprint-recurrent/videos/trial_000_right_camera.mp4','third_view':e/'imprint-recurrent/videos/trial_000_third_person.mp4','mosaic':e/'imprint-recurrent/videos/trial_000_mosaic.mp4'})]
rows=[]
for key,name,success,main,others in items:
 views={}
 folder=out/key;folder.mkdir(exist_ok=True)
 for view,src in {'main_view':main,**(others or {})}.items():
  assert src.is_file(),src
  dest=folder/(view+'.mp4')
  # Keep source videos in their evaluation directories; viewer metadata stays portable within its allowed root.
  if not dest.exists():
   import os
   os.link(src,dest)
  views[view]={'video_path':str(dest.relative_to(out))}
 with av.open(str(main)) as c:
  stream=c.streams.video[0];fps=float(stream.average_rate);frames=stream.frames or sum(1 for _ in c.decode(video=0))
 rows.append(dict(sample_id=key,task_id=key,task_name=name,episode_index=0,video_path=views['main_view']['video_path'],video_views=views,fps=fps,total_frames=frames,task_prompt=name+'。2026-11-04 华为云 A100 完整闭环验证；仅一回合，不代表整体性能。',task_success=success,failure_reason='模型评估回合',annotation_path=None,source='Huawei A100 2026-11-04'))
(out/'metadata.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
print('EXPORTED',len(rows),'rollouts',sum(len(x['video_views']) for x in rows),'camera streams')
