from pathlib import Path
import json,hashlib
from safetensors.torch import load_file
RUN=Path('/cache/cvpzhr/setup/robometer-20261031');DATA=Path('/cache/cvpzhr/datasets/robometer-proprimem-5fps-w8-20261031');SOURCE=Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030')
m=json.loads((DATA/'manifest.json').read_text());report={};candidates=[];source_sets=[]
for split in ('train','validation'):
 count=0;previous=None;history=[];original=[];sources=set();episodes=0;seen=set()
 for line in (DATA/f'{split}.jsonl').open():
  row=json.loads(line)
  if row['episode_id']!=previous:
   assert row['episode_id'] not in seen;seen.add(row['episode_id']);previous=row['episode_id'];history=[];episodes+=1
   original=[json.loads(s) for s in (SOURCE/'cache'/split/row['episode_id']/'samples.jsonl').read_text().splitlines()]
   cache=load_file(row['pixel_cache']);tokens=len(cache['input_input_ids']);patches=cache['pixels'].shape[1]
  i=len(history);history.append(row['frame_index']);expected=[max(0,j) for j in range(i-7,i+1)]
  assert row['window_indices']==expected and row['history_frame_indices']==[history[j] for j in expected]
  assert row['pt']==original[i]['pt'] and row['frame_index']==original[i]['frame_index']
  assert all(j<=row['frame_index'] for j in row['history_frame_indices'])
  if i==len(original)-1 and split=='train':candidates.append(dict(tokens=tokens,patches=patches,row=row))
  sources.add(row['source_id']);count+=1
 assert count==m['splits'][split]['samples'];assert hashlib.sha256((DATA/f'{split}.jsonl').read_bytes()).hexdigest()==m['splits'][split]['sha256']
 source_sets.append(sources);report[split]=dict(samples=count,episodes=episodes,original_videos=len(sources),tasks=m['splits'][split]['tasks'])
assert not source_sets[0]&source_sets[1]
(RUN/'benchmark-candidates.json').write_text(json.dumps(candidates))
report.update(all_pt_equal_to_source=True,all_windows_causal=True,first_frame_and_round_reset_passed=True,original_video_splits_disjoint=True,pixel_cache_bytes=sum(x['pixel_bytes'] for s in m['splits'].values() for x in s['episodes']))
(RUN/'data-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
