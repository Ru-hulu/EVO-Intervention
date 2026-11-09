"""Create portable progress-overlay previews; retain all raw camera videos."""
import json
import subprocess
import sys
from pathlib import Path

root=Path(sys.argv[1])
for row in map(json.loads,(root/'metadata.jsonl').read_text().splitlines()):
    folder=root/row['sample_id']
    p=json.loads((root/'progress'/(row['sample_id']+'.json')).read_text())
    r=json.loads((folder/'result.json').read_text())
    filters=['drawbox=x=0:y=0:w=iw:h=28:color=black@0.72:t=fill']
    events=p['physical_events']
    for i,event in enumerate(events):
        start=event['frame'];end=events[i+1]['frame']-1 if i+1<len(events) else row['total_frames']-1
        label=f"P = {event['progress']:.3f}"
        filters.append(f"drawtext=text='{label}':x=7:y=6:fontsize=16:fontcolor=white:enable='between(n,{start},{end})'")
    marks=[(0,'HISTORY')]+[(e['frame'],{'memory_error':'MEMORY ERROR','memory_retrieval':'HISTORY RECALL','restored':'RECOVERED'}[e['kind']])
                    for e in r['events'] if e['kind'] in ['memory_error','memory_retrieval','restored']]+[(row['total_frames']-1,'NATIVE SUCCESS')]
    if r.get('recording_tail'):
        v=json.loads((folder/'validation.json').read_text())
        marks[-1:] = [(v['first_native_success_frame'],'NATIVE CHECK OK'),(row['total_frames']-1,'PLACED / RELEASED')]
    for i,(start,label) in enumerate(marks):
        end=marks[i+1][0]-1 if i+1<len(marks) else row['total_frames']-1
        filters.append(f"drawtext=text='{label}':x=w-tw-7:y=8:fontsize=12:fontcolor=0x85e5bf:enable='between(n,{start},{end})'")
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(folder/'main_view.mp4'),'-vf',','.join(filters),
                    '-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                    str(folder/'progress_preview.mp4')],check=True)
    print(row['sample_id'],flush=True)
