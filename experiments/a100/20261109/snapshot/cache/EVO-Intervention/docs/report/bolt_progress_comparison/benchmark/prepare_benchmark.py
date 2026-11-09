import io
import json
import math
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import av
import h5py
import numpy as np
from PIL import Image, ImageDraw

SOURCE = Path('/root/autodl-tmp/Robot_Benchmark_Samples_2026-10-15')
OUT = Path('/root/autodl-tmp/EvalProgress/outputs/benchmark_15tasks_3fps_20261016')

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def read_images(record):
    benchmark = record['benchmark']
    if benchmark == 'RoboDojo':
        with av.open(str(SOURCE / record['video_file'])) as cap:
            fps = float(cap.streams.video[0].average_rate)
            frames = [[frame.to_image(), None, None] for frame in cap.decode(video=0)]
    else:
        with h5py.File(SOURCE / record['hdf5_file'], 'r') as h:
            if benchmark == 'LIBERO':
                group = h['data'][record['source_demo']]
                fps = float(record['data_frequency_hz'])
                # OpenGL -> upright RGB, matching the official LIBERO video convention.
                frames = [[Image.fromarray(np.flipud(a)), Image.fromarray(np.flipud(b)), None]
                          for a, b in zip(group['obs/agentview_rgb'][:], group['obs/eye_in_hand_rgb'][:])]
            else:
                fps = float(h['additional_info/frequency'][()])
                keys = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
                values = [h[f'vision/{k}/colors'][:] for k in keys]
                frames = [[Image.open(io.BytesIO(bytes(v[i]))).convert('RGB') for v in values]
                          for i in range(len(values[0]))]
    return fps, frames

def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=1)

def prepare(pair):
    pair = sorted(pair, key=lambda r:r['sample_number'])
    query, reference = pair
    slug = query['benchmark'].lower() + '__' + query['task']
    directory = OUT / 'inputs' / slug
    result = {'id': slug, 'benchmark': query['benchmark'], 'task': query['task'],
              'query_record':query, 'reference_record':reference, 'sampling_fps':3,
              'missing_views':'RoboDojo: both wrist slots white; LIBERO: available wrist in left slot, right slot white; RoboTwin: real head/left/right.'}
    for role, record in [('query',query),('reference',reference)]:
        fps, frames = read_images(record)
        n = len(frames)
        grid = sorted(set([round(i*fps/3) for i in range(math.floor((n-1)*3/fps)+1)] + [n-1]))
        # Reference main frames are cached densely for visual selection of action stages.
        chosen = grid if role=='query' else list(range(n))
        blank = directory / role / 'blank.png'
        save_image(Image.new('RGB',frames[0][0].size,'white'),blank)
        paths = {}
        for i in chosen:
            imgs = frames[i] if role=='query' else frames[i][:1]
            row=[]
            for cam,image in zip(['main','left_wrist','right_wrist'],imgs):
                p = directory/role/cam/f'{i:06d}.png'
                if image is None:p=blank
                else:save_image(image,p)
                row.append(str(p))
            paths[str(i)]=row
        result[role]={'fps':fps,'frame_count':n,'last_time_s':(n-1)/fps,'sample_indices':grid,
                      'frames':paths,'blank_path':str(blank),'input_source':'native HDF5 observations' if record['benchmark']!='RoboDojo' else 'original MP4',
                      'preview_path':str(SOURCE/record['video_file'])}
        if role=='reference':
            selected=np.linspace(0,n-1,25,dtype=int).tolist()
            sheet=Image.new('RGB',(1600,5*266+40),'#f3f5f8')
            draw=ImageDraw.Draw(sheet);draw.text((10,8),slug,fill='black')
            for j,i in enumerate(selected):
                im=frames[i][0].copy();im.thumbnail((312,236))
                x=(j%5)*320;y=(j//5)*266+40
                sheet.paste(im,(x+(320-im.width)//2,y))
                draw.text((x+7,y+238),f'{j}: frame {i} / {i/fps:.2f}s',fill='black')
            sheetpath=OUT/'review'/f'{slug}.jpg';sheetpath.parent.mkdir(exist_ok=True)
            sheet.save(sheetpath,quality=87)
            result['review_indices']=selected
        del frames
    dump(directory/'manifest.json',result)
    print('PREPARED',slug,len(result['query']['sample_indices']),flush=True)
    return result

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    records=json.loads((SOURCE/'MANIFEST.json').read_text())['records']
    groups={}
    for record in records:
        if record['task']=='place_mouse_pad':continue
        groups.setdefault((record['benchmark'],record['task']),[]).append(record)
    assert len(groups)==15 and all(len(v)==2 for v in groups.values())
    with ThreadPoolExecutor(max_workers=3) as pool:tasks=list(pool.map(prepare,groups.values()))
    dump(OUT/'prepared.json',{'tasks':tasks,'paused_task':'RoboTwin/place_mouse_pad',
                             'protocol':'Query sample 1; expert sample 2; 3 FPS native-nearest grid plus true final observation; missing wrist views are white blank images.'})
    shutil.copy2(__file__,OUT/'prepare_benchmark.py')
    print('READY',OUT,'total query points',sum(len(t['query']['sample_indices']) for t in tasks),flush=True)
