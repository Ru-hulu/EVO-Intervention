"""Measure batch-one image encoding separately from the complete Robometer request.

Only creates benchmark artifacts. Does not alter training, caches or checkpoints.
Run on one idle GPU with the uniform8 runtime and pinned upstream on PYTHONPATH.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import transformers

from robometer_training.model import load_adapter, load_model, load_processor, current_logits, progress_values, progress_batch


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--iterations',type=int,default=50);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.config).read_text());torch.set_num_threads(4)
    rows=[json.loads(x) for x in Path(config['validation_data']).read_text().splitlines()]
    episode=[r for r in rows if r['episode_id']==rows[0]['episode_id']];row=episode[len(episode)//2]
    images=[]
    for name in row['image']:
        with Image.open(name) as im:images.append(im.convert('RGB'))
    processor=load_processor(config['model_path'],min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
    model,_=load_model(config['model_path'],processor)
    checkpoint=Path(config['output_dir'])/f"checkpoint-{config['expected_steps']}"
    load_adapter(model,checkpoint);model.requires_grad_(False).to('cuda').eval()
    visual=model.model.visual
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,
                cuda=torch.version.cuda,checkpoint=str(checkpoint),sample_id=row['id'],
                sample_batch_size=1,min_pixels=config['min_pixels'],max_pixels=config['max_pixels'],
                dtype=str(visual.dtype),vision_attention=visual.config._attn_implementation,
                vision_layers=len(visual.blocks),warmup=10,iterations=a.iterations,
                vision_scope='GPU-resident normalized pixels -> projected image embeddings and all DeepStack outputs',
                excluded='model load, camera capture/transport, disk decode; CPU preprocessing/H2D excluded from vision-only cases',
                cases={})

    def save():
        temp=out/'report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'report.json')

    def measure(fn,iterations=None):
        for _ in range(10):fn()
        torch.cuda.synchronize();wall=[];gpu=[]
        for _ in range(iterations or a.iterations):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();t=time.perf_counter();start.record();value=fn();end.record();end.synchronize()
            wall.append((time.perf_counter()-t)*1000);gpu.append(start.elapsed_time(end));del value
        return dict(wall_ms=stats(wall),cuda_elapsed_ms=stats(gpu),iterations=len(wall))

    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        batches={}
        for count,subset in [(1,[images[7]]),(3,images[7:]),(8,images[:8]),(10,images)]:
            raw=dict(processor.image_processor(images=subset,return_tensors='pt'))
            batch={k:v.to('cuda') for k,v in raw.items()};batch['pixel_values']=batch['pixel_values'].to(visual.dtype)
            batches[count]=batch
            def encode(batch=batch):return visual(batch['pixel_values'],grid_thw=batch['image_grid_thw'])
            result=measure(encode)
            result['grids']=raw['image_grid_thw'].tolist()
            result['post_merge_tokens']=int(raw['image_grid_thw'].prod(-1).sum()//visual.spatial_merge_size**2)
            report['cases'][f'vision_batched_{count}']=result;save();print(json.dumps({f'vision_batched_{count}':result}),flush=True)

        for count in [8,10]:
            batch=batches[count];lengths=batch['image_grid_thw'].prod(-1).tolist()
            pixels=torch.split(batch['pixel_values'],lengths)
            def serial():
                return [visual(part,grid_thw=batch['image_grid_thw'][i:i+1]) for i,part in enumerate(pixels)]
            result=measure(serial,20);report['cases'][f'vision_serial_{count}']=result;save()
            print(json.dumps({f'vision_serial_{count}':result}),flush=True)

        # Count actual SDPA invocations, separately from timing. Restore immediately.
        calls=[0];original=F.scaled_dot_product_attention
        def counted(*args,**kwargs):
            calls[0]+=1;return original(*args,**kwargs)
        try:
            F.scaled_dot_product_attention=counted
            visual(batches[10]['pixel_values'],grid_thw=batches[10]['image_grid_thw'])
            torch.cuda.synchronize()
        finally:F.scaled_dot_product_attention=original
        report['sdpa_calls_for_ten_images']=calls[0]
        report['vision_source']=inspect.getfile(type(visual))
        report['vision_source_sha256']=hashlib.sha256(Path(report['vision_source']).read_bytes()).hexdigest()

        cpu=dict(progress_batch(processor,[(row['task'],images)]))
        inputs={k:v.to('cuda') for k,v in cpu.items()}
        report['language_input_tokens']=int(inputs['input_ids'].shape[-1])
        def full():return progress_values(current_logits(model,inputs))
        report['cases']['full_forward_10']=measure(full)
        prep=[];copy=[];forward=[];total=[]
        for _ in range(30):
            torch.cuda.synchronize();started=time.perf_counter()
            raw=dict(progress_batch(processor,[(row['task'],images)]));prepared=time.perf_counter()
            inp={k:v.to('cuda') for k,v in raw.items()};torch.cuda.synchronize();copied=time.perf_counter()
            prediction=float(progress_values(current_logits(model,inp))[0].cpu());ended=time.perf_counter()
            prep.append((prepared-started)*1000);copy.append((copied-prepared)*1000)
            forward.append((ended-copied)*1000);total.append((ended-started)*1000)
        report['online_10_in_memory_rgb']=dict(cpu_prepare_ms=stats(prep),h2d_ms=stats(copy),
                                             forward_and_readback_ms=stats(forward),total_ms=stats(total),
                                             prediction=prediction,iterations=len(total))
        report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        report['status']='completed';save();print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
