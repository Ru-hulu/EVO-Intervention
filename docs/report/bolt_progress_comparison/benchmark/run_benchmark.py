"""Native-frame, batched evaluation of the fifteen approved benchmark tasks."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0',
                  VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false')
BASE=Path('/root/autodl-tmp/EvalProgress')
OUT=BASE/'outputs/benchmark_15tasks_3fps_20261016'
CKPT=Path('/root/autodl-tmp/ckpt/baselines')

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@lru_cache(maxsize=4096)
def get_image(path):
    from PIL import Image
    with Image.open(path) as im:return im.convert('RGB')

def parse(raw,signed):
    matches=re.findall(r'<score>\s*(.*?)\s*</score>',raw,re.S|re.I)
    if len(matches)!=1:return None,'parse_error'
    value=matches[0].strip()
    if value.lower() in ['n/a','na']:return None,'not_applicable'
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?',value):return None,'parse_error'
    value=float(value.rstrip('%').strip())
    if not math.isfinite(value) or not (-100 if signed else 0)<=value<=100:return None,'out_of_range'
    return value,'ok'

def jobs_for(tasks,letters):
    jobs=[]
    for t in tasks:
        for letter in letters:
            destination=OUT/'results'/t['id']/f'{letter}.jsonl'
            done=set()
            if destination.exists():
                done={r['frame_index'] for r in map(json.loads,destination.read_text().splitlines())}
            for pos,i in enumerate(t['query']['sample_indices']):
                if i not in done:jobs.append((t,letter,pos,i))
    return jobs

def save_row(job,**kwargs):
    t,letter,pos,i=job
    row={'task_id':t['id'],'method_id':letter,'frame_index':i,'timestamp_s':i/t['query']['fps'],
         'instruction':t['annotation']['goal'],'score_semantics':'absolute_progress' if letter in 'ABC' else 'signed_start_to_current_grm',**kwargs}
    path=OUT/'results'/t['id']/f'{letter}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def run_robometer(tasks):
    import numpy as np
    import torch
    ann=load_module('robometer_annotator',BASE/'Robometer/scripts/annotate_screw.py')
    options=SimpleNamespace(model_path=CKPT/'Robometer',processor_path=CKPT/'Qwen3-VL-4B-Instruct-processor')
    started=time.perf_counter();backend=ann.create_backend(options)
    torch,model,tokenizer,collator,compute,Trajectory,ProgressSample,discrete,bins=backend
    jobs=jobs_for(tasks,'A');print('READY robometer',len(jobs),flush=True)
    init=time.perf_counter()-started
    cursor=0;batch_size=64;timings=[]
    while cursor<len(jobs):
        batch=jobs[cursor:cursor+batch_size];t0=time.perf_counter();samples=[];contexts=[]
        for t,letter,pos,i in batch:
            context=[t['query']['sample_indices'][int(k)] for k in np.linspace(0,pos,4,dtype=int)]
            frames=np.stack([np.asarray(get_image(t['query']['frames'][str(k)][0])) for k in context])
            trajectory=Trajectory(frames=frames,frames_shape=tuple(frames.shape),task=t['annotation']['goal'],
                                  id=f'{t["id"]}:{i}',metadata={'subsequence_length':4},video_embeddings=None)
            samples.append(ProgressSample(trajectory=trajectory,sample_type='progress'));contexts.append(context)
        try:
            inputs=collator(samples)['progress_inputs']
            inputs={k:v.to('cuda') if hasattr(v,'to') else v for k,v in inputs.items()}
            with torch.inference_mode():result=compute(model,tokenizer,inputs,sample_type='progress',is_discrete_mode=discrete,num_bins=bins)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            del samples
            if 'inputs' in locals():del inputs
            torch.cuda.empty_cache()
            if batch_size<=1:raise
            batch_size//=2;print('OOM reduce Robometer batch',batch_size,flush=True);continue
        elapsed=time.perf_counter()-t0
        values=result['progress_pred'];assert len(values)==len(batch)
        for job,v,context in zip(batch,values,contexts):
            score,status=ann.checked_score(float(v[-1]))
            save_row(job,score_percent=None if score is None else 100*score,status=status,context_frame_indices=context)
        cursor+=len(batch)
        peak=torch.cuda.max_memory_allocated()/1024**3
        timings.append({'batch_size':len(batch),'elapsed_s':elapsed,'peak_gpu_gib':peak})
        print('BATCH robometer',cursor,'/',len(jobs),'size',len(batch),'seconds',round(elapsed,2),'peak_gib',round(peak,2),flush=True)
        del inputs,result,samples
        if batch_size<128 and peak<60:batch_size=min(128,batch_size*2)
    return {'model_initialization_s':init,'batches':timings,'backend':'official Robometer heads + transformers SDPA','max_batch_size':max((x['batch_size'] for x in timings),default=0)}

def prompt_constant(path,name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id==name for n in node.targets):return ast.literal_eval(node.value)
    raise ValueError(name)

def run_vllm(tasks,model_name):
    import torch
    torch.set_num_threads(12)
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    if model_name=='progresslm':
        model_path=CKPT/'ProgressLM';letters='BC';max_pixels=307200;max_images=6
        sys.path.insert(0,str(BASE/'ProgressLM/eval/qwen25vl/codes'))
        from prompts.visual_demo_prompt import build_visual_demo_prompt,VISUAL_DEMO_SYSTEM_PROMPT
        from prompts.text_demo_prompt import build_text_demo_prompt
        sampling=SamplingParams(temperature=0,top_p=1,max_tokens=2048,seed=0)
        template=None
    else:
        model_path=CKPT/('RoboBrain2.5' if model_name=='brain' else 'Robo-Dopamine-GRM-2.0-8B-Preview')
        letters='DE' if model_name=='brain' else 'FG';max_pixels=76800;max_images=8
        template=(BASE/'RoboBrain2.5/scripts/robobrain_grm_prompt.txt').read_text() if model_name=='brain' else prompt_constant(BASE/'Robo-Dopamine/examples/inference.py','SYSTEM_PROMPT')
        sampling=SamplingParams(temperature=.1,top_p=.9,top_k=50,max_tokens=1024,seed=0) if model_name=='brain' else SamplingParams(temperature=0,top_p=1,max_tokens=128,seed=0)
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
    prompt_cache={}
    for t in tasks:
        a=t['annotation'];q=t['query'];r=t['reference']
        refs=[r['frames'][str(x['frame_index'])][0] for x in a['reference_anchors']]
        for letter in letters:
            if model_name=='progresslm':
                parts=build_text_demo_prompt(a['goal'],a['text_steps'],4,q['frames']['0'][0]) if letter=='B' else build_visual_demo_prompt(a['goal'],refs,4,q['frames']['0'][0])
                content=[{'type':'text','text':p['value']} if p['type']=='text' else {'type':'image'} for p in parts]
                messages=([{'role':'system','content':VISUAL_DEMO_SYSTEM_PROMPT}] if letter=='C' else [])+[{'role':'user','content':content}]
            else:
                parts=template.format(task=a['goal']).split('<image>');assert len(parts)==9
                content=[]
                for i,p in enumerate(parts):
                    content.append({'type':'text','text':p})
                    if i<8:content.append({'type':'image'})
                messages=[{'role':'user','content':content}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            prompt_cache[t['id'],letter]=prompt
            path=OUT/'prompts'/t['id']/f'{letter}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prompt)
    jobs=jobs_for(tasks,letters)
    started=time.perf_counter()
    llm=LLM(model=str(model_path),dtype='bfloat16',gpu_memory_utilization=.90,max_model_len=12288,
            limit_mm_per_prompt={'image':max_images},enable_prefix_caching=True,
            mm_processor_kwargs={'min_pixels':12544,'max_pixels':max_pixels},
            mm_processor_cache_gb=4,max_num_seqs=128,max_num_batched_tokens=32768,
            enforce_eager=True,seed=0,trust_remote_code=True)
    init=time.perf_counter()-started;print('READY',model_name,len(jobs),'init_s',round(init,2),flush=True)
    def request(job):
        t,letter,pos,i=job;q=t['query'];r=t['reference'];a=t['annotation'];current=q['frames'][str(i)]
        if letter=='B':paths=[current[0]]
        elif letter=='C':paths=[r['frames'][str(x['frame_index'])][0] for x in a['reference_anchors']]+[current[0]]
        else:
            if letter=='D':anchors=[q['frames']['0'][0],q['blank_path']]
            elif letter in 'EF':anchors=[q['frames']['0'][0],q['frames'][str(q['frame_count']-1)][0]]
            else:anchors=[r['frames']['0'][0],r['frames'][str(r['frame_count']-1)][0]]
            paths=anchors+q['frames']['0']+current
            assert len(paths)==8
        return {'prompt':prompt_cache[t['id'],letter],'multi_modal_data':{'image':[get_image(p) for p in paths]}}
    timings=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0,len(jobs),256):
            batch=jobs[start:start+256];t0=time.perf_counter()
            requests=list(pool.map(request,batch));prepared=time.perf_counter()
            results=llm.generate(requests,sampling,use_tqdm=False)
            elapsed=time.perf_counter()-t0;assert len(results)==len(batch)
            for job,result in zip(batch,results):
                answer=result.outputs[0];value,status=parse(answer.text,model_name!='progresslm')
                save_row(job,score_percent=value,status=status,raw_response=answer.text,
                         finish_reason=answer.finish_reason,output_tokens=len(answer.token_ids))
            timings.append({'batch_size':len(batch),'elapsed_s':elapsed,'prepare_s':prepared-t0,
                            'output_tokens':sum(len(x.outputs[0].token_ids) for x in results)})
            print('BATCH',model_name,start+len(batch),'/',len(jobs),'size',len(batch),'seconds',round(elapsed,2),flush=True)
    return {'model_initialization_s':init,'batches':timings,'backend':'vLLM','model_path':str(model_path),
            'gpu_memory_utilization':.90,'max_concurrent_sequences':128,'request_batch':256,
            'max_num_batched_tokens':32768,'min_pixels':12544,'max_pixels':max_pixels,
            'sampling':str(sampling),'prefix_caching':True,'eager':True,
            'source_note':'Official prompt text preserved; batched engine and explicitly bounded image processing used for throughput.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['robometer','progresslm','brain','dopamine'],required=True)
    parser.add_argument('--only-task')
    args=parser.parse_args()
    tasks=json.loads((OUT/'prepared.json').read_text())['tasks']
    if args.only_task:
        tasks=[t for t in tasks if t['id']==args.only_task]
        assert len(tasks)==1
    annotations=json.loads((OUT/'annotations.json').read_text())
    for t in tasks:t['annotation']=annotations[t['id']]
    started=time.perf_counter();print('START MODEL',args.model,flush=True)
    stats=run_robometer(tasks) if args.model=='robometer' else run_vllm(tasks,args.model)
    stats['total_wall_s']=time.perf_counter()-started
    stats['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stats['timing_scope']='Offline batched throughput; not online single-frame latency.'
    stats['outputs']={}
    letters={'robometer':'A','progresslm':'BC','brain':'DE','dopamine':'FG'}[args.model]
    for t in tasks:
        for letter in letters:
            p=OUT/'results'/t['id']/f'{letter}.jsonl';rows=list(map(json.loads,p.read_text().splitlines()))
            assert [r['frame_index'] for r in rows]==t['query']['sample_indices']
            assert all(abs(r['timestamp_s']-r['frame_index']/t['query']['fps'])<1e-8 for r in rows)
            stats['outputs'][f'{t["id"]}/{letter}']={'points':len(rows),'statuses':dict(Counter(r['status'] for r in rows))}
    if args.only_task:
        main_path=OUT/'runtime'/f'{args.model}.json'
        original=json.loads(main_path.read_text())
        original['outputs'].update(stats['outputs'])
        original['total_wall_s']+=stats['total_wall_s']
        original.setdefault('targeted_reruns',[]).append({'task':args.only_task,'reason':'Remove unsupported expert-to-query tray-color mapping; use the shared category-sorting goal.','runtime':stats})
        dump(main_path,original)
    else:
        dump(OUT/'runtime'/f'{args.model}.json',stats)
    shutil.copy2(__file__,OUT/'run_benchmark.py')
    print('COMPLETE MODEL',args.model,'seconds',round(stats['total_wall_s'],2),flush=True)

if __name__=='__main__':main()
