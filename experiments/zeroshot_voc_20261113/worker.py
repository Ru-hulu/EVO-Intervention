"""One GPU, official weights, resumable complete episodes; labels never enter inference."""
import argparse
import fcntl
import gc
import re
import sys
import time
import traceback
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from common import *


@lru_cache(maxsize=256)
def image(path):
    from PIL import Image
    if path is None:
        return Image.new('RGB', (64,64), (0,0,0))
    with Image.open(path) as im:
        return im.convert('RGB')


def parse(text, tag, lo, hi):
    m = re.search(r'<' + tag + r'>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*</' + tag + '>', text, re.I)
    if not m and tag == 'vlac':
        m = re.fullmatch(r'\s*(?:<score>)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*(?:</score>)?\s*', text)
    if not m:
        return None
    n = float(m.group(1))
    return n/100 if math.isfinite(n) and lo <= n <= hi else None


def images_for(kind, e, i):
    fs=e['frames']; current=fs[i]['image']
    if kind=='grm': return [fs[0]['image'][0],fs[-1]['image'][0],*fs[0]['image'],*current]
    if kind=='proc': return [fs[max(0,j)]['image'][0] for j in range(i-7,i+1)]
    if kind=='robometer': return [fs[j]['image'][0] for j in uniform_indices(i)[:7]]+current
    if kind=='r2': return [current[0]]
    if kind=='vlac': return [fs[max(0,i-1)]['image'][0],current[0]]
    raise ValueError(kind)


class Generative:
    def __init__(self, cfg):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        self.kind=cfg['kind']
        self.processor=AutoProcessor.from_pretrained(cfg['path'],local_files_only=True)
        mm={k:cfg[k] for k in ['min_pixels','max_pixels'] if k in cfg}
        self.engine=LLM(model=cfg['path'],dtype='bfloat16',trust_remote_code=True,
            gpu_memory_utilization=.90,max_model_len=16384,max_num_seqs=cfg['batch'],
            max_num_batched_tokens=16384,limit_mm_per_prompt={'image':1 if self.kind=='r2' else 8},
            enable_prefix_caching=True,seed=42,mm_processor_kwargs=mm)
        self.sampling=SamplingParams(temperature=0,max_tokens=cfg['max_tokens'])
        self.template=(ROOT / {'grm':'grm_prompt.txt','proc':'proc_prompt.txt','r2':'r2_prompt.txt'}[self.kind]).read_text()
        self.pool=ThreadPoolExecutor(12)

    def request(self, item):
        e,i,history=item
        paths=images_for(self.kind,e,i)
        if self.kind=='grm':
            text=self.template.format(task=e['task']);content=[]
            parts=text.split('<image>'); assert len(parts)==9
            for j,part in enumerate(parts):
                content.append({'type':'text','text':part})
                if j<8:content.append({'type':'image'})
        elif self.kind=='proc':
            text=self.template.replace('{task}',e['task'])
            content=[{'type':'image'} for _ in paths]+[{'type':'text','text':text}]
        else:
            text=self.template.format(history_cot=history)
            content=[{'type':'text','text':text}]+[{'type':'image'}]+[{'type':'text','text':f"\nQuestion: {e['task']}\nOutput: "}]
        prompt=self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        return {'prompt':prompt,'multi_modal_data':{'image':[image(p) for p in paths]}}

    def infer(self, items):
        requests=list(self.pool.map(self.request,items))
        outputs=self.engine.generate(requests,self.sampling,use_tqdm=False)
        raw=[r.outputs[0].text for r in outputs]
        tag={'grm':'score','proc':'progress','r2':'answer'}[self.kind]
        values=[parse(s,tag,-100 if self.kind=='grm' else 0,100) for s in raw]
        return values,raw


class Robometer:
    def __init__(self,cfg):
        import torch
        sys.path[:0]=[str(ROOT/'runtime'),'/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
        from robometer_training.model import load_model,load_processor,current_logits,progress_values,progress_batch
        from robometer_training.data import ProgressCollator
        self.torch=torch
        self.processor=load_processor(cfg['path'],min_pixels=4096,max_pixels=65536)
        self.model,_=load_model(cfg['path'],self.processor)
        assert not any('lora_' in name for name,_ in self.model.named_parameters())
        self.model.to('cuda').eval().requires_grad_(False)
        self.collator=ProgressCollator(self.processor,4096)
        self.logits,self.values,self.raw_batch=current_logits,progress_values,progress_batch

    def infer(self,items):
        rr=[]
        for e,i,_ in items:
            rr.append(dict(pixel_cache=e['pixel_cache'],input_views=['cam_high']*8+['cam_left_wrist','cam_right_wrist'],
                           window_indices=uniform_indices(i)[:7]+[i]*3))
        batch=self.collator.encode(rr,supervised=False)
        with self.torch.inference_mode(), self.torch.autocast('cuda',dtype=self.torch.bfloat16):
            logits=self.logits(self.model,{k:v.to('cuda') for k,v in batch.items()})
            assert self.torch.isfinite(logits).all()
            values=self.values(logits).cpu().tolist()
        return values,[None]*len(values)

    def verify_cache(self,e):
        i=len(e['frames'])//2
        cached,_=self.infer([(e,i,None)])
        raw=self.raw_batch(self.processor,[(e['task'],[image(p) for p in images_for('robometer',e,i)])])
        with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16):
            value=self.values(self.logits(self.model,{k:v.to('cuda') for k,v in raw.items()})).item()
        assert abs(value-cached[0])<.002,(value,cached)
        return dict(raw=value,cached=cached[0],absolute_difference=abs(value-cached[0]))


class VLAC:
    def __init__(self,cfg):
        sys.path.insert(0,'/cache/robot-baselines/repos/VLAC')
        from evo_vlac import GAC_model
        self.model=GAC_model(tag='critic')
        self.model.init_model(model_path=cfg['path'],model_type='internvl2',device_map='cuda:0')
        self.model.temperature=0.;self.model.do_sample=False;self.model.top_k=1
        self.model.logprobs=False;self.model.max_tokens=cfg['max_tokens'];self.model.set_config();self.model.set_system_prompt()

    def infer(self,items):
        prompts=[self.model.get_score_prompt(task=e['task'],trajectory_len=0,think=False) for e,i,h in items]
        images=[[image(p) for p in images_for('vlac',e,i)] for e,i,h in items]
        requests=self.model.get_infer_requests(prompt=prompts,images=images)
        responses,_=self.model.chat(requests)
        raw=[r.choices[0].message.content for r in responses]
        return [parse(s,'vlac',-100,100) for s in raw],raw


INITIAL_HISTORY=('This is the initial step, with no history CoT available. Please generate an initial task decomposition '
                 'based on your understanding of the task and the observed video content, identifying completed and '
                 'incomplete steps. Output in the format specified above.')


def main(model,shard):
    import torch
    torch.set_num_threads(4)
    cfg=MODELS[model];kind=cfg['kind'];job=f'{model}-{shard}'
    lock=(ROOT/'jobs'/f'{job}.lock');lock.parent.mkdir(exist_ok=True)
    lockfile=lock.open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'jobs'/f'{job}.json';start=time.time()
    state=dict(status='loading',model=model,shard=shard,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],started_at=now(),completed_samples=0)
    def update(**kw):state.update(kw,updated_at=now());write(path,state)
    update()
    try:
        manifest=read(ROOT/'manifest.json')
        selected=[e for e in manifest['episodes'] if e['shard']==shard]
        pending=[];done=0
        for e in selected:
            output=episode_path(model,e['episode_id'])
            if output.exists():
                r=read(output)
                assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['manifest_sha256']==sha(ROOT/'manifest.json')
                assert r['target_pt']==[f['pt'] for f in e['frames']]
                done+=len(e['frames'])
            else:pending.append(e)
        total=sum(len(e['frames']) for e in selected)
        update(total_samples=total,completed_samples=done,total_episodes=len(selected))
        if not pending: update(status='completed',finished_at=now());return
        engine=Robometer(cfg) if kind=='robometer' else VLAC(cfg) if kind=='vlac' else Generative(cfg)
        audit=dict(model=model,checkpoint=cfg['path'],official_only=True,local_adapter_loaded=False,
                   config_sha256=sha(Path(cfg['path'])/'config.json'),torch=torch.__version__,
                   protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'manifest.json'))
        if kind=='robometer':audit['cached_raw_agreement']=engine.verify_cache(pending[0])
        smoke=[(next(e for e in manifest['episodes'] if e['suite']==s),0,INITIAL_HISTORY) for s in ['State','Order','Count']]
        sp,sr=engine.infer(smoke)
        assert len(sp)==3 and any(v is not None for v in sp),'All three smoke responses malformed: '+repr(sr)
        audit['smoke']=[dict(suite=e['suite'],value=v,raw_response=r) for (e,i,h),v,r in zip(smoke,sp,sr)]
        write(ROOT/'jobs'/f'{job}-audit.json',audit)
        update(status='running',loaded_at=now(),smoke_passed=True)
        t0=time.time();new_samples=0;invalid=0
        def save_episode(e,pred,raw):
            assert len(pred)==len(e['frames'])
            write(episode_path(model,e['episode_id']),dict(model=model,episode_id=e['episode_id'],
                  checkpoint=cfg['path'],protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'manifest.json'),
                  predictions=pred,raw_responses=raw,target_pt=[f['pt'] for f in e['frames']],
                  frame_indices=[f['frame_index'] for f in e['frames']],completed_at=now()))
        def progress(n,values):
            nonlocal done,new_samples,invalid
            done+=n;new_samples+=n;invalid+=sum(v is None for v in values)
            update(completed_samples=done,invalid_predictions=invalid,samples_per_second=new_samples/max(time.time()-t0,1e-6),
                   elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        batch=cfg['batch']
        if kind=='r2':
            waiting=iter(pending);active=[]
            def fill():
                while len(active)<batch:
                    e=next(waiting,None)
                    if e is None:break
                    active.append(dict(e=e,i=0,history=INITIAL_HISTORY,pred=[],raw=[]))
            fill()
            while active:
                values,raw=engine.infer([(a['e'],a['i'],a['history']) for a in active])
                for a,v,text in zip(active,values,raw):
                    a['pred'].append(v);a['raw'].append(text);a['i']+=1
                    m=re.search(r'<think>(.*?)</think>',text,re.S)
                    a['history']=m.group(1).strip() if m else text
                    if a['i']==len(a['e']['frames']):save_episode(a['e'],a['pred'],a['raw'])
                progress(len(active),values)
                active[:]=[a for a in active if a['i']<len(a['e']['frames'])];fill()
        else:
            for e in pending:
                pred,raw=([0.],[None]) if kind=='vlac' else ([],[])
                if kind=='vlac':progress(1,[0.])
                i=len(pred)
                while i<len(e['frames']):
                    items=[(e,j,None) for j in range(i,min(i+batch,len(e['frames'])))]
                    try:values,answers=engine.infer(items)
                    except torch.cuda.OutOfMemoryError:
                        if batch==1:raise
                        batch=max(1,batch//2);gc.collect();torch.cuda.empty_cache();update(batch_size=batch);continue
                    if kind=='vlac':
                        # Exact native critic_to_value_simple(simple=True), in unit rather than percentage scale.
                        for delta in values:pred.append(pred[-1]+(1-pred[-1])*delta if pred[-1] is not None and delta is not None else None)
                    else:pred.extend(values)
                    raw.extend(answers);i+=len(items);progress(len(items),pred[-len(items):])
                save_episode(e,pred,raw)
        assert done==total
        update(status='completed',finished_at=now(),elapsed_seconds=time.time()-start)
    except BaseException:
        update(status='failed',error=traceback.format_exc(),finished_at=now());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=list(MODELS));p.add_argument('--shard',type=int,required=True)
    a=p.parse_args();main(a.model,a.shard)
