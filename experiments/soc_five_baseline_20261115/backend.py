"""Exact singleton native prediction with bounded asynchronous CPU preparation."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
from .common import *


@lru_cache(maxsize=64)
def image(path):
    from PIL import Image
    if path is None:return Image.new('RGB',(64,64),(0,0,0))
    with Image.open(path) as im:return im.convert('RGB')


class Backend:
    def __init__(self,info):
        from experiments.group2_nomemory_20261115.backend import Backend as Loader
        from experiments.prm_five_groups_20261114.official_baseline import native_module
        self.loader=Loader(info);self.name=info['model'];self.cfg=info['config']
        self.model=self.loader.model;self.processor=self.loader.processor
        self.native=native_module() if self.name in MODELS[:2] else None
        if self.native:
            self.shell=object.__new__(self.native.Generative);self.shell.kind='grm';self.shell.processor=self.processor
            self.shell.template=(self.native.ROOT/'grm_prompt.txt').read_text()
            assert self.native.MODELS[self.name]['path']==self.cfg['model_path']
        self.audit=dict(self.loader.audit)
        self.audit.update(memory_enabled=self.name in ('procvlm-2b','robometer-4b'),
            future_endpoint_reference=bool(self.native),previous_text_used=False,
            input_boundary=['task','images'],batch=1,backend_sha256=sha(__file__),
            execution='bounded CPU prefetch; singleton unmodified model forward')
        self.pool=ThreadPoolExecutor(max_workers=1);self.pending={};self.ep=None
        self.processor.tokenizer.padding_side='left' if self.name=='procvlm-2b' else self.processor.tokenizer.padding_side

    def drain(self):
        for future in self.pending.values():future.result()
        self.pending.clear()

    def begin(self,episode):
        assert set(episode)=={'key','task','images','positions'}
        self.drain();self.ep=episode;image.cache_clear()

    def prepare(self,i):
        e=self.ep;ss=slots(self.name,i,len(e['images']))
        paths=[e['images'][j][v] for j,v in ss];images=[image(p) for p in paths]
        if self.native:
            # Prompt rendering is native; no progress values or labels enter it.
            prompt=self.shell.request((dict(task=e['task'],frames=[dict(image=[None]*3)]),0,None))['prompt']
            options={k:self.native.MODELS[self.name][k] for k in ('min_pixels','max_pixels')}
            batch=dict(self.processor(text=[prompt],images=images,return_tensors='pt',**options))
        elif self.name=='robometer-4b':
            batch=dict(self.loader.api.progress_batch(self.processor,[(e['task'],images)]))
        else:
            from experiments.soc35_model_caches_20261114.proc_data import chat_prompt
            batch=dict(self.processor(text=[chat_prompt(self.processor,e['task'])],images=images,return_tensors='pt',
                min_pixels=4096,max_pixels=65536))
        assert 'labels' not in batch and 'past_key_values' not in batch
        assert batch['image_grid_thw'].shape[0]==len(ss)
        assert batch['input_ids'].shape[1]<=self.cfg['max_length']
        return batch,dict(sample_index=i,frame_index=e['positions'][i],slots=ss,
            input_images=len(ss),input_tokens=int(batch['attention_mask'].sum()))

    def forward(self,prepared):
        import torch
        from experiments.group2_nomemory_20261115.protocol import parse_progress
        batch,audit=prepared
        batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
        # Recovery GRM's native generation did not autocast; its weights and
        # visual inputs use the model's native dtype. Preserve that path exactly.
        if self.native:
            with torch.inference_mode():
                width=batch['input_ids'].shape[1]
                output=self.model.generate(**batch,max_new_tokens=self.native.MODELS[self.name]['max_tokens'],
                    do_sample=False,use_cache=True,temperature=None,top_p=None,top_k=None)
            raw=self.processor.tokenizer.decode(output[0,width:],skip_special_tokens=True)
            value=self.native.parse(raw,'score',-100,100)
        else:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                if self.name=='robometer-4b':
                    logits=self.loader.api.current_logits(self.model,batch)
                    value=float(self.loader.api.progress_values(logits).item())
                    raw=dict(progress_logits=logits.float().cpu().tolist()[0])
                else:
                    width=batch['input_ids'].shape[1]
                    output=self.model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True,
                        temperature=None,top_p=None,top_k=None)
                    raw=self.processor.tokenizer.decode(output[0,width:],skip_special_tokens=True);value=parse_progress(raw)
        if value is not None and not math.isfinite(value):raw=dict(response=raw,nonfinite=repr(value));value=None
        return dict(prediction=value,raw_response=raw,input_audit=audit)

    def predict(self,i):
        for j in range(i,min(i+3,len(self.ep['images']))):
            if j not in self.pending:self.pending[j]=self.pool.submit(self.prepare,j)
        return self.forward(self.pending.pop(i).result())

    def smoke(self,episode):
        """Compare prefetched tensors/outputs against direct native requests."""
        import torch
        self.begin(episode);checks=[]
        for i in sorted({0,min(7,len(episode['images'])-1),len(episode['images'])//2,len(episode['images'])-1}):
            direct=self.prepare(i)
            if self.native:
                native_ep=dict(task=episode['task'],frames=[dict(image=x) for x in episode['images']])
                request=self.shell.request((native_ep,i,None));opts=self.cfg['processor_config']
                native_input=dict(self.processor(text=[request['prompt']],images=request['multi_modal_data']['image'],return_tensors='pt',**opts))
            elif self.name=='procvlm-2b':
                from experiments.soc35_model_caches_20261114.proc_data import online_prompt
                native_input=online_prompt(self.processor,episode['task'],[episode['images'][j][v] for j,v in slots(self.name,i,len(episode['images']))])
            else:
                native_input=direct[0] # unchanged native progress_batch call
            assert set(native_input)==set(direct[0]) and all(torch.equal(native_input[k],direct[0][k]) for k in native_input)
            reference=self.forward((native_input,direct[1]));prefetched=self.pool.submit(self.prepare,i).result()
            assert all(torch.equal(prefetched[0][k],native_input[k]) for k in native_input)
            observed=self.forward(prefetched)
            assert (reference['prediction'] is None)==(observed['prediction'] is None)
            error=0 if reference['prediction'] is None else abs(reference['prediction']-observed['prediction'])
            assert error<=.002
            checks.append(dict(sample_index=i,tensors_equal=True,max_abs_error=error,
                valid=observed['prediction'] is not None,prediction=observed['prediction']))
        assert any(c['valid'] for c in checks),'No valid outputs in the real-input smoke test'
        self.drain()
        return dict(status='passed',checks=checks,initialization=self.audit,batch=1,gpu_forward_tested=True)
