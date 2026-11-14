"""Strictly restore each final suite adapter; every call is current-observation only."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import os
from .common import *
from .protocol import current_robometer_messages,parse_progress

@lru_cache(maxsize=512)
def image(path):
    from PIL import Image
    if path is None:return Image.new('RGB',(64,64),0)
    with Image.open(path) as im:return im.convert('RGB')

class Backend:
    def __init__(self,info):
        import torch
        self.name=info['model'];self.cfg=info['config'];self.pool=ThreadPoolExecutor(max_workers=4)
        assert sha(info['proof'])==info['proof_sha256']
        if self.name=='forest':
            from mem_lite.runtime import load_checkpoint
            from mem_lite.train_stage3 import trainable_state
            from mem_lite.datasets.m3_inputs import M3Collator
            from experiments.prm_five_groups_20261114.forest_adapter import state_digest
            assert sha(info['checkpoint'])==info['checkpoint_sha256']
            ck=torch.load(info['checkpoint'],map_location='cpu',weights_only=True)
            assert ck['epoch']==1 and ck['next_batch']==0 and ck['config']==self.cfg
            assert self.cfg['run_protocol']['suite']==info['suite']
            assert self.cfg['run_protocol']['train_episodes']==35
            expected=state_digest(ck['model']);del ck
            self.model,self.processor,config=load_checkpoint(info['checkpoint'],device='cuda',event_split_method='max_depth')
            assert config==self.cfg and state_digest(trainable_state(self.model))==expected
            self.collator=M3Collator(self.processor,config['model']['history_dim'],config['max_sequence_length'],config['history_capacity'])
            # No streaming predictor or memory manager is ever constructed.
            self.memory=None
        else:
            from experiments.soc35_finetune_20261114.train import load_model
            os.environ['RANK']='0'
            for p,h in self.cfg['weights_sha256'].items():assert sha(p)==h
            self.model,self.processor,initial=load_model(self.cfg)
            ck=Path(info['checkpoint']);assert sha(ck/'trainable-state.pt')==info['checkpoint_sha256']
            state=torch.load(ck/'trainable-state.pt',map_location='cpu',weights_only=True)
            current={n:v for n,v in self.model.named_parameters() if v.requires_grad}
            assert set(current)==set(state)
            with torch.no_grad():
                for n,p in current.items():
                    assert p.shape==state[n].shape and p.dtype==state[n].dtype
                    p.copy_(state[n])
            expected=read(ck/'trainable-state.json')['sha256']
            assert tensor_digest(current.items())==expected
            del state,current
            self.model.to('cuda')
            if self.name=='robometer-4b':
                from experiments.soc35_finetune_20261114.data import robometer_api
                self.api=robometer_api()
        self.model.eval().requires_grad_(False)
        self.audit=dict(model=self.name,suite=info['suite'],checkpoint=info['checkpoint'],
            checkpoint_sha256=info['checkpoint_sha256'],loaded_trainable_state_sha256=expected,
            strict_final_state=True,epoch=1,memory_enabled=False,history_length=0,
            dropout_disabled=True,previous_text_used=False,future_endpoint_reference=False,
            input_boundary=['current_images','task_prompt'],backend_sha256=sha(__file__))

    def encode(self,samples):
        import torch
        assert all(set(s)=={'images','task'} for s in samples)
        expected=1 if self.name=='procvlm-2b' else 3
        assert all(len(s['images'])==expected and s['images'][0] for s in samples)
        flat=list(self.pool.map(image,[p for s in samples for p in s['images']]))
        if self.name=='forest':
            batch=self.collator([dict(task_prompt=s['task'],images=flat[i*3:(i+1)*3],history_tokens=None)
                for i,s in enumerate(samples)])
            assert self.memory is None
            assert not batch['history_valid_mask'].any() and not batch['history_mask'].any()
            assert not batch['history_tokens'].count_nonzero()
        elif self.name=='robometer-4b':
            self.processor.tokenizer.padding_side='right'
            texts=[self.processor.apply_chat_template(current_robometer_messages(self.api.progress_messages(s['task'],10)),
                tokenize=False,add_generation_prompt=False,add_vision_id=True) for s in samples]
            batch=dict(self.processor(text=texts,images=flat,return_tensors='pt',padding=True))
            token=self.processor.tokenizer.convert_tokens_to_ids('<|prog_token|>')
            assert ((batch['input_ids']==token).sum(dim=1)==1).all()
        else:
            from experiments.soc35_model_caches_20261114.proc_data import user_prompt
            self.processor.tokenizer.padding_side='left'
            texts=[self.processor.apply_chat_template([dict(role='user',content=[
                dict(type='image'),dict(type='text',text=user_prompt(s['task']))])],
                tokenize=False,add_generation_prompt=True) for s in samples]
            batch=dict(self.processor(text=texts,images=flat,return_tensors='pt',padding=True,
                min_pixels=self.cfg['processor_config']['min_pixels'],max_pixels=self.cfg['processor_config']['max_pixels']))
        assert 'labels' not in batch and 'past_key_values' not in batch
        assert batch['image_grid_thw'].shape[0]==expected*len(samples)
        assert batch['input_ids'].shape[1]<=self.cfg['max_sequence_length' if self.name=='forest' else 'max_length']
        lengths=batch['attention_mask'].sum(dim=1).tolist()
        fingerprints=[]
        for i in range(len(samples)):
            ids=batch['input_ids'][i][batch['attention_mask'][i].bool()].tolist()
            fingerprints.append(hashlib.sha256(repr((samples[i],ids)).encode()).hexdigest())
        return batch,lengths,fingerprints

    def predict(self,samples):
        import torch
        batch,lengths,fingerprints=self.encode(samples)
        batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            if self.name=='forest':
                values=self.model(batch)['pt'].float().cpu().tolist();raw=[None]*len(samples)
            elif self.name=='robometer-4b':
                output,_=self.model(**batch,sample_type='progress')
                logits=output.progress_logits['A']
                # Native RBM removes the time axis when there is just one PT
                # token. The training helper's [:,-1] would select a bin here.
                if logits.ndim==3:
                    assert logits.shape[1]==1
                    logits=logits[:,0,:]
                assert logits.ndim==2 and logits.shape==(len(samples),10),logits.shape
                values=self.api.progress_values(logits).float().cpu().tolist();raw=[None]*len(samples)
            else:
                width=batch['input_ids'].shape[1]
                output=self.model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True,
                    temperature=None,top_p=None,top_k=None)
                raw=self.processor.tokenizer.batch_decode(output[:,width:],skip_special_tokens=True)
                values=[parse_progress(s) for s in raw]
        return [dict(prediction=p,raw_response=r,input_tokens=int(n),input_fingerprint=f)
            for p,r,n,f in zip(values,raw,lengths,fingerprints)]
