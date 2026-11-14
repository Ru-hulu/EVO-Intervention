from copy import deepcopy
import pytest
import torch
from PIL import Image

from experiments.prm_five_groups_20261114.forest_adapter import (
    check_final_proof,check_metadata,state_digest,ForestRuntime)
from experiments.prm_five_groups_20261114.forest_eval import verify_prefix


def test_only_matching_final_epoch_checkpoint_is_eligible():
    proof={'status':'passed','suite':'State','trial_only':False,'epoch':1,'step':3830,
        'optimizer_scheduler_rng_verified':True,'fresh_initialization_verified_both_ranks':True,
        'checkpoint':'/tmp/State/epoch-01.pt'}
    check_final_proof(proof,'State','/tmp/State/epoch-01.pt')
    for changed in [dict(proof,trial_only=True),dict(proof,suite='Count'),dict(proof,epoch=0),
                    dict(proof,checkpoint='/tmp/State/last.pt'),dict(proof,fresh_initialization_verified_both_ranks=False)]:
        with pytest.raises(AssertionError):check_final_proof(changed,'State','/tmp/State/epoch-01.pt')


def test_checkpoint_configuration_and_resume_state_are_verified():
    config={'run_protocol':{'suite':'State','trial_only':False,'trial_weights_reused':False,
        'artificial_augmentation':False,'epochs':1,'world_size':2,'effective_batch':18,'fps':5},
        'total_optimizer_steps':10,'base_revision':'official','model_path':'base','seed':42,'epochs':1,
        'learning_rate':1e-4,'lr_schedule':'cosine','warmup_ratio':.03,'data_signature':'fixed train',
        'history_source':{'memory_config':{'event_split_method':'max_depth'}},'event_split_method':'max_depth',
        'history_capacity':1008,'model':{'history_dim':2560},'max_pixels':65536,'max_sequence_length':8192}
    plan={'forest_training':{'base_revision':'official','base_path':'base',
                            'memory_config':{'event_split_method':'max_depth'}}}
    ckpt={'format':'mem-lite-streamforest-v1','config':config,'epoch':1,'next_batch':0,'step':10,
          'scheduler':{'last_epoch':10},'optimizer':{'state':{0:{}}},'rng_by_rank':[{},{}]}
    check_metadata(ckpt,'State',plan,deepcopy(config))
    for field,value in [('step',9),('epoch',0),('rng_by_rank',[{}]),('optimizer',{'state':{}})]:
        bad=deepcopy(ckpt);bad[field]=value
        with pytest.raises(AssertionError):check_metadata(bad,'State',plan,config)
    train=deepcopy(config);train['data_signature']='another training split'
    with pytest.raises(AssertionError):check_metadata(ckpt,'State',plan,train)


def test_loaded_trainable_digest_detects_missing_keys_values_and_dtype():
    state={'m3_lora':{'x':torch.tensor([1.,2.])},'progress_head':{'weight':torch.ones(2,2)}}
    original=state_digest(state);assert original==state_digest(deepcopy(state))
    bad=deepcopy(state);bad['m3_lora']['x'][0]=0
    assert state_digest(bad)!=original
    bad=deepcopy(state);bad['m3_lora']['x']=bad['m3_lora']['x'].double()
    assert state_digest(bad)!=original
    bad=deepcopy(state);del bad['m3_lora']['x']
    assert state_digest(bad)!=original


def test_prefix_rebuild_and_no_memory_preserve_native_interfaces(monkeypatch):
    import mem_lite.runtime as native
    import mem_lite.datasets.m3_inputs as inputs
    class Stream:
        def __init__(self):self.values=[]
        def step(self,image,position,episode,prompt):
            past=list(self.values);self.values.append(image.getpixel((0,0))[0]);return past
    class Runner:
        def __init__(self,*a,**kw):self.calls=0;self.reset()
        def reset(self):self.stream=Stream()
        def __call__(self,images,position,episode,prompt):
            self.calls+=1;past=self.stream.step(images[0],position,episode,prompt)
            return {'pt':torch.tensor([sum(past)/100]),'history_length':len(past)}
    records=[]
    class Collator:
        def __init__(self,*a,**kw):pass
        def __call__(self,samples):
            records.extend(samples);return {'history_valid_mask':torch.zeros(1,1,dtype=torch.bool)}
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=0
        def forward(self,batch):self.calls+=1;return {'pt':torch.tensor([.3])}
    monkeypatch.setattr(native,'StreamingProgressPredictor',Runner)
    monkeypatch.setattr(inputs,'M3Collator',Collator)
    config={'history_source':{'memory_config':{'clip_model_path':'frozen'}},'max_sequence_length':8192,
            'model':{'history_dim':2560},'history_capacity':1008}
    images=[[Image.new('RGB',(2,2),(i,0,0)),None,None] for i in (1,2,3)]
    full=ForestRuntime(Model(),None,config)
    values=[full.predict(im,i*6,'episode','goal') for i,im in enumerate(images)]
    resumed=ForestRuntime(Model(),None,config)
    for i in range(2):resumed.advance_memory(images[i],i*6,'episode','goal')
    assert resumed.predict(images[2],12,'episode','goal')==values[2]
    assert resumed.runner.calls==1 and full.runner.calls==3
    with pytest.raises(ValueError):resumed.predict(images[0],6,'episode','goal')
    assert resumed.predict(images[0],0,'new episode','goal')['history_length']==0
    no_memory=ForestRuntime(Model(),None,config,memory_enabled=False)
    no_memory.advance_memory(images[0],0,'episode','goal')
    assert no_memory.runner is None and no_memory.model.calls==0
    assert no_memory.predict(images[1],6,'episode','goal')['history_length']==0
    assert all(set(r)=={'images','task_prompt','history_tokens'} and r['history_tokens'] is None for r in records)


def test_resume_rejects_changed_checkpoint_truth_or_incomplete_alignment():
    row={'key':'episode','frame_indices':[0,6,12],'target_pt':[0,.2,1]}
    prefix={'episode':'episode','manifest_sha256':'manifest','checkpoint_sha256':'final',
        'frame_indices':[0,6],'target_pt':[0,.2],'predictions':[.1,.3],
        'history_lengths':[0,64],'raw_nonfinite':[None,None]}
    assert verify_prefix(prefix,row,'manifest','final')==2
    for field,value in [('checkpoint_sha256','trial'),('target_pt',[0,.5]),('frame_indices',[0,12]),
                        ('predictions',[.1,.3,.4,.5])]:
        bad=deepcopy(prefix);bad[field]=value
        with pytest.raises(AssertionError):verify_prefix(bad,row,'manifest','final')
