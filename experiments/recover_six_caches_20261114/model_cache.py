"""CPU-only official preprocessing; lossless PT is separate from native prompts.

Pixel tensors are stored once per source observation. Each training row gathers
the native window/pair through explicit indices, avoiding repeated image copies.
No invented reasoning or unverified relative-score response is emitted.
"""
import argparse
import os
import sys
import time

from .common import *


def setup(model):
    import torch
    from PIL import Image
    import importlib.util
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('recover_native',OLD/'worker.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    cfg=dict(native.MODELS[model])
    if model=='robometer-4b':
        sys.path[:0]=[str(OLD/'runtime'),'/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
        from robometer_training.model import load_processor,progress_batch
        processor=load_processor(cfg['path'],min_pixels=4096,max_pixels=65536)
        def encode(task,paths):
            return dict(progress_batch(processor,[(task,[native.image(p) for p in paths])]))
        prompt=lambda task:None
    elif model=='vlac-2b':
        from swift.llm import get_model_tokenizer,get_template
        from swift.llm.template.template.internvl import transform_image
        sys.path.insert(0,'/cache/robot-baselines/repos/VLAC')
        from evo_vlac import GAC_model
        m,processor=get_model_tokenizer(cfg['path'],model_type='internvl2',torch_dtype=torch.bfloat16,load_model=False)
        assert m is None
        template=get_template(processor.model_meta.template,processor)
        template.set_mode('pt')
        model_shell=GAC_model('critic');model_shell.set_system_prompt()
        def encode(task,paths):
            text=model_shell.get_score_prompt(task,trajectory_len=0,think=False)
            request=model_shell.get_infer_requests(text,[native.image(p) for p in paths])[0]
            result=template.encode(request)
            return {k:(torch.tensor(v) if isinstance(v,list) else v) for k,v in result.items() if v is not None and k in ['input_ids','attention_mask','pixel_values']}
        def pixel(im):
            # GAC_model resizes to 448-square BEFORE the Swift image transform.
            return {'pixel_values':transform_image(model_shell._process_image_to_pil(im),448,12).to(torch.bfloat16)}
        prompt=lambda task:dict(system=model_shell.system_prompt,user=model_shell.get_score_prompt(task,trajectory_len=0,think=False))
        return cfg,processor,encode,pixel,prompt
    else:
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(cfg['path'],local_files_only=True)
        for k in ['min_pixels','max_pixels']:
            if k in cfg:setattr(processor.image_processor,k,cfg[k])
        # The inference engine forwards these keyword overrides to the official processor.
        opts={k:cfg[k] for k in ['min_pixels','max_pixels'] if k in cfg}
        obj=object.__new__(native.Generative);obj.kind=cfg['kind'];obj.processor=processor
        obj.template=(OLD/('grm_prompt.txt' if cfg['kind']=='grm' else 'proc_prompt.txt')).read_text()
        def prompt(task):
            e={'task':task,'frames':[{'image':[None,None,None]}]}
            return obj.request((e,0,None))['prompt']
        def encode(task,paths):
            return dict(processor(text=[prompt(task)],images=[native.image(p) for p in paths],return_tensors='pt',**opts))
    def pixel(im):
        opts={k:cfg[k] for k in ['min_pixels','max_pixels'] if k in cfg}
        return dict(processor.image_processor(images=[im],return_tensors='pt',**opts))
    return cfg,processor,encode,pixel,prompt


def smoke(model):
    import torch
    from PIL import Image
    selection=read(ROOT/'selection.json');row=selection['records'][0]
    observed=rows(epdir(row)/'observations.jsonl');n=len(observed)
    cfg,processor,encode,pixel,prompt=setup(model)
    checked=[]
    for i in [0,n//2,n-1]:
        paths=[observed[j]['images'][v] for j,v in slots(model,i,n)]
        online=encode(row['task_prompt'],paths);stored=[]
        for p in paths:
            with Image.open(p) as image:stored.append(pixel(image.convert('RGB')))
        for k in ['pixel_values','image_grid_thw']:
            if k in online:
                a=torch.cat([s[k] for s in stored]);b=online[k]
                assert torch.equal(a.to(torch.bfloat16) if a.is_floating_point() else a,
                                   b.to(torch.bfloat16) if b.is_floating_point() else b),(model,k,a.shape,b.shape)
        checked.append(i)
    save(ROOT/f'{model}-cpu-smoke.json',dict(status='passed',model=model,sample_indices=checked,
         official_preprocessing_equal=True,model_forward_executed=False,cuda_used=False))


def main(model):
    import torch
    from PIL import Image
    from safetensors.torch import save_file,load_file
    selection=read(ROOT/'resolved-selection.json');source_checks(selection)
    cfg,processor,encode,pixel,prompt=setup(model)
    all_records=[];receipts=[];total_bytes=0
    for row in selection['records']:
        observed=rows(epdir(row)/'samples.jsonl');n=len(observed)
        views=range(3) if model in ['robometer-4b','robobrain25-4b','dopamine-3b'] else [0]
        pixel_hashes={}
        for i,r in enumerate(observed):
            if i%30==0:save(ROOT/f'{model}-status.json',dict(status='preprocessing',pid=os.getpid(),key=row['key'],
                observations=i,completed_episodes=len(receipts),updated=time.time()))
            for v in views:
                p=pixel_path(model,row,i,v);p.parent.mkdir(parents=True,exist_ok=True)
                assert not p.exists(),f'Preserve prior pixels: {p}'
                with Image.open(r['images'][v]) as image:batch=pixel(image.convert('RGB'))
                tensors={k:val.detach().contiguous().to(torch.bfloat16) if val.is_floating_point() else val.contiguous()
                         for k,val in batch.items() if isinstance(val,torch.Tensor)}
                assert all(torch.isfinite(v).all() for v in tensors.values())
                save_file(tensors,str(p),metadata={'source_png_sha256':sha(r['images'][v]),'model':model})
                pixel_hashes[str(p)]=sha(p);total_bytes+=p.stat().st_size
        prepared=[];canonical=None;checked=0
        for i,r in enumerate(observed):
            pairs=slots(model,i,n);images=[observed[j]['images'][v] for j,v in pairs]
            caches=[str(pixel_path(model,row,j,v)) for j,v in pairs]
            # Different frame content must not change template geometry or token indices.
            if i in {0,n//2,n-1}:
                online=encode(row['task_prompt'],images)
                stored=[load_file(p) for p in caches]
                for key in ['pixel_values','image_grid_thw']:
                    if key in online:
                        actual=torch.cat([s[key] for s in stored])
                        expected=online[key].to(actual.dtype)
                        assert torch.equal(actual,expected),(model,row['key'],i,key,actual.shape,expected.shape)
                text={k:v.squeeze(0).contiguous() for k,v in online.items()
                      if isinstance(v,torch.Tensor) and k not in ['pixel_values','image_grid_thw']}
                if canonical is None:canonical=text
                else:
                    assert canonical.keys()==text.keys()
                    for key in text:assert torch.equal(canonical[key],text[key]),(model,key,'template changed')
                checked+=1
            record=dict(id=f"{row['key']}/{i:06d}",episode_id=row['key'],task_id=row['task_id'],split='train',
                sample_index=i,frame_index=r['frame_index'],pt=r['pt'],task=row['task_prompt'],image=images,
                slots=[list(p) for p in pairs],pixel_caches=caches,
                prompt_tensor_file=str(OUT/model/'prompts'/row['task_id']/(row['sample_id']+'.safetensors')),
                source_progress_sha256=row['progress_sha256'],selection_sha256=sha(ROOT/'selection.json'),
                uses_future_endpoint_reference=model in ['robobrain25-4b','dopamine-3b'])
            if model=='vlac-2b':
                record.update(before_pt=observed[max(0,i-1)]['pt'],after_pt=r['pt'],
                    transition=i>0,critic_target=None,critic_target_status='not_assigned_by_input_cache',
                    initial_progress=observed[0]['pt'])
            if model in ['robobrain25-4b','dopamine-3b']:
                record.update(before_pt=observed[0]['pt'],after_pt=r['pt'])
            if model=='procvlm-2b':record.update(previous_cot_input=False,remaining_actions=None,assistant_target=None)
            prepared.append(record)
        target=Path(prepared[0]['prompt_tensor_file']);target.parent.mkdir(parents=True,exist_ok=True)
        save_file(canonical,str(target),metadata={'model':model,'task_prompt':row['task_prompt']})
        save(target.with_suffix('.json'),dict(native_prompt=prompt(row['task_prompt']),sha256=sha(target)))
        dest=OUT/model/'episodes'/row['task_id']/(row['sample_id']+'.jsonl');write_rows(dest,prepared)
        receipt=dict(key=row['key'],samples=n,transitions=n-1 if model=='vlac-2b' else None,
            samples_file=str(dest),samples_sha256=sha(dest),pixel_sha256=pixel_hashes,
            native_cached_input_equal_examples=checked,prompt_sha256=sha(target))
        save(dest.with_suffix('.verification.json'),receipt)
        receipts.append({k:v for k,v in receipt.items() if k!='pixel_sha256'});all_records.extend(prepared)
    train=OUT/model/'train.jsonl';write_rows(train,all_records)
    source_checks(selection)
    save(OUT/model/'cache-ready.json',dict(status='input_cache_ready',episodes=10,samples=len(all_records),
         model=model,checkpoint=cfg['path'],checkpoint_config_sha256=sha(Path(cfg['path'])/'config.json'),
         processor_only=True,model_forward_executed=False,original_pt_preserved=True,
         raw_supervision_ready=True,sft_response_targets_assigned=False,
         train_sha256=sha(train),pixel_bytes=total_bytes,episodes_verified=receipts,
         cuda_used=False,native_cached_input_equality_verified=True))
    save(ROOT/f'{model}-status.json',dict(status='completed',episodes=10,samples=len(all_records),updated=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=MODELS[:-1]);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();smoke(a.model) if a.smoke else main(a.model)
