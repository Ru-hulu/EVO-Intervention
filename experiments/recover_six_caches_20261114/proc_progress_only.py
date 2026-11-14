"""Add progress-only assistant supervision to the immutable ProcVLM input cache.

No CoT is generated or teacher-forced. Prompt/image/padding tokens are ignored by
the token loss; only the progress answer and its EOS are supervised. The original
PT and the native percentage-scale regression target are retained independently.
"""
import argparse
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import time

from .common import ROOT, OUT, read, rows, save, sha, write_rows

BASE = OUT/'procvlm-2b'
TARGET = BASE/'progress-only'
POLICY = 'procvlm-progress-only-assistant-v1'


def answer_text(pt):
    if isinstance(pt,bool) or not math.isfinite(pt) or not 0 <= pt <= 1:
        raise ValueError('PT must be finite and in [0,1]')
    with localcontext() as context:
        context.prec=40
        percent=format(Decimal(str(pt))*100,'f')
    if '.' in percent:percent=percent.rstrip('0').rstrip('.')
    return f'<progress>{percent}%</progress>'


def supervised_ids(prompt_ids,answer_ids,max_length):
    if not prompt_ids or not answer_ids or len(prompt_ids)+len(answer_ids)>max_length:
        raise ValueError('Empty input/answer or length limit exceeded; never truncate')
    return list(prompt_ids)+list(answer_ids),[-100]*len(prompt_ids)+list(answer_ids)


class ProgressOnlyCollator:
    def __init__(self,tokenizer,max_length=16384):
        self.tokenizer,self.max_length=tokenizer,max_length

    def __call__(self,records):
        import torch
        from .loader import load_input
        sequences=[];labels=[];pixels=[];grids=[];mm_types=[]
        for row in records:
            if row.get('supervision_policy')!=POLICY or row.get('remaining_actions') is not None:
                raise ValueError('Refusing non-progress-only supervision')
            expected=answer_text(row['pt'])
            if row['assistant_target']!=expected:
                raise ValueError('Answer does not match original PT')
            ids=self.tokenizer.encode(expected+self.tokenizer.eos_token,add_special_tokens=False)
            if ids!=row['answer_token_ids']:raise ValueError('Tokenizer/target changed')
            batch,_=load_input(row)
            required={'input_ids','attention_mask','pixel_values','image_grid_thw'}
            if not required<=set(batch) or set(batch)-required-{'mm_token_type_ids'}:
                raise ValueError('Unexpected native ProcVLM input fields')
            assert batch['attention_mask'].eq(1).all()
            sequence,label=supervised_ids(batch['input_ids'][0].tolist(),ids,self.max_length)
            sequences.append(sequence);labels.append(label)
            pixels.append(batch['pixel_values']);grids.append(batch['image_grid_thw'])
            mm_types.append(batch['mm_token_type_ids'][0].tolist()+[0]*len(ids)
                            if 'mm_token_type_ids' in batch else None)
        width=max(map(len,sequences));n=len(records)
        input_ids=torch.full((n,width),self.tokenizer.pad_token_id,dtype=torch.long)
        attention_mask=torch.zeros((n,width),dtype=torch.long)
        target_ids=torch.full((n,width),-100,dtype=torch.long)
        for i,(sequence,label) in enumerate(zip(sequences,labels)):
            input_ids[i,:len(sequence)]=torch.tensor(sequence)
            attention_mask[i,:len(sequence)]=1
            target_ids[i,:len(sequence)]=torch.tensor(label)
        result=dict(input_ids=input_ids,attention_mask=attention_mask,labels=target_ids,
                    pixel_values=torch.cat(pixels),image_grid_thw=torch.cat(grids),
                    progress_gt=torch.tensor([100*r['pt'] for r in records],dtype=torch.float32))
        if any(t is not None for t in mm_types):
            assert all(t is not None for t in mm_types)
            result['mm_token_type_ids']=torch.zeros((n,width),dtype=torch.long)
            for i,t in enumerate(mm_types):result['mm_token_type_ids'][i,:len(t)]=torch.tensor(t)
        return result


def verify(destination=TARGET):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    from PIL import Image
    from .loader import load_input
    receipt=read(destination/'manifest.json');source=rows(BASE/'train.jsonl');prepared=rows(destination/'train.jsonl')
    assert receipt['source_train_sha256']==sha(BASE/'train.jsonl')
    assert receipt['train_sha256']==sha(destination/'train.jsonl')
    assert len(prepared)==len(source)==2550
    for p,d in read(ROOT/'procvlm-2b-native-provenance.json')['source_sha256'].items():assert sha(p)==d
    processor=AutoProcessor.from_pretrained(receipt['checkpoint'],local_files_only=True)
    tokenizer=processor.tokenizer;collator=ProgressOnlyCollator(tokenizer)
    chosen=[];seen=set();max_sequence=0;supervised_tokens=0
    for a,b in zip(source,prepared):
        assert all(b[k]==v for k,v in a.items() if k!='assistant_target')
        answer=answer_text(a['pt']);assert b['assistant_target']==answer
        value=Decimal(answer.removeprefix('<progress>').removesuffix('%</progress>'))/100
        assert value==Decimal(str(a['pt']))
        suffix=tokenizer.encode(answer+tokenizer.eos_token,add_special_tokens=False)
        assert suffix==b['answer_token_ids']
        prefix=load_file(a['prompt_tensor_file'])['input_ids'].tolist()
        ids,labels=supervised_ids(prefix,suffix,16384)
        assert labels[:len(prefix)]==[-100]*len(prefix) and labels[len(prefix):]==suffix
        assert tokenizer.decode(suffix,skip_special_tokens=False)==answer+tokenizer.eos_token
        max_sequence=max(max_sequence,len(ids));supervised_tokens+=len(suffix)
        # Detect BPE changes at the prompt/answer boundary without touching pixels.
        prompt=read(Path(a['prompt_tensor_file']).with_suffix('.json'))['native_prompt']
        raw_prefix=tokenizer.encode(prompt,add_special_tokens=False)
        assert tokenizer.encode(prompt+answer+tokenizer.eos_token,add_special_tokens=False)==raw_prefix+suffix
        if a['episode_id'] not in seen:
            chosen.append(b);seen.add(a['episode_id'])
    # Real official processor comparison on one sample from every selected episode.
    for row in chosen:
        cached=collator([row]);prompt=read(Path(row['prompt_tensor_file']).with_suffix('.json'))['native_prompt']
        images=[]
        for p in row['image']:
            with Image.open(p) as im:images.append(im.convert('RGB'))
        online=dict(processor(text=[prompt+row['assistant_target']+tokenizer.eos_token],
                              images=images,return_tensors='pt'))
        for k,v in online.items():
            expected=v.to(cached[k].dtype)
            assert torch.equal(cached[k],expected),(row['id'],k)
        original,_=load_input(row)
        for k in ['pixel_values','image_grid_thw']:assert torch.equal(cached[k],original[k])
    # Mixed answer lengths: padded positions must never receive loss.
    mixed=[prepared[0],prepared[len(prepared)//2],prepared[-1]];batch=collator(mixed)
    assert (batch['labels'][batch['attention_mask']==0]==-100).all()
    for i,row in enumerate(mixed):
        kept=batch['labels'][i][batch['labels'][i]!=-100].tolist()
        assert kept==row['answer_token_ids']
    result=dict(status='passed',policy=POLICY,samples=2550,episodes=10,
        raw_pt_unchanged=True,all_input_records_and_pixel_references_unchanged=True,
        cot_supervision=False,teacher_forced_cot=False,remaining_action_supervision=False,
        loss_scope='progress answer (<progress>percentage%</progress>) and EOS only',
        prompt_image_and_padding_loss_mask=-100,real_processor_comparisons=len(chosen),
        maximum_sequence_tokens=max_sequence,supervised_tokens=supervised_tokens,
        target_scale='0-100 percent; unrounded original PT retained',
        cuda_used=False,model_forward_executed=False,training_started=False,verified=time.time())
    save(destination/'verification.json',result)
    return result


def build():
    from transformers import AutoTokenizer
    assert read(ROOT/'final-verification.json')['status']=='passed'
    assert not TARGET.exists(),'Preserve an existing supervision cache'
    staging=BASE/'progress-only.staging';assert not staging.exists(),'Inspect interrupted preparation'
    source=rows(BASE/'train.jsonl');before=sha(BASE/'train.jsonl')
    checkpoint=read(BASE/'cache-ready.json')['checkpoint']
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    assert tokenizer.eos_token and tokenizer.pad_token_id is not None
    prepared=[]
    for row in source:
        answer=answer_text(row['pt'])
        prepared.append(dict(row,assistant_target=answer,
            answer_token_ids=tokenizer.encode(answer+tokenizer.eos_token,add_special_tokens=False),
            supervision_policy=POLICY,cot_supervision=False,progress_gt=100*row['pt']))
    write_rows(staging/'train.jsonl',prepared)
    save(staging/'manifest.json',dict(policy=POLICY,user_instruction='ProcVLM的CoT答案只监督进度。',
        source_train=str(BASE/'train.jsonl'),source_train_sha256=before,
        selection_sha256=sha(ROOT/'selection.json'),checkpoint=checkpoint,
        train_sha256=sha(staging/'train.jsonl'),samples=len(prepared),episodes=len({r['episode_id'] for r in prepared}),
        native_prompt_unchanged=True,tokenizer_eos_token=tokenizer.eos_token,
        pixel_caches_reused=True,remaining_action_supervision=False,training_started=False))
    finalize()


def finalize():
    staging=BASE/'progress-only.staging'
    assert staging.is_dir() and not TARGET.exists()
    result=verify(staging)
    staging.rename(TARGET)
    save(ROOT/'proc-progress-only-verification.json',dict(result,cache=str(TARGET),
         manifest_sha256=sha(TARGET/'manifest.json')))
    print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group()
    action.add_argument('--verify',action='store_true');action.add_argument('--finalize',action='store_true');args=parser.parse_args()
    if args.verify:print(verify())
    elif args.finalize:finalize()
    else:build()
