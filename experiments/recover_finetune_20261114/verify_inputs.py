"""CPU audit: preserve every cached input, mask prompts/padding, never truncate."""
import argparse
from functools import lru_cache
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from .common import *
from .data import CachedCollator,grm_answer


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args()
    cfg=read(ROOT/args.model/'config.json');data=rows(cfg['train_data'])
    tokenizer=AutoTokenizer.from_pretrained(Path(cfg['model_path'])/('processor' if args.model=='robometer-4b' else ''),local_files_only=True)
    if args.model=='robometer-4b':
        for token in ['<|split_token|>','<|reward_token|>','<|pref_token|>','<|sim_token|>','<|prog_token|>']:
            if token not in tokenizer.get_vocab():tokenizer.add_special_tokens({'additional_special_tokens':[token]})
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    @lru_cache(maxsize=16)
    def prompt(path):return load_file(path)['input_ids'].tolist()
    longest=0
    for r in data:
        ids=prompt(r['prompt_tensor_file']);answer=[]
        if args.model!='robometer-4b':
            text=r['assistant_target'] if args.model=='procvlm-2b' else grm_answer(r)
            suffix=text+tokenizer.eos_token;answer=tokenizer.encode(suffix,add_special_tokens=False)
            native=read(Path(r['prompt_tensor_file']).with_suffix('.json'))['native_prompt']
            assert tokenizer.encode(native+suffix,add_special_tokens=False)==tokenizer.encode(native,add_special_tokens=False)+answer
        longest=max(longest,len(ids)+len(answer))
    assert longest<=cfg['max_length']
    chosen=[data[0],data[len(data)//2],data[-1]]
    batch=CachedCollator(tokenizer,args.model,cfg['max_length'])(chosen)
    from experiments.recover_six_caches_20261114.loader import load_input
    inputs=[load_input(r)[0] for r in chosen]
    for key in ('pixel_values','image_grid_thw'):assert torch.equal(batch[key],torch.cat([b[key] for b in inputs]))
    for i,r in enumerate(chosen):
        prefix=inputs[i]['input_ids'][0]
        assert torch.equal(batch['input_ids'][i,:len(prefix)],prefix)
        if args.model=='robometer-4b':assert abs(batch['labels'][i].item()-r['pt'])<1e-7
        else:
            assert batch['labels'][i,:len(prefix)].eq(-100).all()
            expected=(r['assistant_target'] if args.model=='procvlm-2b' else grm_answer(r))+tokenizer.eos_token
            assert batch['labels'][i][batch['labels'][i]!=-100].tolist()==tokenizer.encode(expected,add_special_tokens=False)
            assert batch['labels'][i][batch['attention_mask'][i]==0].eq(-100).all()
    save(ROOT/args.model/'input-verification.json',dict(status='passed',samples=len(data),
        maximum_sequence=longest,native_prompt_suffix_boundary_all_samples=True,
        mixed_batch_inputs_unchanged=True,labels_masked=True,training_started_by_verifier=False))
    print(args.model,longest,'passed',flush=True)


if __name__=='__main__':main()
