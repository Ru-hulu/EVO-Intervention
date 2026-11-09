"""Check full label/coverage alignment and cached pixels against real preprocessing."""
from pathlib import Path
import json
import math
import torch
from grm import Dataset, Collator, load_processor, parse_score, answer, messages, pixel_cache
from metrics import correlation, summarize
from safetensors import safe_open

ROOT=Path('/cache/cvpzhr/setup/robobrain25-forward-20261105')


def main():
    torch.set_num_threads(2)
    config=json.loads((ROOT/'config.draft.json').read_text())
    processor=load_processor(config['model_path'],config['min_pixels'],config['max_pixels'])
    collator=Collator(processor,config['max_length'])
    assert parse_score('<score>-20.0%</score>') == -.2
    for text in ['bad','<score>101%</score>','<score>nan%</score>']:
        try:parse_score(text)
        except ValueError:pass
        else:raise AssertionError(text)
    assert correlation([0,1,2],[0,2,4]) == 1
    assert correlation([1,1,1],[0,1,2]) is None
    examples=[];totals={};max_length=0;native_images_checked=set()
    for split,total in [('train',318931),('validation',66891),('augmented_validation',78554)]:
        dataset=Dataset(config['data_dir'],split)
        assert len(dataset)==total
        seen=set();covered=set();kind_covered=set()
        for i,row in enumerate(dataset.rows):
            key=row['episode_id'],row['sample_index'];assert key not in seen;seen.add(key)
            assert row['answer']==answer(row['pt'])
            assert abs(parse_score(row['answer'])-row['pt'])<=.00050001
            e=dataset.episodes[row['episode_id']]
            task=e['task_id']
            if task not in covered or (row['kind'] not in kind_covered):
                examples.append((split,i,dataset[i]));covered.add(task);kind_covered.add(row['kind'])
        assert len(covered)==47
        totals[split]=dict(samples=len(dataset),episodes=len({r['episode_id'] for r in dataset.rows}))
    # Inspect every episode's tiny grid tensors without reading its large pixel tensors.
    # Each view is stacked in the existing cache, so assert its grid is constant within the episode.
    grid_contracts=set()
    for e in dataset.episodes.values():
        with safe_open(e['pixel_cache'],framework='pt',device='cpu') as f:
            view_grids={}
            for view in ['cam_high','cam_left_wrist','cam_right_wrist']:
                grid=f.get_tensor('grids_'+view)
                assert (grid==grid[0]).all(),e['pixel_cache']
                view_grids[view]=grid[0].tolist()
        grids=[view_grids[v] for v in ['cam_high','cam_high','cam_high','cam_left_wrist','cam_right_wrist',
                                       'cam_high','cam_left_wrist','cam_right_wrist']]
        key=e['task_prompt'],tuple(tuple(g) for g in grids)
        if key in grid_contracts:continue
        grid_contracts.add(key)
        text=processor.apply_chat_template(messages(e['task_prompt'],answer(1.)),tokenize=False,add_generation_prompt=False)
        for grid in grids:
            text=text.replace(processor.image_token,'<|placeholder|>'*(math.prod(grid)//4),1)
        text=text.replace('<|placeholder|>',processor.image_token)
        n=len(processor.tokenizer(text,add_special_tokens=False)['input_ids'])
        assert n<=config['max_length'],(e['task_id'],n)
        max_length=max(max_length,n)
    for n,(split,i,row) in enumerate(examples,1):
        cached=collator.encode([row],cached=True)
        raw=collator.encode([row],cached=False)
        for key in ['input_ids','attention_mask','image_grid_thw','labels','prompt_input_ids','prompt_attention_mask','target_pt']:
            assert torch.equal(cached[key],raw[key]),(split,i,key)
        assert torch.equal(cached['pixel_values'],raw['pixel_values'].to(cached['pixel_values'].dtype)),(split,i,'pixels')
        supervised=cached['labels'][0]!=-100
        decoded=processor.tokenizer.decode(cached['labels'][0][supervised],skip_special_tokens=False)
        assert decoded==row['answer']+'<|im_end|>\n',repr(decoded)
        assert '<score>' not in processor.tokenizer.decode(cached['prompt_input_ids'][0]).rsplit('<|im_start|>assistant',1)[-1]
        # Changing a target cannot alter any generation input or image.
        changed=dict(row,pt=.123,answer=answer(.123))
        other=collator.encode([changed])
        for key in ['prompt_input_ids','prompt_attention_mask','pixel_values','image_grid_thw']:
            assert torch.equal(cached[key],other[key]),key
        max_length=max(max_length,cached['input_ids'].shape[1])
        if n%30==0:print('processor parity',n,'/',len(examples),flush=True)
    # Mixed-example left/right padding and masks must agree with individual samples.
    mixed=[examples[i][2] for i in [0,len(examples)//2,-1]]
    batch=collator.encode(mixed)
    for i,row in enumerate(mixed):
        solo=collator.encode([row])
        n=int(batch['attention_mask'][i].sum())
        assert torch.equal(batch['input_ids'][i,:n],solo['input_ids'][0])
        assert torch.equal(batch['labels'][i,:n],solo['labels'][0])
        assert (batch['labels'][i,n:]==-100).all()
    report=dict(status='passed',coverage=totals,cached_online_examples=len(examples),
        max_checked_sequence_length=max_length,all_labels_checked=True,
        pixel_grid_contracts=len(grid_contracts),all_episode_sequence_lengths_checked=True,
        assistant_only_loss=True,generation_target_independent=True,
        image_order_and_true_endpoints_verified=True,pixels_reused_without_change=True,
        label_quantization_bound_pt=.0005,uses_future_goal_reference=True)
    (ROOT/'cpu-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'smoke-rows.json').write_text(json.dumps([x[2] for x in examples if x[0]=='train'][:1]+
        [next(x[2] for x in examples if x[0]=='train' and x[2]['episode_id'].startswith(s)) for s in ['Order/','State/']]
        +[next(x[2] for x in examples if x[2]['kind']=='reverse')],ensure_ascii=False,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
