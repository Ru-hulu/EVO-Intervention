"""Check draft decoding against actual ordinary greedy outputs and wrong drafts."""
import argparse
import json
from pathlib import Path
import time
import torch

from qwen_training.closed_loop_rollout import save_json
from qwen_training.dataset import VLMProgressDataset
from qwen_training.evaluate_checkpoint import load_model
from qwen_training.evaluation_metrics import parse_response
from qwen_training.prompts import build_messages
from qwen_training.rollout_inference import generate_verified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    ref=json.loads((args.directory/'batches.json').read_text())
    source=Path(ref['source'])
    prior=[json.loads(l) for l in (source/'predictions.jsonl').read_text().splitlines()]
    indices=ref['indices'];ds=VLMProgressDataset(source/'rollout_index.jsonl')
    torch.set_num_threads(2);torch.manual_seed(0)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=load_model(ck,torch.device('cuda:0'));del ck
    samples=[ds[i] for i in indices]
    results=[]
    for size,mode in [(5,'previous_frame'),(8,'previous_frame'),(5,'wrong_draft'),(5,'repeat_verified_output'),(8,'repeat_verified_output')]:
        images=torch.stack([s['images'] for s in samples[:size]]).cuda()
        masks=torch.stack([s['frame_mask'] for s in samples[:size]])
        messages=build_messages([prior[i]['input_state'] for i in indices[:size]])
        drafts=[prior[i-1]['response'] if i else None for i in indices[:size]]
        reference=next(r for r in ref['benchmarks'] if r['kind']=='batched' and r['batch_size']==size)
        if mode=='wrong_draft':drafts=['错误草稿，必须回退到模型生成。']*size
        if mode=='repeat_verified_output':drafts=reference['responses']
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            visual=model.encode_history(images,masks)
            torch.cuda.synchronize();begin=time.monotonic()
            out=generate_verified(model,messages,**visual,drafts=drafts,minimum_margin=1.0)
            torch.cuda.synchronize();elapsed=time.monotonic()-begin
        values=out['progress'].cpu().tolist()
        result=dict(batch_size=size,mode=mode,generation_seconds=elapsed,
                    draft_accepted=out['draft_accepted'],minimum_margins=out['draft_minimum_margin'],
                    reference_text_exact=sum(a==b for a,b in zip(out['responses'],reference['responses'])),
                    reference_state_exact=sum(parse_response(a)==parse_response(b) for a,b in zip(out['responses'],reference['responses'])),
                    progress_difference_pp=max(abs(a-b)*100 for a,b in zip(values,reference['progress'])),
                    responses=out['responses'],progress=values)
        results.append(result)
        save_json(args.directory/'draft.json',dict(status='running',results=results))
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if result['reference_text_exact']!=size:
            raise ValueError('Draft decoder does not match ordinary greedy reference text')
        if mode=='wrong_draft' and any(out['draft_accepted']):
            raise ValueError('Invalid draft was accepted')
    save_json(args.directory/'draft.json',dict(status='complete',results=results))


if __name__=='__main__':
    main()
