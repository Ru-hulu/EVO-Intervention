"""Measure real generated-output throughput for independent rollout streams."""

import argparse
import json
from pathlib import Path
import time

import torch

from qwen_training.closed_loop_rollout import save_json, sha256
from qwen_training.dataset import VLMProgressDataset
from qwen_training.evaluate_checkpoint import load_model
from qwen_training.evaluation_metrics import parse_response
from qwen_training.prompts import build_messages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=load_model(checkpoint,device)
    del checkpoint
    dataset=VLMProgressDataset(args.source/'rollout_index.jsonl')
    prior=[json.loads(line) for line in (args.source/'predictions.jsonl').read_text().splitlines()]
    indices=[0,25,70,105,177,195,225,234]
    started=time.monotonic()
    samples=[dataset[i] for i in indices]
    decode_seconds=time.monotonic()-started
    states=[prior[i]['input_state'] for i in indices]
    output=dict(checkpoint_sha256=sha256(args.checkpoint),source=str(args.source),indices=indices,
                input_decode_seconds=decode_seconds,threads=2,benchmarks=[])

    def run(items):
        torch.cuda.synchronize()
        begin=time.monotonic()
        images=torch.stack([samples[i]['images'] for i in items]).to(device)
        masks=torch.stack([samples[i]['frame_mask'] for i in items])
        messages=build_messages([states[i] for i in items])
        torch.cuda.synchronize()
        prepared=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            visual=model.encode_history(images,masks)
            torch.cuda.synchronize()
            encoded=time.monotonic()
            result=model.generate(messages,**visual,max_new_tokens=192)
            progress=result['progress'].float().cpu().tolist()
            torch.cuda.synchronize()
        ended=time.monotonic()
        return dict(batch_size=len(items),prepare_seconds=prepared-begin,vision_seconds=encoded-prepared,
                    generation_seconds=ended-encoded,total_seconds=ended-begin,
                    samples_per_second=len(items)/(ended-begin),responses=result['responses'],progress=progress,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)

    run([0])  # Warm kernels using an actual input; excluded from throughput.
    baselines=[]
    for i in range(len(samples)):
        torch.cuda.reset_peak_memory_stats()
        result=run([i]);baselines.append(result)
        output['benchmarks'].append(dict(kind='single',indices=[indices[i]],**result))
        save_json(args.output,output)
        print(json.dumps({k:v for k,v in result.items() if k not in ['responses','progress']}),flush=True)
    for size in [2,5,8]:
        torch.cuda.reset_peak_memory_stats()
        result=run(list(range(size)))
        result['reference_text_exact']=sum(result['responses'][i]==baselines[i]['responses'][0] for i in range(size))
        result['reference_state_exact']=sum(parse_response(result['responses'][i])==parse_response(baselines[i]['responses'][0]) for i in range(size))
        result['reference_progress_max_difference_pp']=max(abs(result['progress'][i]-baselines[i]['progress'][0])*100 for i in range(size))
        output['benchmarks'].append(dict(kind='batched',indices=indices[:size],**result))
        save_json(args.output,output)
        print(json.dumps({k:v for k,v in result.items() if k not in ['responses','progress']}),flush=True)
    output['status']='complete'
    save_json(args.output,output)


if __name__=='__main__':
    main()
