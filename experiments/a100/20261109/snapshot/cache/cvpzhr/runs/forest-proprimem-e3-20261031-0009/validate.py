"""最终检查点的四卡缓存验证；按索引分片，不补齐或丢弃验证样本。"""
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
DATA=Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030')


@torch.no_grad()
def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',local);torch.cuda.set_device(device);dist.init_process_group('nccl')
    config=json.loads((RUN/'config.json').read_text());output=Path(config['output_dir'])/'validation-03'
    output.mkdir(parents=True,exist_ok=True)
    model,processor,config=load_checkpoint(str(Path(config['output_dir'])/'last.pt'),device)
    files=json.loads((DATA/'validation.sample_files.json').read_text())
    dataset=M3Dataset(files,config['model']['history_dim'])
    assert dataset.history_source==config['history_source'] and dataset.history_capacity==config['history_capacity']
    indices=list(range(rank,len(dataset),world))
    loader=DataLoader(Subset(dataset,indices),batch_size=min(config['batch_size'],16),shuffle=False,
                      collate_fn=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity),
                      num_workers=config['num_workers'],pin_memory=True,persistent_workers=True,multiprocessing_context='spawn')
    totals=torch.zeros(5,dtype=torch.float64,device=device);seen=0;started=perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with (output/f'predictions-rank{rank}.jsonl').open('w') as stream:
        for batch in loader:
            target=batch['target_pt'].to(device);result=model(to_device(batch['inputs'],device),target)
            prediction=result['pt'];error=(prediction-target).double()
            if not torch.isfinite(prediction).all() or not torch.isfinite(result['loss']):raise ValueError('Nonfinite validation output')
            totals+=torch.stack((result['loss_sum'].double(),torch.tensor(len(target),device=device),
                                 error.abs().sum(),error.square().sum(),error.sum()))
            for offset,(pred,pt) in enumerate(zip(prediction.tolist(),target.tolist())):
                index=indices[seen+offset];episode,sample,_=dataset.samples[index]
                stream.write(json.dumps(dict(index=index,episode_id=str(dataset.episodes[episode][0]),
                    frame_index=sample['frame_index'],target_pt=pt,predicted_pt=pred),ensure_ascii=False)+'\n')
            seen+=len(target)
            if seen%256<len(target):
                stream.flush();(output/f'status-rank{rank}.json').write_text(json.dumps(dict(samples=seen,total=len(indices),seconds=perf_counter()-started)))
    torch.cuda.synchronize(device)
    elapsed=torch.tensor(perf_counter()-started,device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    dist.all_reduce(totals)
    peak=torch.tensor(torch.cuda.max_memory_allocated(device)/2**30,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
    if rank==0:
        count=int(totals[1]);assert count==len(dataset)
        metrics=dict(num_samples=count,loss=float(totals[0]/count),mae_pp=float(100*totals[2]/count),
                     rmse_pp=float(100*(totals[3]/count).sqrt()),bias_pp=float(100*totals[4]/count),
                     seconds=float(elapsed),samples_per_second=count/float(elapsed),peak_allocated_gib=float(peak),
                     input_mode='cached causal history; current three slots, missing view filled black',
                     latency_note='Batch throughput; not single-frame online latency')
        records=[]
        for source in sorted(output.glob('predictions-rank*.jsonl')):records.extend(json.loads(line) for line in source.read_text().splitlines())
        records.sort(key=lambda row:row['index']);assert [r['index'] for r in records]==list(range(len(dataset)))
        with (output/'predictions.jsonl').open('w') as stream:
            for row in records:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
