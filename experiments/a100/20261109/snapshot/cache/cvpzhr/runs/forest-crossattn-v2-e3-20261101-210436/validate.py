"""Final cached evaluation; preserve the prior four-shard batch composition."""
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device

RUN=Path(__file__).resolve().parent
DATA=Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030')

def validation_batches(count,batch_size,rank,world):
    # Prior Forest used sample indices shard::4 and batch=4 on each shard.
    batches=[]
    for shard in range(4):
        indices=list(range(shard,count,4))
        batches.extend(indices[i:i+batch_size] for i in range(0,len(indices),batch_size))
    return batches[rank::world]

@torch.no_grad()
def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',local);torch.cuda.set_device(device);dist.init_process_group('nccl')
    try:
        config=json.loads((RUN/'config.json').read_text())
        output=Path(config['output_dir'])/'validation-03';output.mkdir(parents=True,exist_ok=True)
        model,processor,saved=load_checkpoint(Path(config['output_dir'])/'last.pt',device)
        dataset=M3Dataset(json.loads((DATA/'validation.sample_files.json').read_text()),saved['model']['history_dim'])
        assert dataset.history_source==saved['history_source'] and dataset.history_capacity==saved['history_capacity']
        batches=validation_batches(len(dataset),4,rank,world)
        loader=DataLoader(dataset,batch_sampler=batches,
            collate_fn=Stage3Collator(processor,dataset.history_dim,saved['max_sequence_length'],dataset.history_capacity),
            num_workers=4,pin_memory=True,persistent_workers=True,multiprocessing_context='spawn')
        totals=torch.zeros(5,dtype=torch.float64,device=device);seen=0;started=perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        with (output/f'predictions-rank{rank}.jsonl').open('w') as stream:
            for indices,batch in zip(batches,loader,strict=True):
                target=batch['target_pt'].to(device);result=model(to_device(batch['inputs'],device),target)
                prediction=result['pt'];error=(prediction-target).double()
                assert torch.isfinite(prediction).all() and torch.isfinite(result['loss'])
                totals+=torch.stack((result['loss_sum'].double(),torch.tensor(len(target),device=device),
                                    error.abs().sum(),error.square().sum(),error.sum()))
                for index,pred,pt in zip(indices,prediction.tolist(),target.tolist(),strict=True):
                    episode,sample,_=dataset.samples[index]
                    stream.write(json.dumps({'index':index,'episode_id':str(dataset.episodes[episode][0]),
                        'frame_index':sample['frame_index'],'target_pt':pt,'predicted_pt':pred},ensure_ascii=False)+'\n')
                seen+=len(target)
                if seen%256<len(target):
                    stream.flush();(output/f'status-rank{rank}.json').write_text(json.dumps({
                        'samples':seen,'total':sum(map(len,batches)),'seconds':perf_counter()-started}))
        torch.cuda.synchronize(device)
        elapsed=torch.tensor(perf_counter()-started,device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        dist.all_reduce(totals)
        peak=torch.tensor(torch.cuda.max_memory_allocated(device)/2**30,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if rank==0:
            count=int(totals[1]);assert count==len(dataset)==32164
            records=[]
            for source in sorted(output.glob('predictions-rank*.jsonl')):
                records.extend(json.loads(line) for line in source.read_text().splitlines())
            records.sort(key=lambda row:row['index']);assert [r['index'] for r in records]==list(range(count))
            with (output/'predictions.jsonl').open('w') as stream:
                for row in records:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            metrics={'num_samples':count,'loss':float(totals[0]/count),'mae_pp':float(100*totals[2]/count),
                     'rmse_pp':float(100*(totals[3]/count).sqrt()),'bias_pp':float(100*totals[4]/count),
                     'seconds':float(elapsed),'samples_per_second':count/float(elapsed),'peak_allocated_gib':float(peak),
                     'world_size':world,'batch_size':4,'batch_layout':'same virtual four-shard groups as prior Forest',
                     'input_mode':'cached causal history; current three slots, missing view filled black',
                     'latency_note':'Batch throughput; not single-frame online latency'}
            (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics),flush=True)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
