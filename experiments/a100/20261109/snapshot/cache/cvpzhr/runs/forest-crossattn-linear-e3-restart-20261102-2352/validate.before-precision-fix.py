"""Final stateful evaluation: disjoint videos, ordered frames, no dummy supervision."""
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.sequence import EpisodeWindows, EpisodeWindowSampler, SequenceCollator
from mem_lite.runtime import load_checkpoint, to_device

RUN=Path(__file__).resolve().parent


def write(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


class ValidationCollator(SequenceCollator):
    def __call__(self,window):
        result=super().__call__(window)
        result['indices']=[i for row in window['indices'] for i in row]
        return result


def assert_coverage(records,dataset):
    records.sort(key=lambda row:row['index'])
    if [row['index'] for row in records]!=list(range(len(dataset))):
        raise ValueError('Validation records have missing or duplicated indices')
    for i,row in enumerate(records):
        episode,sample,_=dataset.samples[i]
        assert row['episode_id']==str(dataset.episodes[episode][0].resolve())
        assert row['frame_index']==sample['frame_index'] and row['target_pt']==float(sample['pt'])
        assert math.isfinite(row['predicted_pt'])


@torch.no_grad()
def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',local);torch.cuda.set_device(device);dist.init_process_group('nccl')
    try:
        settings=json.loads((RUN/'settings.json').read_text())
        output=Path(settings['output'])/'validation-03'
        if rank==0:output.mkdir(exist_ok=False)
        dist.barrier()
        model,processor,cfg=load_checkpoint(Path(settings['output'])/'last.pt',str(device))
        files=json.loads(Path(settings['validation_files']).read_text())
        data=M3Dataset(files,history_dim=cfg['model']['history_dim'])
        assert len(data)==32164 and data.history_source==cfg['history_source']
        sampler=EpisodeWindowSampler(data,4,8,rank,world,cfg['seed'])
        loader=DataLoader(EpisodeWindows(data),batch_size=None,sampler=sampler,
                          collate_fn=ValidationCollator(processor,data.history_dim,cfg['max_sequence_length'],data.history_capacity),
                          num_workers=4,persistent_workers=True,pin_memory=True,multiprocessing_context='spawn')
        torch.cuda.reset_peak_memory_stats(device)
        state=None;loss_sum=0.0;count=0;started=time.perf_counter()
        part=output/f'predictions-rank{rank}.jsonl'
        with part.open('w') as stream:
            for window_index,batch in enumerate(loader):
                if batch['reset']:state=None
                inputs=[to_device(frame,device) for frame in batch['inputs']]
                result=model(inputs,batch['target_pt'].to(device),state=state)
                state=result['state']
                loss_sum+=float(result['loss_sum']);count+=int(result['num_samples'])
                predictions=result['pt'].cpu().tolist()
                for index,pred in zip(batch['indices'],predictions,strict=True):
                    if index is None:continue
                    ep,sample,_=data.samples[index]
                    record=dict(index=index,episode_id=str(data.episodes[ep][0].resolve()),
                                frame_index=sample['frame_index'],target_pt=float(sample['pt']),predicted_pt=pred)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                if window_index%10==0 or window_index+1==len(loader):
                    stream.flush()
                    write(output/f'status-rank{rank}.json',dict(rank=rank,samples=count,windows_done=window_index+1,
                        windows_total=len(loader),seconds=time.perf_counter()-started,updated_at=time.time()))
        write(output/f'metrics-rank{rank}.json',dict(num_samples=count,loss_sum=loss_sum,
              seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        dist.barrier()
        if rank==0:
            records=[];parts=[]
            for i in range(world):
                records.extend(json.loads(line) for line in (output/f'predictions-rank{i}.jsonl').read_text().splitlines())
                parts.append(json.loads((output/f'metrics-rank{i}.json').read_text()))
            assert_coverage(records,data)
            errors=[r['predicted_pt']-r['target_pt'] for r in records]
            metrics=dict(mode='stateful_cached_history',num_samples=len(records),
                         loss=math.fsum(p['loss_sum'] for p in parts)/len(records),
                         mae_pp=100*math.fsum(abs(e) for e in errors)/len(errors),
                         rmse_pp=100*math.sqrt(math.fsum(e*e for e in errors)/len(errors)),
                         bias_pp=100*math.fsum(errors)/len(errors),
                         seconds=time.perf_counter()-started,
                         peak_allocated_gib_per_gpu=[p['peak_allocated_gib'] for p in parts],
                         batch_video_per_gpu=4,sequence_window=8,coverage_verified=True,
                         note='Frames are ordered within each video; both memory states reset at video boundaries. Batch evaluation is not online single-frame latency.')
            with (output/'predictions.jsonl').open('w') as stream:
                for record in records:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            baseline=Path('/cache/cvpzhr/outputs/forest-proprimem-e3-20261031-0009/validation-03/predictions.jsonl')
            prior=[json.loads(line) for line in baseline.read_text().splitlines()]
            key=lambda r:(r['episode_id'],r['frame_index'])
            prior_map={key(r):r['target_pt'] for r in prior}
            assert len(prior)==len(prior_map)==len(records)
            assert all(key(r) in prior_map and r['target_pt']==prior_map[key(r)] for r in records)
            metrics['same_samples_and_pt_as_forest']=True
            write(output/'metrics.json',metrics)
            print(json.dumps(metrics,ensure_ascii=False),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
