"""Wrap the production bounded benchmark with per-rank input wait timing only."""
import json,os,sys,time
from pathlib import Path
from qwen_training import benchmark_distributed as bench

original=bench.training.phase_batches
phase_runs={'train':0,'validation':0}
output=Path(sys.argv[sys.argv.index('--output-dir')+1])
def timed_phase_batches(loader):
 phase='train' if isinstance(loader.batch_sampler,bench.BoundedSampler) else 'validation'
 phase_runs[phase]+=1
 epoch=phase_runs[phase]
 iterator=iter(original(loader));rows=[];index=0
 try:
  while True:
   started=time.perf_counter()
   try:batch=next(iterator)
   except StopIteration:break
   ready=time.perf_counter();index+=1
   yield batch
   returned=time.perf_counter()
   rows.append({'rank':int(os.environ['RANK']),'phase':phase,'epoch':epoch,'phase_batch':index,'input_wait_seconds':ready-started,'consumer_seconds':returned-ready})
 finally:
  iterator.close()
  output.mkdir(parents=True,exist_ok=True)
  with (output/('input-timing-rank-'+os.environ['RANK']+'.jsonl')).open('a') as stream:
   for row in rows:stream.write(json.dumps(row)+'\n')

if __name__=='__main__':
 bench.training.phase_batches=timed_phase_batches
 bench.main()
