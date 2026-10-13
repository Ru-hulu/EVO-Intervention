"""Prepare and run all held-out episodes at 30 Hz on independent GPU workers."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from qwen_training.closed_loop_rollout import feedback_state, prepare, save_json, sha256
from qwen_training.evaluation_metrics import parse_response
from qwen_training.prompts import build_messages, describe_target


def prepare_one(job):
    spec,config,checkpoint_meta,output=job
    args=SimpleNamespace(dataset_root=Path(spec['dataset_root']),split=Path(spec['split']),
                         video_id=spec['video_id'],task_index=spec['task_index'],episode=spec['episode'],step_frames=1)
    rows,truth,manifest=prepare(args,config)
    directory=Path(output)/spec['relative_dir']
    directory.mkdir(parents=True,exist_ok=False)
    manifest.update(checkpoint_meta)
    manifest.update(generation='greedy_with_previous_prediction_draft_verification',max_new_tokens=192,
                    minimum_draft_margin=1.0,source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                    ['validation_rollout_suite.py','rollout_inference.py','closed_loop_rollout.py','model.py','prompts.py','dataset.py']})
    save_json(directory/'manifest.json',manifest)
    save_json(directory/'ground_truth.json',truth)
    (directory/'rollout_index.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    return dict(spec,samples=len(rows),frame_start=manifest['frame_start'],frame_end=manifest['frame_end'])


def setup(args):
    import torch
    root=args.directory
    if (root/'suite.json').exists():
        raise FileExistsError('Suite already prepared; refusing to replace it')
    digest=sha256(args.checkpoint)
    if digest!=args.expected_checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=ck['config']
    checkpoint_meta=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=digest,
                         checkpoint_epoch=ck['epoch'],checkpoint_step=ck['step'],model_path=config['model_path'])
    del ck
    split=json.loads(args.split.read_text())
    if set(map(tuple,split['train']))&set(map(tuple,split['validation'])):
        raise ValueError('Video split overlaps')
    roots={Path(path).parent.parent.name:Path(path).parent.parent for path in config['index_files']}
    jobs=[]
    for dataset,video in split['validation']:
        dataset_root=roots[dataset]
        path=dataset_root/'labels'/(video+'.JSON')
        if not path.exists():path=path.with_suffix('.json')
        annotation=json.loads(path.read_text())
        for task_index,task in enumerate(annotation['tasks']):
            for episode in task['episodes']:
                spec=dict(dataset=dataset,dataset_root=str(dataset_root),video_id=video,
                          task_index=task_index,episode=episode['episode'],split=str(args.split),
                          relative_dir=f'episodes/{dataset}/{video}/task-{task_index:02d}-episode-{episode["episode"]:02d}')
                jobs.append((spec,config,checkpoint_meta,str(root)))
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        episodes=list(pool.map(prepare_one,jobs))
    assignments=[[] for _ in args.gpus]
    capacity=(len(episodes)+len(assignments)-1)//len(assignments)
    for episode in sorted(episodes,key=lambda x:x['samples'],reverse=True):
        rank=min((i for i,a in enumerate(assignments) if len(a)<capacity),
                 key=lambda i:(sum(e['samples'] for e in assignments[i]),len(assignments[i])))
        assignments[rank].append(episode)
    suite=dict(status='prepared',created_unix=time.time(),checkpoint=checkpoint_meta,
               split=str(args.split),split_sha256=sha256(args.split),validation_videos=split['validation'],
               episodes=episodes,assignments=assignments,gpus=args.gpus,samples=sum(e['samples'] for e in episodes),
               prediction_hz=30,history_offsets_seconds=[-5,-4,-3,-2,-1,0],
               protocol='Each episode independently blank-padded; first state annotated, all later states predicted.',
               coverage='All annotated episode frames; unannotated lead-in and trailing footage excluded.',
               expected_reports=len(episodes),expected_figures=2*len(episodes))
    save_json(root/'suite.json',suite)
    print(json.dumps(dict(status='prepared',episodes=len(episodes),samples=suite['samples'],
                         rank_samples=[sum(e['samples'] for e in a) for a in assignments])),flush=True)


class EpisodeFrames:
    """Decode each source frame once, preserving native RGB pixels and PTS alignment."""
    def __init__(self,rows):
        import av
        import numpy as np
        import torch
        self.start=rows[0]['frame_index'];self.end=rows[-1]['frame_index']
        paths=rows[0]['video_paths']
        with av.open(paths['top']) as video:
            stream=video.streams.video[0]
            height,width=stream.height,stream.width
        self.array=np.empty((self.end-self.start+1,3,3,height,width),dtype=np.uint8)

        def decode(view,name):
            seen=np.zeros(self.end-self.start+1,dtype=bool)
            with av.open(paths[name]) as video:
                stream=video.streams.video[0];stream.codec_context.thread_count=1
                if stream.average_rate!=30 or (stream.height,stream.width)!=(height,width):
                    raise ValueError('Source view FPS or dimensions differ')
                origin=stream.start_time or 0
                video.seek(origin+int(self.start/stream.average_rate/stream.time_base),stream=stream,backward=True)
                frames=video.decode(stream)
                try:
                    for frame in frames:
                        index=round((frame.pts-origin)*stream.time_base*stream.average_rate)
                        if self.start<=index<=self.end:
                            if seen[index-self.start]:raise ValueError('Duplicate source frame timestamp')
                            self.array[index-self.start,view]=frame.to_ndarray(format='rgb24').transpose(2,0,1)
                            seen[index-self.start]=True
                        if index>=self.end:break
                finally:
                    frames.close();stream.codec_context.close(strict=False)
            if not seen.all():raise ValueError('Missing decoded source frames')

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda x:decode(*x),enumerate(('top','left','right'))))
        self.tensor=torch.from_numpy(self.array)
        self.blank=torch.zeros_like(self.tensor[0])

    def window(self,indices):
        import torch
        return torch.stack([self.blank if i is None else self.tensor[i-self.start] for i in indices],dim=1)


def run_worker(args):
    import torch
    from qwen_training.dataset import VLMProgressDataset
    from qwen_training.evaluate_checkpoint import load_model
    from qwen_training.rollout_inference import generate_verified
    root=args.directory
    suite=json.loads((root/'suite.json').read_text())
    rank=args.rank
    status_path=root/f'status-rank{rank:02d}.json'
    result_path=root/f'complete-rank{rank:02d}.json'
    torch.set_num_threads(2);torch.manual_seed(0)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    contexts=[]
    try:
        for spec in suite['assignments'][rank]:
            directory=root/spec['relative_dir']
            rows=[json.loads(line) for line in (directory/'rollout_index.jsonl').read_text().splitlines()]
            if (directory/'predictions.jsonl').exists():raise FileExistsError('Refusing duplicate episode execution')
            save_json(status_path,dict(status='decoding',rank=rank,episode=spec['relative_dir']))
            cache=EpisodeFrames(rows)
            baseline=VLMProgressDataset(directory/'rollout_index.jsonl')
            checked=sorted({0,len(rows)//2,len(rows)-1})
            for index in checked:
                if not torch.equal(cache.window(rows[index]['frame_indices']),baseline[index]['images']):
                    raise ValueError('Cached video differs from original dataset decoding')
            save_json(directory/'decode_audit.json',dict(status='passed',checked_steps=checked,
                      bytes=cache.array.nbytes,frames=len(rows),views=3))
            contexts.append(dict(spec=spec,directory=directory,rows=rows,cache=cache,step=0,
                                 state=rows[0]['input_state'],previous_response=None,
                                 output=(directory/'predictions.jsonl').open('x'),accepted=0))
        save_json(status_path,dict(status='loading_model',rank=rank))
        if sha256(suite['checkpoint']['checkpoint'])!=suite['checkpoint']['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after preparation')
        ck=torch.load(suite['checkpoint']['checkpoint'],map_location='cpu',weights_only=True)
        model=load_model(ck,device);del ck
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();steps=0;total=sum(len(c['rows']) for c in contexts)
        active=list(contexts)

        def prefetch(items):
            images=torch.stack([c['cache'].window(c['rows'][i]['frame_indices']) for c,i in items]).pin_memory()
            masks=torch.tensor([c['rows'][i]['frame_mask'] for c,i in items],dtype=torch.bool)
            return images,masks

        with ThreadPoolExecutor(max_workers=1) as prefetcher,torch.inference_mode():
            future=prefetcher.submit(prefetch,[(c,0) for c in active])
            while active:
                tick=time.monotonic()
                images,masks=future.result()
                following=[(c,c['step']+1) for c in active if c['step']+1<len(c['rows'])]
                future=prefetcher.submit(prefetch,following) if following else None
                messages=build_messages([c['state'] for c in active],state_drop_probability=0.0)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    visual=model.encode_history(images.to(device,non_blocking=True),masks)
                    prediction=generate_verified(model,messages,**visual,
                                 drafts=[c['previous_response'] for c in active],minimum_margin=1.0,max_new_tokens=192)
                    values=prediction['progress'].float().cpu().tolist()
                duration=time.monotonic()-tick
                for i,context in enumerate(active):
                    meta=context['rows'][context['step']]
                    response=prediction['responses'][i]
                    value=values[i]
                    if not 0<=value<=1:raise ValueError('Nonfinite or invalid progress')
                    row=dict(meta,input_state=context['state'],prompt=messages[i],response=response,
                             predicted_progress=value,parsed=parse_response(response),target_response=describe_target(meta['target']),
                             mode='closed_loop',input_state_source='initial_ground_truth' if context['step']==0 else 'previous_prediction',
                             seconds=duration,amortized_seconds=duration/len(active),batch_size=len(active),gpu_rank=rank,
                             draft_source='previous_prediction' if context['previous_response'] is not None else None,
                             draft_accepted=prediction['draft_accepted'][i],draft_minimum_margin=prediction['draft_minimum_margin'][i])
                    context['output'].write(json.dumps(row,ensure_ascii=False)+'\n');context['output'].flush()
                    context['state']=feedback_state(response)
                    context['previous_response']=response
                    context['accepted']+=int(prediction['draft_accepted'][i])
                    context['step']+=1
                    if context['step']==len(context['rows']):
                        context['output'].close()
                        directory=context['directory']
                        completion=dict(status='complete',samples=context['step'],elapsed_seconds=time.monotonic()-started,
                                        predictions_sha256=sha256(directory/'predictions.jsonl'),ground_truth_sha256=sha256(directory/'ground_truth.json'),
                                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,finished_unix=time.time(),rank=rank,
                                        draft_accepted=context['accepted'],draft_fallback=context['step']-context['accepted'])
                        save_json(directory/'result.json',completion);save_json(directory/'status.json',completion)
                        context['cache']=None
                steps+=1
                processed=sum(c['step'] for c in contexts)
                if steps%10==0 or processed==total:
                    status=dict(status='running',rank=rank,completed=processed,samples=total,
                                completed_episodes=sum(c['step']==len(c['rows']) for c in contexts),
                                active_episodes=len(following),elapsed_seconds=time.monotonic()-started,
                                samples_per_second=processed/(time.monotonic()-started),
                                draft_accepted=sum(c['accepted'] for c in contexts),
                                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    save_json(status_path,status);print(json.dumps(status),flush=True)
                active=[c for c in active if c['step']<len(c['rows'])]
        final=dict(status='complete',rank=rank,samples=total,episodes=len(contexts),elapsed_seconds=time.monotonic()-started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,finished_unix=time.time())
        save_json(result_path,final);save_json(status_path,final)
    except Exception as error:
        for c in contexts:
            if not c['output'].closed:c['output'].close()
        failure=dict(status='failed',rank=rank,error=repr(error),finished_unix=time.time())
        save_json(result_path,failure);save_json(status_path,failure)
        raise


def launch(args):
    root=args.directory
    lock=(root/'launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    suite=json.loads((root/'suite.json').read_text())
    if (root/'launch.json').exists():raise FileExistsError('Suite launch already exists')
    # All selected GPUs must still be idle; do not stop unrelated workloads.
    for gpu in suite['gpus']:
        memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory>=512:raise RuntimeError(f'GPU {gpu} is busy')
    processes=[]
    for rank,gpu in enumerate(suite['gpus']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
        command=[sys.executable,'-u','-m','qwen_training.validation_rollout_suite','worker','--directory',str(root),'--rank',str(rank)]
        with (root/f'inference-rank{rank:02d}.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
    save_json(root/'launch.json',dict(pid=os.getpid(),workers=[dict(rank=i,gpu=suite['gpus'][i],pid=p.pid) for i,p in enumerate(processes)],started_unix=time.time()))
    while any(p.poll() is None for p in processes):
        ranks=[]
        for rank in range(len(processes)):
            path=root/f'status-rank{rank:02d}.json'
            ranks.append(json.loads(path.read_text()) if path.exists() else dict(status='starting',rank=rank))
        save_json(root/'status.json',dict(status='running',samples=suite['samples'],
                  completed=sum(r.get('samples',0) if r['status']=='complete' else r.get('completed',0) for r in ranks),ranks=ranks,updated_unix=time.time()))
        time.sleep(30)
    codes=[p.wait() for p in processes]
    records=[json.loads((root/e['relative_dir']/'result.json').read_text()) for e in suite['episodes'] if (root/e['relative_dir']/'result.json').exists()]
    success=all(code==0 for code in codes) and len(records)==len(suite['episodes']) and all(r['status']=='complete' for r in records)
    result=dict(status='complete' if success else 'failed',worker_exit_codes=codes,completed_episodes=len(records),
                samples=sum(r.get('samples',0) for r in records),expected_samples=suite['samples'],finished_unix=time.time())
    save_json(root/'result.json',result);save_json(root/'status.json',result)
    if not success:raise RuntimeError('One or more rollout workers failed; preserve partial artifacts')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','launch','worker'])
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--split',type=Path)
    parser.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    parser.add_argument('--rank',type=int)
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    {'setup':setup,'launch':launch,'worker':run_worker}[args.action](args)


if __name__=='__main__':
    main()
