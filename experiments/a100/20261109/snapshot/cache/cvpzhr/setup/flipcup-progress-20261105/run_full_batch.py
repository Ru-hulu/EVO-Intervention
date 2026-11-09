"""Benchmark and distribute unchanged single-frame QwenDelta across four GPUs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'pydeps'))
ENCODER = None
WORKER = None
BARRIER = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def initialize_worker(slots, model, barrier=None):
    global ENCODER, WORKER, BARRIER
    gpu, slot = slots.get(timeout=60)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from original_video_to_delta import QwenDelta
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.08, 0)
    ENCODER = QwenDelta(Path(model), 'cuda', 262144)
    # Warm CUDA before the synchronized benchmark timer.
    ENCODER.encode(torch.zeros((3,720,1280),dtype=torch.uint8))
    torch.cuda.synchronize()
    WORKER = {'pid':os.getpid(), 'gpu':gpu, 'slot':slot}
    BARRIER = barrier


def compute(item, dataset, fingerprint, benchmark_frames=None):
    import av
    import torch
    dataset = Path(dataset)
    name = item['episode']
    ENCODER.last_image, ENCODER.last_phi = None, None
    if benchmark_frames:
        BARRIER.wait(timeout=240)
    started = time.perf_counter()
    status = {**WORKER, 'episode':name, 'frames':0, 'total_frames':item['frames'], 'phase':'running'}
    status_path = ROOT/'full_workers'/f'worker_{os.getpid()}.json'
    if not benchmark_frames:
        atomic_json(status_path,status)
    values, previous = [], None
    with av.open(str(dataset/'videos/main_view'/f'{name}.mp4')) as video:
        video.streams.video[0].thread_count = 1
        for frame in video.decode(video=0):
            current = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1)
            delta = 0.0 if previous is None else ENCODER.image2delta(previous,current)
            if not math.isfinite(delta) or delta < 0:
                raise ValueError(f'Invalid delta: {name} frame {len(values)}')
            values.append(f'{delta:.6f}')
            previous = current
            if benchmark_frames and len(values) >= benchmark_frames:
                break
            if not benchmark_frames and len(values)%150 == 0:
                status.update(frames=len(values),seconds=time.perf_counter()-started)
                atomic_json(status_path,status)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    if benchmark_frames:
        return {**WORKER, 'seconds':seconds,'frames':len(values),'values':values}
    if len(values)!=item['frames']:
        raise ValueError(f'Wrong frame count: {name}')
    target=dataset/'delta'/f'{name}-delta.txt'
    temporary=target.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text('\n'.join(values)+'\n')
    temporary.replace(target)
    receipt={'episode':name,'frames':len(values),'source_sha256':item['sha256'],
             'delta_sha256':digest(target),'fingerprint':fingerprint,'seconds':seconds,
             'worker':WORKER}
    atomic_json(dataset/'delta'/f'{name}.receipt.json',receipt)
    status.update(phase='completed',frames=len(values),seconds=seconds)
    atomic_json(status_path,status)
    return receipt


def verified_existing(dataset, item, fingerprint):
    target=dataset/'delta'/f'{item["episode"]}-delta.txt'
    receipt_path=dataset/'delta'/f'{item["episode"]}.receipt.json'
    if not target.exists() and not receipt_path.exists():
        return False
    if not target.exists() or not receipt_path.exists():
        raise ValueError(f'Incomplete prior result: {item["episode"]}')
    receipt=json.loads(receipt_path.read_text())
    values=[float(x) for x in target.read_text().splitlines()]
    assert receipt['source_sha256']==item['sha256']
    assert receipt['fingerprint']==fingerprint and receipt['delta_sha256']==digest(target)
    assert receipt['frames']==len(values)==item['frames'] and values[0]==0
    assert all(math.isfinite(v) and v>=0 for v in values)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    parser.add_argument('--benchmark-frames',type=int,default=128)
    args=parser.parse_args()
    lock=(ROOT/'full_batch.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'phase':'preflight','started_at':datetime.now(timezone.utc).isoformat()}
    def update(**changes):
        state.update(changes,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(ROOT/'full_compute_status.json',state)
    update()
    try:
        source=json.loads((ROOT/'source_manifest.json').read_text())
        provenance=json.loads((ROOT/'provenance.json').read_text())
        fingerprint=provenance['fingerprint']
        expected={k:v for k,v in provenance.items() if k!='fingerprint'}
        assert hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()==fingerprint
        assert digest(ROOT/'original_video_to_delta.py')==provenance['original_encoder_sha256']
        for name,version in provenance['versions'].items():
            assert importlib.metadata.version(name)==version,(name,version)
        for name,spec in provenance['model']['downloaded_files'].items():
            assert digest(args.model_dir/name)==spec['sha256'],name
        annotations=[]
        for item in source['videos']:
            name=item['episode']
            assert digest(args.dataset_root/'videos/main_view'/f'{name}.mp4')==item['sha256'],name
            path=args.dataset_root/'videos/annotation'/f'{name}.json'
            a=json.loads(path.read_text())
            assert a['num_stages']==13 and a['video_sha256']==item['sha256']
            assert a['boundary_reviews']==[] and a['total_frames']==item['frames']
            annotations.append({'episode':name,'sha256':digest(path)})
        assert len(source['videos'])==35
        todo=[v for v in source['videos'] if not verified_existing(args.dataset_root,v,fingerprint)]
        reused=[v for v in source['videos'] if v not in todo]
        (ROOT/'full_workers').mkdir(exist_ok=True)
        run_provenance={'delta_provenance':provenance,'annotations':annotations,
                        'normalizer_sha256':digest(ROOT/'prepare_flipcup_progress.py'),
                        'runner_sha256':digest(Path(__file__)), 'gpus':args.gpus,
                        'reused_episodes':[v['episode'] for v in reused]}
        atomic_json(ROOT/'full_provenance.json',run_provenance)
        update(total_episodes=35,total_frames=source['total_frames'],
               remaining_episodes=len(todo),remaining_frames=sum(v['frames'] for v in todo),
               episodes_completed=len(reused),frames_completed=sum(v['frames'] for v in reused))
        ctx=mp.get_context('spawn')
        benchmarks=[]
        if todo:
            baseline=source['videos'][0]
            known=(args.dataset_root/'delta'/f'{baseline["episode"]}-delta.txt').read_text().splitlines()[:args.benchmark_frames]
            for workers in [1,2,4,8]:
                update(phase='benchmarking',benchmark_workers_per_gpu=workers)
                slots=ctx.Queue()
                for slot in range(workers):slots.put((args.gpus[0],slot))
                barrier=ctx.Barrier(workers)
                load_started=time.monotonic()
                with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,
                        initializer=initialize_worker,initargs=(slots,str(args.model_dir),barrier)) as pool:
                    futures=[pool.submit(compute,baseline,str(args.dataset_root),fingerprint,args.benchmark_frames) for _ in range(workers)]
                    results=[f.result() for f in futures]
                for result in results:
                    assert result['values']==known,'Parallel encoder differs from the approved preview'
                elapsed=max(r['seconds'] for r in results)
                throughput=sum(r['frames'] for r in results)/elapsed
                row={'workers_per_gpu':workers,'frames_per_second_per_gpu':throughput,
                     'inference_seconds':elapsed,'including_startup_seconds':time.monotonic()-load_started,
                     'exact_delta_match':True}
                benchmarks.append(row)
                atomic_json(ROOT/'full_benchmarks.json',benchmarks)
                print(json.dumps({'benchmark':row}),flush=True)
                if len(benchmarks)>1 and throughput<benchmarks[-2]['frames_per_second_per_gpu']*1.10:
                    break
            best=max(benchmarks,key=lambda x:x['frames_per_second_per_gpu'])
            workers_per_gpu=best['workers_per_gpu']
            workers=len(args.gpus)*workers_per_gpu
            slots=ctx.Queue()
            for slot in range(workers_per_gpu):
                for gpu in args.gpus:slots.put((gpu,slot))
            run_started=time.monotonic()
            completed_frames=sum(v['frames'] for v in reused)
            completed_episodes=len(reused)
            update(phase='running',workers=workers,workers_per_gpu=workers_per_gpu,
                   benchmark=best,estimated_minutes=sum(v['frames'] for v in todo)/
                   (best['frames_per_second_per_gpu']*len(args.gpus))/60)
            with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,
                    initializer=initialize_worker,initargs=(slots,str(args.model_dir))) as pool:
                # Longest episodes first keep the final wave balanced.
                pending={pool.submit(compute,item,str(args.dataset_root),fingerprint):item
                         for item in sorted(todo,key=lambda x:x['frames'],reverse=True)}
                while pending:
                    done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        result=future.result()
                        item=pending.pop(future)
                        completed_frames+=item['frames'];completed_episodes+=1
                        print(json.dumps({'completed':result}),flush=True)
                    partial=0
                    current=[]
                    pending_names={v['episode'] for v in pending.values()}
                    for path in (ROOT/'full_workers').glob('worker_*.json'):
                        status=json.loads(path.read_text())
                        if status['episode'] in pending_names:
                            partial+=status['frames'];current.append(status)
                    update(episodes_completed=completed_episodes,frames_completed=completed_frames,
                           in_progress_frames=partial,active_workers=current,
                           elapsed_compute_seconds=time.monotonic()-run_started)
        update(phase='normalizing',active_workers=[])
        for item in source['videos']:
            assert verified_existing(args.dataset_root,item,fingerprint)
        from prepare_flipcup_progress import prepare_batch
        from validate_preview import validate
        output=args.dataset_root/'progress_13stages'
        result=prepare_batch(args.dataset_root,output_dir=output)
        validate(args.dataset_root,output,[v['episode'] for v in source['videos']])
        assert result['episode_count']==35 and result['total_frames']==source['total_frames']
        assert result['rollback_count']==70
        update(phase='completed',episodes_completed=35,frames_completed=source['total_frames'],
               in_progress_frames=0,output_dir=str(output),
               result={k:v for k,v in result.items() if k!='episodes'})
        print(json.dumps(state),flush=True)
    except Exception as exc:
        update(phase='failed',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
