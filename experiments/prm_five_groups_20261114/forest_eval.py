"""FoREST group-1 worker, gated on the matching verified final suite checkpoint."""
import argparse
import fcntl
import json
import os
import time
import traceback

from .eval_common import EVAL,PLAN,ensure_images,result_path,trajectory_metrics,aggregate
from .prepare_cache import save,sha
from .forest_adapter import load_verified_forest


def verify_prefix(prefix,row,manifest_sha,checkpoint_sha):
    n=len(prefix['predictions'])
    assert n<=len(row['frame_indices']) and len(prefix['frame_indices'])==len(prefix['target_pt'])==n
    assert prefix['episode']==row['key'] and prefix['manifest_sha256']==manifest_sha
    assert prefix['checkpoint_sha256']==checkpoint_sha
    assert prefix['frame_indices']==row['frame_indices'][:n] and prefix['target_pt']==row['target_pt'][:n]
    assert len(prefix['history_lengths'])==len(prefix['raw_nonfinite'])==n
    return n


def images_at(root,row,index):
    from PIL import Image
    images=[]
    for v,path in enumerate(row['videos']):
        if path:
            with Image.open(root/f'{index:08d}-{v}.png') as image:images.append(image.convert('RGB'))
        else:images.append(None)
    return images


def run(args):
    import torch
    name=f'forest-{args.suite.lower()}-group1-{args.shard}';jobs=EVAL/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    lock=(jobs/f'{name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    state={'model':'forest','suite':args.suite,'shard':args.shard,'pid':os.getpid(),
        'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'status':'verifying_checkpoint','started':time.time(),
        'completed_episodes':0,'completed_samples':0,'memory_fraction':args.memory_fraction}
    def update(**kw):state.update(kw,updated=time.time());save(jobs/f'{name}.json',state)
    update()
    try:
        manifest_path=EVAL/'group1-manifest.json';m=json.loads(manifest_path.read_text());digest=sha(manifest_path)
        assert m['plan_sha256']==sha(PLAN)
        selected=[r for i,r in enumerate(m['episodes']) if r['suite']==args.suite and i%4==args.shard]
        blocked=[{'key':r['key'],'issues':r['input_issues']} for r in selected if r.get('input_issues')]
        update(total_episodes=len(selected),input_blockers=blocked)
        runtime,audit=load_verified_forest(args.suite)
        audit.update(manifest_sha256=digest,plan_sha256=sha(PLAN),worker_code_sha256=sha(__file__))
        save(jobs/f'{name}-initialization.json',audit);update(status='running')
        partial=jobs/f'{name}-partial.json';pause=EVAL/f'pause-{name}'
        for row in selected:
            if row.get('input_issues'):continue
            path=result_path('forest',row['key'])
            if path.exists():
                result=json.loads(path.read_text());assert result['checkpoint_sha256']==audit['checkpoint_sha256']
                assert result['manifest_sha256']==digest and result['frame_indices']==row['frame_indices']
                assert result['target_pt']==row['target_pt'] and len(result['predictions'])==len(row['target_pt'])
                update(completed_episodes=state['completed_episodes']+1,
                       completed_samples=state['completed_samples']+len(result['predictions']));continue
            update(episode=row['key'],phase='decode_and_verify_images')
            root=ensure_images(row);pred=[];lengths=[];raw=[];runtime.reset()
            if partial.exists():
                prefix=json.loads(partial.read_text())
                if prefix['episode']==row['key']:
                    n=verify_prefix(prefix,row,digest,audit['checkpoint_sha256'])
                    pred=prefix['predictions'];lengths=prefix['history_lengths'];raw=prefix['raw_nonfinite']
                    update(phase='rebuilding_native_memory_from_saved_prefix')
                    for i in range(n):
                        if pause.exists():update(status='paused_by_resource_guard');return
                        position=row['frame_indices'][i]
                        runtime.advance_memory(images_at(root,row,position),position,row['key'],row['task_prompt'])
            state['completed_samples']+=len(pred)
            while len(pred)<len(row['frame_indices']):
                if pause.exists():update(status='paused_by_resource_guard');return
                i=len(pred);position=row['frame_indices'][i]
                value=runtime.predict(images_at(root,row,position),position,row['key'],row['task_prompt'])
                pred.append(value['prediction']);lengths.append(value['history_length']);raw.append(value['raw_nonfinite'])
                save(partial,{'episode':row['key'],'manifest_sha256':digest,'checkpoint_sha256':audit['checkpoint_sha256'],
                    'predictions':pred,'history_lengths':lengths,'raw_nonfinite':raw,
                    'frame_indices':row['frame_indices'][:len(pred)],'target_pt':row['target_pt'][:len(pred)]})
                update(phase='inference',samples_in_episode=len(pred),completed_samples=state['completed_samples']+1)
            result={'model':'forest','regime':'suite_fine_tuned','suite':row['suite'],'key':row['key'],
                'task_id':row['task_id'],'group':1,'condition':'baseline','input_fps':5,
                'manifest_sha256':digest,'plan_sha256':sha(PLAN),'checkpoint':audit['checkpoint'],
                'checkpoint_sha256':audit['checkpoint_sha256'],'base_revision':audit['base_revision'],
                'initialization_audit':str(jobs/f'{name}-initialization.json'),'uses_future_endpoint_reference':False,
                'frame_indices':row['frame_indices'],'timestamps':row['timestamps'],'target_pt':row['target_pt'],
                'progress_sha256':row['progress_sha256'],'source_video_sha256':row['video_hashes'],
                'predictions':pred,'history_lengths':lengths,'raw_responses':[None]*len(pred),'raw_nonfinite':raw,
                'metrics':trajectory_metrics(pred,row['target_pt'],row['timestamps']),'finished':time.time()}
            save(path,result);partial.unlink(missing_ok=True);update(completed_episodes=state['completed_episodes']+1)
        update(status='completed_with_input_blockers' if blocked else 'completed',finished=time.time());aggregate('forest')
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',choices=['State','Order','Count'],required=True)
    parser.add_argument('--shard',type=int,required=True);parser.add_argument('--memory-fraction',type=float,default=.30)
    args=parser.parse_args();assert 0<=args.shard<4 and 0<args.memory_fraction<=.35;run(args)
