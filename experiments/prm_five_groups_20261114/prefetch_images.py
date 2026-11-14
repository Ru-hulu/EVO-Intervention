"""Bounded CPU image preparation to keep first-group GPUs supplied with inputs.

Decode using the same pinned backend into a private staging directory. Publish
only a complete verified directory, atomically and without replacing anything
an already-running consumer created. Main-view caches reuse identical PNGs.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

from . import eval_common
from .prepare_cache import save, sha
from .overlap_guard import resources

HERE=eval_common.EVAL/'image-prefetch'


def publish(source, destination):
    """Linux RENAME_NOREPLACE also refuses an empty consumer-owned directory."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    rename=ctypes.CDLL(None,use_errno=True).renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(destination),1)==0:return True
    error=ctypes.get_errno()
    if error in (errno.EEXIST,errno.ENOTEMPTY):return False
    raise OSError(error,os.strerror(error),str(destination))


def link_main(episode, full, stage):
    p=json.loads((full/'verification.json').read_text())
    assert p['input']=={'video_hashes':episode['video_hashes'],'frame_indices':episode['frame_indices']}
    assert p['decoded_native_frames']==episode['frames']
    stage.mkdir(parents=True,exist_ok=True);hashes={}
    for index in episode['frame_indices']:
        name=f'{index:08d}-0.png';digest=p['png_sha256'][name]
        assert sha(full/name)==digest
        os.link(full/name,stage/name);hashes[name]=digest
    save(stage/'verification.json',{'input':{'video_hashes':episode['video_hashes'][:1],
         'frame_indices':episode['frame_indices']},'png_sha256':hashes,'decoded_native_frames':episode['frames'],
         'lossless':True,'missing_view_policy':'RGB black 64x64 placeholder',
         'reuse':'verified byte-identical main-view PNG hardlinks'})


def prepare(episode,main_only):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import av
    assert av.__version__=='19.0.2'
    original=eval_common.IMAGE_ROOT
    destination=original/episode['key']/('fps5-main' if main_only else 'fps5')
    if destination.exists():return {'key':episode['key'],'main_only':main_only,'status':'consumer_owned_or_ready'}
    staging=original/'.prefetch-staging'/str(uuid.uuid4())
    staged=staging/episode['key']/destination.name
    started=time.time()
    try:
        full=original/episode['key']/'fps5'
        linked=main_only and (full/'verification.json').is_file()
        if linked:link_main(episode,full,staged)
        else:
            eval_common.IMAGE_ROOT=staging
            eval_common.ensure_images(episode,main_only)
        proof=json.loads((staged/'verification.json').read_text())
        assert proof['decoded_native_frames']==episode['frames']
        for name,digest in proof['png_sha256'].items():assert sha(staged/name)==digest
        done=publish(staged,destination)
        return {'key':episode['key'],'main_only':main_only,'status':'published' if done else 'consumer_won_race',
                'main_view_hardlinks':linked,'png_count':len(proof['png_sha256']),'seconds':time.time()-started}
    finally:
        eval_common.IMAGE_ROOT=original
        shutil.rmtree(staging,ignore_errors=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    HERE.mkdir(parents=True,exist_ok=True)
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'controller.pid').write_text(str(os.getpid()));os.nice(10)
    manifest=json.loads((eval_common.EVAL/'group1-manifest.json').read_text())
    assert manifest['expected_episodes']==385 and manifest['expected_samples']==64827
    episodes=manifest['episodes']
    # Robometer consumes forward shard order; prepare the far end first.
    full=[(r,False) for r in reversed(episodes) if not r['input_issues'] and
          not (eval_common.IMAGE_ROOT/r['key']/'fps5/verification.json').exists()]
    main=[(r,True) for r in episodes if not any(i.get('view')==0 for i in r['input_issues']) and
          not (eval_common.IMAGE_ROOT/r['key']/'fps5-main/verification.json').exists()]
    tasks=full+main;initial_oom=resources()['oom_kill'];started=time.time();results=[];pending={};index=0
    save(HERE/'status.json',{'status':'starting','pid':os.getpid(),'started':started,'updated':started,
         'total_candidates':len(tasks),'processed':0,'cpu_workers':2,'gpu_workers':0})
    with ProcessPoolExecutor(max_workers=2) as pool:
        while index<len(tasks) or pending:
            sample=resources()
            blocked=(sample['oom_kill']>initial_oom or sample['under_oom'] or
                     sample['resident_and_shmem']>.65*sample['memory_limit'] or
                     sample['memory_working_set']>.85*sample['memory_limit'] or sample['disk_free']<150*2**30)
            while not blocked and len(pending)<2 and index<len(tasks):
                row,mode=tasks[index];index+=1
                pending[pool.submit(prepare,row,mode)]=(row['key'],mode)
            if pending:
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    key,mode=pending.pop(future)
                    try:result=future.result()
                    except Exception:result={'key':key,'main_only':mode,'status':'failed','error':traceback.format_exc()}
                    results.append(result);save(HERE/'results.json',results)
            elif blocked:time.sleep(5)
            save(HERE/'status.json',{'status':'paused_for_resources' if blocked else 'preparing',
                'pid':os.getpid(),'started':started,'updated':time.time(),'cpu_workers':2,'gpu_workers':0,
                'total_candidates':len(tasks),'processed':len(results),'active':list(pending.values()),
                'published':sum(r['status']=='published' for r in results),
                'failures':[r for r in results if r['status']=='failed'],'resource_sample':sample})
    coverage={mode:sum((eval_common.IMAGE_ROOT/r['key']/mode/'verification.json').exists() for r in episodes)
              for mode in ['fps5','fps5-main']}
    save(HERE/'completion.json',{'finished':time.time(),'coverage':coverage,'expected':385,
         'published':sum(r['status']=='published' for r in results),'failures':[r for r in results if r['status']=='failed'],
         'manifest_sha256':sha(eval_common.EVAL/'group1-manifest.json'),'decoder':'PyAV 19.0.2',
         'protocol_changed':False,'cpu_workers':2,'gpu_workers':0})
    state=json.loads((HERE/'status.json').read_text());state.update(status='completed',updated=time.time(),coverage=coverage)
    save(HERE/'status.json',state)


if __name__=='__main__':run()
