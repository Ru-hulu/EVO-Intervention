"""Supplement only missing R05 PT outside the original Recover dataset.

First effective pour per cup and its actual associated put-down accumulate raw
Qwen3-VL visual change. Repeated pours/put-downs and idle intervals stay flat.
The waiting controller uses an otherwise free GPU during a cache phase, never
sharing a training GPU or restarting a healthy job.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .prepare_cache import BASE, DATA, sha, save
from .overlap_guard import resources, pause_reasons, alive

HERE=BASE/'recover-pour-labels'
OUTPUT=BASE/'labels/Recover/R05'
MODEL=Path('/cache/cvpzhr/models/Qwen3-VL-4B-Instruct')
REVISION='ebb281ec70b05090aa6165b016eac8ec08e71b17'
VISION_SHA='050e4a6234f755fc0efede85bd4b2c12c0e5387de122a34e68bd956b8b46fd27'
PLAN=Path(__file__).parent/'experiment-plan.json'
CODE=Path(__file__).parents[2]
PYTHON='/cache/cvpzhr/envs/forest-labels/bin/python'


def first_visit_mask(annotation):
    """Inclusive source boundaries; never infer a put-down that is not annotated."""
    import numpy as np
    if annotation['frame_interval']!='inclusive' or annotation['frame_index_base']!=0:
        raise ValueError('Unsupported annotation boundaries')
    n=annotation['total_frames'];mask=np.zeros(n,dtype=bool);covered=np.zeros(n,dtype=bool)
    seen={};previous=None;last_end=-1;events=[]
    for stage in annotation['stages']:
        a,b=stage['source_start_frame'],stage['source_end_frame']
        if not 0<=a<=b<n or a<=last_end or stage['outcome']!='completed':
            raise ValueError('Overlapping, unordered, incomplete or invalid action segment')
        last_end=b;covered[a:b+1]=True
        action=stage['action'];target=stage.get('target_position')
        if action=='pour':
            if target not in {'left','middle','right'}:raise ValueError('Unknown cup')
            occurrence=seen.get(target,0)+1
            if stage['target_occurrence']!=occurrence:raise ValueError('Inconsistent pour occurrence')
            seen[target]=occurrence;include=occurrence==1
            previous=(target,occurrence,False)
        elif action=='put_down':
            if previous is None or previous[2]:raise ValueError('Put-down without preceding pour')
            target,occurrence,_=previous
            if stage['after_pour_target']!=target or stage['target_occurrence']!=occurrence:
                raise ValueError('Put-down is associated with the wrong pour')
            include=occurrence==1;previous=(target,occurrence,True)
        else:raise ValueError('Unknown action')
        mask[a:b+1]=include
        events.append({'segment_index':stage['segment_index'],'action':action,'target':target,
                       'occurrence':occurrence,'start':a,'end_inclusive':b,'contributes':include})
    if set(seen)!={'left','middle','right'}:raise ValueError('Missing first pour for a cup')
    for idle in annotation['idle_intervals']:
        a,b=idle['source_start_frame'],idle['source_end_frame']
        if not 0<=a<=b<n or covered[a:b+1].any():raise ValueError('Invalid or overlapping idle range')
        covered[a:b+1]=True
    if not covered.all():raise ValueError('Unannotated source frames; do not invent idle labels')
    return mask,events


def progress_from_deltas(raw,annotation):
    import numpy as np
    from mem_lite.datasets.prepare_proprimem import whole_video_progress
    raw=np.asarray(raw,dtype=np.float64)
    mask,events=first_visit_mask(annotation)
    if raw.shape!=mask.shape or not np.isfinite(raw).all() or (raw<0).any() or raw[0]!=0:
        raise ValueError('Invalid native-frame visual deltas')
    effective=np.where(mask,raw,0.)
    pt=whole_video_progress(effective)
    increments=np.diff(pt,prepend=pt[0])
    assert (increments[~mask]==0).all() and (increments>=0).all()
    return effective,pt,events


def preflight():
    HERE.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text());root=DATA/'Recover';manifest=root/'manifest.json'
    expected=plan['datasets']['suites']['Recover']['manifest_sha256']
    assert sha(manifest)==expected
    task=next(t for t in json.loads(manifest.read_text())['tasks'] if t['task_id']=='R05')
    meta=root/task['trajectory_index'];assert sha(meta)==task['trajectory_index_sha256']
    rows={r['sample_id']:r for r in map(json.loads,meta.read_text().splitlines())}
    records=[]
    for split in ['train','val']:
        for sample in task[split]:
            r=rows[sample];ep=f"episode_{r['episode_index']:06d}"
            annotation=meta.parent/r['annotation_path'];video=meta.parent/r['video_path']
            assert sha(annotation)==r['annotation_sha256']
            a=json.loads(annotation.read_text());assert a['schema_version']=='repeated-pouring-stage-annotation-v1'
            assert a['total_frames']==r['total_frames'] and a['video_sha256']==r['video_sha256']==sha(video)
            assert a['fps']==r['fps']==30
            mask,events=first_visit_mask(a)
            assert not (meta.parent/f'annotation/{ep}.progress.json').exists()
            records.append({'episode':ep,'sample_id':sample,'split':split,'frames':r['total_frames'],
                'annotation':str(annotation),'annotation_sha256':sha(annotation),'video':str(video),
                'video_sha256':sha(video),'contributing_frames':int(mask.sum()),'events':events})
    assert len(records)==35 and sum(r['split']=='train' for r in records)==28
    before=HERE/'source-files-before.json'
    if not before.exists():
        save(before,{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    receipt={'records':records,'manifest_sha256':expected,'metadata_sha256':sha(meta),
        'plan_sha256':sha(PLAN),'source_file_inventory':str(before),
        'source_file_inventory_sha256':sha(before),'frames':sum(r['frames'] for r in records),
        'label_output':str(OUTPUT),'original_recover_files_modified':False}
    path=HERE/'preflight.json'
    if path.exists():assert json.loads(path.read_text())==receipt
    else:save(path,receipt)
    return receipt


def load_encoder(device):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from mem_lite.memory.download_clip import verify
    from mem_lite.memory.vision import _vision_state,_vision_sha256
    receipt=json.loads((MODEL/'vision-download-receipt.json').read_text())
    assert receipt['repo']=='Qwen/Qwen3-VL-4B-Instruct' and receipt['revision']==REVISION
    for f in receipt['files']:assert verify(MODEL/f['name'],f['identity']),f['name']
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True).vision_config
    with init_empty_weights(include_buffers=False):
        model=Qwen3VLVisionModel._from_config(config,attn_implementation='sdpa')
    state=_vision_state(MODEL);digest=_vision_sha256(state)
    assert digest==VISION_SHA
    dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32
    model.load_state_dict({k:v.to(dtype=dtype) for k,v in state.items()},strict=True,assign=True)
    model=model.to(device=device).eval().requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    return model,processor,digest


def read_records():
    proof=json.loads((HERE/'preflight.json').read_text())
    assert proof['plan_sha256']==sha(PLAN)
    assert proof['manifest_sha256']==sha(DATA/'Recover/manifest.json')
    assert proof['source_file_inventory_sha256']==sha(proof['source_file_inventory'])
    return proof['records']


def verify_episode(row):
    import numpy as np
    meta=OUTPUT/f"{row['episode']}.progress.json";csvpath=meta.with_suffix('.csv')
    d=json.loads(meta.read_text());assert sha(csvpath)==d['csv_sha256']
    assert d['video_sha256']==sha(row['video'])==row['video_sha256']
    assert d['annotation_sha256']==sha(row['annotation'])==row['annotation_sha256']
    assert d['model_revision']==REVISION and d['plan_sha256']==sha(PLAN)
    with csvpath.open() as stream:values=list(csv.DictReader(stream))
    assert [int(x['frame_index']) for x in values]==list(range(row['frames']))
    raw=np.array([float(x['delta']) for x in values]);actual=np.array([float(x['pt']) for x in values])
    effective,expected,_=progress_from_deltas(raw,json.loads(Path(row['annotation']).read_text()))
    assert np.array_equal(actual,expected)
    assert np.array_equal(effective,np.array([float(x['effective_delta']) for x in values]))
    assert actual[0]==0 and actual[-1]==1
    return {'episode':row['episode'],'split':row['split'],'frames':row['frames'],
            'csv_sha256':sha(csvpath),'vision_sha256':d['vision_sha256']}


def free_label_slot(gpu,initial_oom):
    """Cached training records are checked without changing the live controller."""
    status=json.loads((BASE/'status.json').read_text())
    if time.time()-status['updated']>90:return False
    group=next(g for g in status['groups'] if gpu in g['gpus'])
    if group['phase']!='cache':return False
    sample=resources()
    return not pause_reasons(sample,gpu,initial_oom)


def worker(gpu):
    import numpy as np
    import torch
    from mem_lite.datasets.prepare_progress import extract_deltas
    lock=(HERE/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    OUTPUT.mkdir(parents=True,exist_ok=True);initial_oom=resources()['oom_kill']
    model,processor,digest=load_encoder('cuda')
    audit={'revision':REVISION,'model':str(MODEL),'vision_sha256':digest,'strict_load':True,
        'model_class':type(model).__name__,'dtype':str(next(model.parameters()).dtype),
        'batch_size':16,'min_pixels':4096,'max_pixels':262144,'all_native_frames':True,
        'cuda_memory_fraction':.15,'plan_sha256':sha(PLAN),'code_sha256':sha(__file__)}
    save(HERE/f'initialization-{os.getpid()}.json',audit)
    for row in read_records():
        meta=OUTPUT/f"{row['episode']}.progress.json"
        if meta.exists():verify_episode(row);continue
        if not free_label_slot(gpu,initial_oom):
            save(HERE/'worker-status.json',{'status':'paused_for_training_or_resources','pid':os.getpid(),'updated':time.time()})
            return 75
        save(HERE/'worker-status.json',{'status':'running','episode':row['episode'],'gpu':gpu,'pid':os.getpid(),'updated':time.time()})
        assert sha(row['annotation'])==row['annotation_sha256'] and sha(row['video'])==row['video_sha256']
        rawpath=OUTPUT/f"{row['episode']}.deltas.npz";rawreceipt=rawpath.with_suffix('.json')
        if rawpath.exists():
            receipt=json.loads(rawreceipt.read_text())
            assert receipt['sha256']==sha(rawpath) and receipt['source']==row and receipt['audit']==audit
            with np.load(rawpath,allow_pickle=False) as data:raw=data['delta']
        else:
            raw,info=extract_deltas(row['video'],model,processor,16,262144)
            assert info['frame_count']==row['frames'] and info['fps']==30
            tmp=rawpath.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez(f,delta=raw,timestamps=info['timestamps_seconds'])
            tmp.replace(rawpath);save(rawreceipt,{'sha256':sha(rawpath),'source':row,'audit':audit})
        effective,pt,events=progress_from_deltas(raw,json.loads(Path(row['annotation']).read_text()))
        csvpath=meta.with_suffix('.csv');tmp=csvpath.with_suffix('.csv.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['frame_index','delta','effective_delta','pt'])
            w.writerows(zip(range(row['frames']),raw,effective,pt,strict=True))
        tmp.replace(csvpath)
        save(meta,dict(audit,model_revision=REVISION,video_sha256=row['video_sha256'],
            annotation_sha256=row['annotation_sha256'],csv_sha256=sha(csvpath),events=events,
            rule='first effective pour per cup and associated actual put-down accumulate raw delta; repeats/idle flat; normalize final cumulative sum',
            raw_delta_sha256=sha(rawpath),total_frames=row['frames'],original_recover_files_modified=False))
        verify_episode(row)
    save(HERE/'worker-status.json',{'status':'completed','pid':os.getpid(),'updated':time.time()})
    return 0


def finalize():
    proofs=[verify_episode(row) for row in read_records()]
    assert len(proofs)==35 and len({p['vision_sha256'] for p in proofs})==1
    expected=json.loads((HERE/'source-files-before.json').read_text());root=DATA/'Recover'
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(expected)
    for rel,digest in expected.items():assert sha(root/rel)==digest,rel
    receipt={'status':'passed','episodes':35,'train':28,'val':7,'proofs':proofs,
        'source_files_unchanged':len(expected),'recover_manifest_unchanged':True,
        'all_native_frames_verified':True,'plan_sha256':sha(PLAN),'finished':time.time()}
    save(HERE/'verification.json',receipt)
    return receipt


def wait_and_run():
    HERE.mkdir(parents=True,exist_ok=True)
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=HERE/'worker-launch.json'
    if previous.exists() and alive(json.loads(previous.read_text())['pid']):
        raise RuntimeError('Prior label worker still alive; inspect before adopting or resuming')
    (HERE/'controller.pid').write_text(str(os.getpid()))
    read_records();started=time.time()
    while True:
        if all((OUTPUT/f"{r['episode']}.progress.json").exists() for r in read_records()):
            finalize()
            # GPU worker has exited before queue insertion; healthy cache/train jobs continue.
            from .prepare_cache import make_index
            make_index(append_recover=True)
            save(HERE/'status.json',{'status':'completed_and_indexed','finished':time.time()})
            return
        status=json.loads((BASE/'status.json').read_text());initial_oom=resources()['oom_kill'];chosen=None
        if time.time()-status['updated']<90:
            for group in status['groups']:
                if group['phase']=='cache':
                    chosen=next((g for g in group['gpus'] if not (BASE/f'drain-gpu-{g}').exists()),None)
                    if chosen is not None:break
        if chosen is None:
            save(HERE/'status.json',{'status':'waiting_for_free_cache_phase_gpu','controller_pid':os.getpid(),'started':started,'updated':time.time()})
            time.sleep(30);continue
        flag=BASE/f'drain-gpu-{chosen}'
        try:flag.touch(exist_ok=False)
        except FileExistsError:continue
        save(HERE/'status.json',{'status':'draining_cache_cooperatively','gpu':chosen,'controller_pid':os.getpid(),'updated':time.time()})
        try:
            while free_label_slot(chosen,initial_oom):
                workers=json.loads((BASE/'cache-worker-pids.json').read_text())
                busy=any(w['gpu']==chosen and alive(w['pid']) for w in workers)
                uuids={int(line.split(',')[0]):line.split(',')[1].strip() for line in subprocess.check_output(
                    ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
                used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
                if not busy and uuids[chosen] not in used and resources()['gpus'][chosen]['used_mib']<1000:break
                time.sleep(10)
            else:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(chosen),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True')
            with (HERE/'worker.log').open('a') as log:
                p=subprocess.Popen([PYTHON,'-m','experiments.prm_five_groups_20261114.recover_pour_labels','worker','--gpu',str(chosen)],
                    cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(HERE/'worker-launch.json',{'pid':p.pid,'gpu':chosen,'started':time.time()})
            rc=p.wait()
            if rc not in [0,75]:raise RuntimeError(f'Label worker failed with exit {rc}; preserve outputs and inspect worker.log')
        finally:
            latest=json.loads((BASE/'status.json').read_text())
            group=next(g for g in latest['groups'] if chosen in g['gpus'])
            if group['phase']=='cache':flag.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','worker','finalize','wait'])
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.mode=='worker':raise SystemExit(worker(args.gpu))
    if args.mode=='preflight':
        proof=preflight();print(json.dumps({'episodes':len(proof['records']),'frames':proof['frames']}))
    elif args.mode=='finalize':print(json.dumps(finalize()))
    else:
        try:wait_and_run()
        except BlockingIOError:raise SystemExit('Another label controller holds the lock')
        except BaseException:
            save(HERE/'status.json',{'status':'failed','error':traceback.format_exc(),'updated':time.time()})
            raise
