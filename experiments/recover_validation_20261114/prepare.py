"""Freeze held-out episodes, preserve PT, supplement two R05 labels, decode exactly."""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from .common import *
from experiments.prm_five_groups_20261114.prepare_cache import RECOVER_PROMPTS, progress_values
from experiments.prm_five_groups_20261114.eval_common import sampling_indices


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'selection.json').exists(), 'Do not replace frozen selection'
    manifest = read(DATA/'manifest.json'); chosen = select(manifest['tasks']); records = []
    source = {str(DATA/'manifest.json'): sha(DATA/'manifest.json')}
    train_selection = read('/cache/cvpzhr/setup/recover-six-model-caches-20261114/selection.json')
    train_keys = {r['key'] for r in train_selection['records']}
    for t in manifest['tasks']:
        meta = DATA/t['trajectory_index']; assert sha(meta) == t['trajectory_index_sha256']
        source[str(meta)] = sha(meta); indexed = {r['sample_id']:r for r in rows(meta)}
        for key in chosen[t['task_id']]:
            r = indexed[key]; stem = f"episode_{r['episode_index']:06d}"
            ann = meta.parent/'annotation'/f'{stem}.json'; source[str(ann)] = sha(ann)
            views = [r['video_views'][v] for v in ('main_view','left_view','right_view')]
            videos = [str(meta.parent/v['video_path']) for v in views]
            for p,v in zip(videos,views):
                assert sha(p) == v['video_sha256']; source[p] = sha(p)
            pt = ann.with_name(stem+'.progress.json')
            if pt.exists(): source[str(pt)] = sha(pt)
            else: assert t['task_id'] == 'R05'
            row = dict(key=f"Recover/{t['task_id']}/{key}", suite='Recover', task_id=t['task_id'],
                task_name=t['task_name'], sample_id=key, split='val', frames=r['total_frames'], fps=r['fps'],
                videos=videos, video_hashes=[source[p] for p in videos], annotation_path=str(ann),
                annotation_sha256=source[str(ann)], task_prompt=RECOVER_PROMPTS[t['task_id']],
                progress_path=str(pt) if pt.exists() else None, progress_sha256=source.get(str(pt)))
            assert row['key'] not in train_keys
            row['frame_indices'] = sampling_indices(row['frames'], row['fps'], 5)
            row['timestamps'] = [i/row['fps'] for i in row['frame_indices']]
            records.append(row)
    assert len(records) == 10 and set(chosen) == {'R01','R02','R03','R04','R05'}
    proofs = {}
    for name in MODELS:
        if name == 'forest':
            proof_path = FOREST/'training-status.json'; proof=read(proof_path)
            assert proof['status']=='completed' and proof['final_checkpoint_verified'] and proof['fresh_initialization_verified']
            checkpoint=Path(proof['checkpoint']); assert sha(checkpoint)==proof['checkpoint_sha256']
            proofs[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                proof=str(proof_path),proof_sha256=sha(proof_path),python=PYTHON)
        else:
            home=PROC if name=='procvlm-2b' else TRAIN
            proof_path=home/name/'formal/verification.json'; proof=read(proof_path)
            assert proof['status']=='passed' and not proof['pilot'] and proof['epoch']==1
            assert proof['optimizer_scheduler_two_rank_rng'] and proof['fresh_official_initialization']
            assert proof['selection_sha256']==sha('/cache/cvpzhr/setup/recover-six-model-caches-20261114/selection.json')
            checkpoint=Path(proof['checkpoint']); assert sha(checkpoint/'trainable-state.pt')==proof['trainable_state_sha256']
            cfg=read(checkpoint/'run_config.json')
            proofs[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=proof['trainable_state_sha256'],
                proof=str(proof_path),proof_sha256=sha(proof_path),python=cfg['python'],config=cfg)
    save(ROOT/'selection.json',dict(seed=42,per_task=2,episodes=10,selection=chosen,records=records,
        sources=source,models=list(MODELS),excluded=['vlac-2b','r2vlm-alfred-rl'],model_checkpoints=proofs,
        train_selection_sha256=sha('/cache/cvpzhr/setup/recover-six-model-caches-20261114/selection.json'),
        training_validation_disjoint=True,sample_fps=5,include_last_frame=True,created=time.time()))
    print(chosen,flush=True)


def decode(row):
    from contextlib import ExitStack
    import av
    root=IMAGES/row['task_id']/row['sample_id']; receipt=root/'verification.json'
    if receipt.exists():
        check=read(receipt); assert check['video_hashes']==row['video_hashes'] and check['frame_indices']==row['frame_indices']
        for name,digest in check['png_sha256'].items():assert sha(root/name)==digest
        return check
    root.mkdir(parents=True,exist_ok=True);wanted=set(row['frame_indices']);seen=[];hashes={}
    with ExitStack() as stack:
        videos=[stack.enter_context(av.open(p)) for p in row['videos']]
        for video in videos:
            assert video.streams.video[0].average_rate==row['fps']
            video.streams.video[0].codec_context.thread_count=1
        total=0
        for i,frames in enumerate(zip(*(v.decode(video=0) for v in videos),strict=True)):
            total=i+1
            if i not in wanted:continue
            seen.append(i)
            for v,f in enumerate(frames):
                p=root/f'{i:08d}-{v}.png';f.to_image().convert('RGB').save(p);hashes[p.name]=sha(p)
    assert total==row['frames'] and seen==row['frame_indices']
    check=dict(key=row['key'],video_hashes=row['video_hashes'],frame_indices=seen,png_sha256=hashes,decoded_frames=total,av_version=av.__version__)
    save(receipt,check);return check


def labels():
    import numpy as np
    import torch
    from experiments.prm_five_groups_20261114.recover_pour_labels import load_encoder,progress_from_deltas,REVISION,MODEL
    from mem_lite.datasets.prepare_progress import extract_deltas
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    frozen=read(ROOT/'selection.json');missing=[r for r in frozen['records'] if not r['progress_path']]
    model,processor,digest=load_encoder('cuda')
    save(ROOT/'label-initialization.json',dict(model=str(MODEL),revision=REVISION,vision_sha256=digest,
        frozen=True,batch=16,max_pixels=262144,pid=os.getpid()))
    for row in missing:
        p=ROOT/'labels'/row['task_id']/(row['sample_id']+'.progress.csv')
        assert not p.exists(), 'Inspect existing label before retry'
        raw,info=extract_deltas(row['videos'][0],model,processor,16,262144)
        assert info['frame_count']==row['frames'] and info['fps']==row['fps']
        effective,pt,events=progress_from_deltas(raw,read(row['annotation_path']))
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w') as f:
            w=csv.writer(f);w.writerow(['frame_index','delta','effective_delta','pt'])
            w.writerows((i,float(a),float(b),float(c)) for i,(a,b,c) in enumerate(zip(raw,effective,pt)))
        with p.open() as f:restored=list(csv.DictReader(f))
        assert np.array_equal(np.array([float(x['pt']) for x in restored]),pt)
        save(p.with_suffix('.json'),dict(csv_sha256=sha(p),vision_sha256=digest,revision=REVISION,
            video_sha256=row['video_hashes'][0],annotation_sha256=row['annotation_sha256'],events=events,
            policy='first-effective-pour-and-associated-putdown-cumulative-visual-change-v1'))
    save(ROOT/'labels-ready.json',dict(status='passed',supplemented_episodes=len(missing),updated=time.time()))


def finalize():
    frozen=read(ROOT/'selection.json')
    for p,digest in frozen['sources'].items():assert sha(p)==digest,p
    for row in frozen['records']:
        if not row['progress_path']:
            p=ROOT/'labels'/row['task_id']/(row['sample_id']+'.progress.csv');receipt=read(p.with_suffix('.json'))
            assert sha(p)==receipt['csv_sha256'] and receipt['video_sha256']==row['video_hashes'][0]
            row.update(progress_path=str(p),progress_sha256=sha(p))
        truth=progress_values(row);row['target_pt']=[truth[i] for i in row['frame_indices']]
        root=IMAGES/row['task_id']/row['sample_id'];receipt=read(root/'verification.json')
        assert receipt['frame_indices']==row['frame_indices']
        for name,digest in receipt['png_sha256'].items():assert sha(root/name)==digest
        row['image_receipt_sha256']=sha(root/'verification.json')
    frozen['selection_sha256']=sha(ROOT/'selection.json')
    frozen['expected_samples']=sum(len(r['frame_indices']) for r in frozen['records'])
    assert not (ROOT/'inputs.json').exists()
    save(ROOT/'inputs.json',frozen)
    save(ROOT/'preparation.json',dict(status='passed',episodes=10,samples=frozen['expected_samples'],
        inputs_sha256=sha(ROOT/'inputs.json'),original_sources_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','decode','labels','finalize']);a=p.parse_args()
    if a.action=='decode':
        with ProcessPoolExecutor(2) as pool:list(pool.map(decode,read(ROOT/'selection.json')['records']))
    else:globals()[a.action]()
