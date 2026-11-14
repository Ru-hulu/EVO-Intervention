"""Freeze two train episodes per Recover task and decode immutable 5 FPS RGB."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

from .common import *
from experiments.prm_five_groups_20261114.prepare_cache import RECOVER_PROMPTS, progress_values


def freeze():
    assert not (ROOT/'selection.json').exists(), 'Existing selection must not be resampled'
    ROOT.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(DATA/'manifest.json'); chosen = select(manifest['tasks'])
    assert set(chosen) == {'R01','R02','R03','R04','R05'}
    sources = {str(DATA/'manifest.json'):sha(DATA/'manifest.json')}; records=[]
    for task in manifest['tasks']:
        meta = DATA/task['trajectory_index']; sources[str(meta)] = sha(meta)
        assert sources[str(meta)] == task['trajectory_index_sha256']
        indexed = {r['sample_id']:r for r in rows(meta)}
        for key in chosen[task['task_id']]:
            r = indexed[key]; episode = f"episode_{r['episode_index']:06d}"
            views = [r['video_views'][v] for v in ['main_view','left_view','right_view']]
            videos = [str(meta.parent/v['video_path']) for v in views]
            for path, view in zip(videos, views):
                sources[path] = sha(path); assert sources[path] == view['video_sha256']
            annotation = meta.parent/(r.get('annotation_path') or f'annotation/{episode}.json')
            sources[str(annotation)] = sha(annotation)
            progress = meta.parent/f'annotation/{episode}.progress.json'
            if progress.exists():
                sources[str(progress)] = sha(progress)
            else:
                assert task['task_id'] == 'R05'
            rec = dict(key=f"Recover/{task['task_id']}/{key}",suite='Recover',task_id=task['task_id'],
                task_name=task['task_name'],sample_id=key,split='train',frames=r['total_frames'],fps=r['fps'],
                videos=videos,video_hashes=[sources[v] for v in videos],annotation_path=str(annotation),
                annotation_sha256=sources[str(annotation)],task_prompt=RECOVER_PROMPTS[task['task_id']],
                progress_path=str(progress) if progress.exists() else None,
                progress_sha256=sources.get(str(progress)),generate_missing_progress=not progress.exists())
            if rec['progress_path']: progress_values(rec)
            records.append(rec)
    frozen = dict(seed=42,episodes=10,tasks=5,per_task=2,sample_fps=5,
        include_episode_last_frame=True,selection=chosen,records=records,source_sha256=sources,
        available_train=sum(len(t['train']) for t in manifest['tasks']),
        excluded_validation=sum(len(t['val']) for t in manifest['tasks']),models=MODELS,
        user_request='Recovery在训练集中，每个任务选择2条轨迹作为训练轨迹。为6个模型生成各自的训练缓存。',
        training_started=False,desktop_automation_created=False,created=time.time())
    source_checks(frozen); save(ROOT/'selection.json',frozen)
    save(ROOT/'cache-contract.json',dict(
        scope='Model-specific input/pixel/history caches plus lossless original PT supervision; no training launch',
        raw_pt_never_rounded=True,rollback_and_plateau_preserved=True,
        grm_future_endpoint=True,proc_previous_cot_input=False,
        proc_remaining_action_answers='Not fabricated; response/loss specification is separate from this input cache',
        vlac_supervision='Store before_pt/after_pt unchanged. No absolute-PT-as-critic target and no clipping.',
        vlac_critic_response_target='Not assigned by cache preparation',
        generative_sft_answer_targets='Deferred until fine-tuning objective is specified',
        forest='Frozen official Qwen3.5 vision+CLIP/max_depth native streaming memory',
        sample_fps=5,original_dataset_modified=False))
    print(json.dumps({'episodes':10,'selection':chosen,'native_frames':sum(r['frames'] for r in records)}),flush=True)


def decode(row):
    from mem_lite.datasets.prepare_stream import iter_observations
    target=epdir(row); assert not target.exists(),f'Preserve existing output: {target}'
    target.mkdir(parents=True)
    for path,digest in zip(row['videos'],row['video_hashes']): assert sha(path)==digest
    records=[]; hashes={}
    # Decode positions/images only; zeros never become a supervision label.
    for i, observation in enumerate(iter_observations(row['videos'],dict.fromkeys(range(row['frames']),0.),
                                                      sample_fps=5,end_frame=row['frames'])):
        paths=[]
        for v,image in enumerate(observation['images']):
            assert image is not None
            p=target/f'{observation["frame_index"]:08d}-{v}.png';image.save(p)
            paths.append(str(p));hashes[p.name]=sha(p)
        records.append(dict(sample_index=i,frame_index=observation['frame_index'],images=paths))
    assert records[0]['frame_index']==0 and records[-1]['frame_index']==row['frames']-1
    write_rows(target/'observations.jsonl',records)
    receipt=dict(key=row['key'],samples=len(records),decoded_views=3,png_sha256=hashes,
        observations_sha256=sha(target/'observations.jsonl'),video_hashes=row['video_hashes'],labels_present=False)
    save(target/'decode-verification.json',receipt)
    return {k:v for k,v in receipt.items() if k!='png_sha256'}


def main():
    frozen=read(ROOT/'selection.json');source_checks(frozen)
    with ProcessPoolExecutor(2) as pool:
        receipts=list(pool.map(decode,frozen['records']))
    save(ROOT/'decode-verification.json',dict(status='passed',episodes=10,
        samples=sum(r['samples'] for r in receipts),records=receipts,selection_sha256=sha(ROOT/'selection.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','decode']);args=p.parse_args()
    freeze() if args.action=='freeze' else main()
