import random
import time
from .common import *
from experiments.prm_five_groups_20261114.prepare_cache import progress_values
from experiments.prm_five_groups_20261114.eval_common import sampling_indices


def choose(task, train, seed=42):
    eligible = sorted(task['val'])
    assert eligible and not set(eligible).intersection(task['train'])
    assert not set(eligible).intersection(train)
    return random.Random(seed).choice(eligible)


def main():
    verify_code(); assert not (ROOT/'selection.json').exists(), 'Do not replace frozen example'
    train = read(TRAIN_SELECTION); data = Path(train['source_dataset'])
    manifest = read(data/'manifest.json'); task = next(t for t in manifest['tasks'] if t['task_id'] == 'R03')
    key = choose(task, train['selection']['R03']); meta = data/task['trajectory_index']
    assert sha(meta) == task['trajectory_index_sha256']
    r = next(r for r in rows(meta) if r['sample_id'] == key); stem = f"episode_{r['episode_index']:06d}"
    ann = meta.parent/'annotation'/f'{stem}.json'; pt = ann.with_name(stem+'.progress.json')
    views = [r['video_views'][v] for v in ('main_view','left_view','right_view')]
    videos = [str(meta.parent/v['video_path']) for v in views]
    sources = {str(p):sha(p) for p in [data/'manifest.json',meta,ann,pt]}
    for path, v in zip(videos, views):
        sources[path] = sha(path); assert sources[path] == v['video_sha256']
    assert read(pt)['stage_annotation_sha256'] == sha(ann)
    prompt = train['records'][0]['task_prompt']
    row = dict(key=f'Recover/R03/{key}', suite='Recover', task_id='R03', task_name=task['task_name'],
        sample_id=key, split='val', frames=r['total_frames'], fps=r['fps'], videos=videos,
        video_hashes=[sources[p] for p in videos], annotation_path=str(ann), annotation_sha256=sha(ann),
        progress_path=str(pt), progress_sha256=sha(pt), task_prompt=prompt)
    row['frame_indices'] = sampling_indices(row['frames'], row['fps'], 5)
    row['timestamps'] = [i/row['fps'] for i in row['frame_indices']]
    truth = progress_values(row); row['target_pt'] = [truth[i] for i in row['frame_indices']]
    from experiments.recover_validation_20261114 import prepare as old
    old.IMAGES = IMAGES; decoded = old.decode(row)
    row['image_receipt_sha256'] = sha(IMAGES/'R03'/key/'verification.json')
    save(ROOT/'selection.json', dict(seed=42,selection_method='Random choice from sorted full R03 validation split before prediction',
        candidates=task['val'], records=[row], sources=sources, models=MODELS, sample_fps=5,
        expected_samples=len(row['frame_indices']), include_last_frame=True, training_validation_disjoint=True,
        train_selection_sha256=sha(TRAIN_SELECTION), selected_without_model_predictions=True, created=time.time()))
    save(ROOT/'preparation.json', dict(status='passed', key=row['key'], samples=len(row['frame_indices']),
        original_frames=row['frames'], png_count=len(decoded['png_sha256']), selection_sha256=sha(ROOT/'selection.json')))
    print(key, len(row['frame_indices']), 'observations; held-out and decoded', flush=True)


def bind_final_weights():
    assert read(TRAIN/'status.json')['status'] == 'completed', 'Wait for all five training jobs'
    data = read(ROOT/'selection.json'); checkpoints = {}
    for path, digest in data['sources'].items(): assert sha(path) == digest, path
    for name in MODELS:
        home = TRAIN/name; proof_path = home/'formal/verification.json'; proof = read(proof_path)
        cfg = read(home/'config.json'); validate_proof(name, proof, cfg)
        assert read(home/'formal/exit.json')['returncode'] == 0
        p = Path(proof['checkpoint']); digest = proof.get('checkpoint_sha256', proof.get('trainable_state_sha256'))
        assert sha(p if name == 'forest' else p/'trainable-state.pt') == digest
        checkpoints[name] = dict(checkpoint=str(p), checkpoint_sha256=digest, config=cfg,
            proof=str(proof_path), proof_sha256=sha(proof_path), python=cfg['python'])
    assert not (ROOT/'inputs.json').exists(), 'Do not replace checkpoint binding'
    data.update(model_checkpoints=checkpoints, selection_sha256=sha(ROOT/'selection.json'))
    save(ROOT/'inputs.json', data)
    save(ROOT/'training-final-verification.json', dict(status='passed', models=MODELS,
        all_five_final_checkpoints_verified=True, inputs_sha256=sha(ROOT/'inputs.json'), updated=time.time()))


if __name__ == '__main__': main()
