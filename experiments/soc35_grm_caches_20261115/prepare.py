"""Bind existing 35-episode subsets to current train splits; do not resample."""
import csv
from .common import *


def main():
    assert not (ROOT/'selection.json').exists(), 'Never replace a frozen selection'
    original = read(SOURCE)
    manifests = {s: read(f'/cache/cvpzhr/datasets/ProPriMem/{s}/manifest.json') for s in SUITES}
    counts = check_selection(original['episodes'], manifests)
    hashes = {str(SOURCE): sha(SOURCE)}
    for s in SUITES:
        p = Path(f'/cache/cvpzhr/datasets/ProPriMem/{s}/manifest.json')
        hashes[str(p)] = sha(p)
    images = {}; metadata_label_notes = []
    for ep in original['episodes']:
        summary_path = Path(ep['source_samples']).parent/'summary.json'
        summary = read(summary_path)
        for p, expected in [(ep['source_samples'], ep['source_samples_sha256']),
                            (ep['observations'], ep['observations_sha256']),
                            (ep['progress_path'], ep['progress_sha256']),
                            (summary_path, ep['source_summary_sha256'])]:
            assert sha(p) == expected, str(p)
            hashes[str(p)] = expected
        info = summary['source']; assert info['split'] == 'train'
        assert summary['history_source']['sample_fps'] == 5
        assert summary['history_source']['include_episode_last_frame']
        task = next(t for t in manifests[ep['suite']]['tasks'] if t['task_id'] == ep['task_id'])
        meta = Path(f'/cache/cvpzhr/datasets/ProPriMem/{ep["suite"]}')/task['trajectory_index']
        assert sha(meta) == task['trajectory_index_sha256']
        hashes[str(meta)] = sha(meta)
        current = next(r for r in rows(meta) if r['sample_id'] == ep['sample_id'])
        assert current.get('training_eligible', True)
        # Some downloaded metadata still describes time-ratio labels. The frozen
        # verified visual-PT sidecar, not this stale description, binds supervision.
        label_sidecar = Path(ep['progress_path']).with_suffix('.json')
        label = read(label_sidecar)
        assert label['policy'] == 'whole-video-cumulative-visual-delta-v1'
        assert label['csv_sha256'] == ep['progress_sha256']
        assert label['video_sha256'] == info['video_hashes'][0]
        assert label['frame_count'] == info['frames'] == current['total_frames']
        assert label['protocol_sha256'] == sha('/cache/cvpzhr/setup/proprimem-visual-pt-20261113/protocol.json')
        hashes[str(label_sidecar)] = sha(label_sidecar)
        if 'progress_path' in current:
            assert (meta.parent/current['progress_path']).resolve() == Path(ep['progress_path']).resolve()
            assert current['progress_sha256'] == ep['progress_sha256']
        else:
            metadata_label_notes.append(dict(episode_id=ep['episode_id'],metadata_policy=current.get('progress_policy'),
                 authoritative_visual_pt=str(label_sidecar),csv_sha256=ep['progress_sha256']))
        current_views = current['video_views']
        view_names = ['main_view','wrist_view',None] if 'wrist_view' in current_views else ['main_view','left_view','right_view']
        for v, p, h in zip(view_names, info['videos'], info['video_hashes']):
            if p:
                assert (meta.parent/current_views[v]['video_path']).resolve() == Path(p).resolve()
                assert current_views[v]['video_sha256'] == h
        for p, digest in zip(info['videos'], info['video_hashes']):
            if p:
                assert sha(p) == digest, p
                hashes[p] = digest
        observed = rows(ep['observations']); source = rows(ep['source_samples'])
        assert len(observed) == len(source) == ep['samples']
        with open(ep['progress_path']) as f: truth = {int(r['frame_index']): float(r['pt']) for r in csv.DictReader(f)}
        assert observed[0]['frame_index'] == 0 and observed[-1]['frame_index'] == info['frames']-1
        assert observed[0]['pt'] == 0 and observed[-1]['pt'] == 1
        image_digest = hashlib.sha256()
        for i, (a, b) in enumerate(zip(observed, source)):
            assert a['sample_index'] == i and a['frame_index'] == b['frame_index']
            assert a['pt'] == b['pt'] == truth[a['frame_index']] and 0 <= a['pt'] <= 1
            if i: assert a['frame_index'] > observed[i-1]['frame_index']
            for v in VIEWS:
                p = a['images'][v]
                if p not in images: images[p] = sha(p)
                if v in a['missing_views']:
                    assert b['images'][v] is None
                    image_digest.update(f'{i}/{v}:missing\n'.encode())
                else:
                    assert os.path.samefile(Path(ep['source_samples']).parent/b['images'][v], p)
                    image_digest.update(f'{i}/{v}:{images[p]}\n'.encode())
        assert image_digest.hexdigest() == ep['source_image_sequence_sha256']
    save(ROOT/'source-images.json', images)
    hashes['/cache/cvpzhr/setup/proprimem-visual-pt-20261113/protocol.json'] = sha('/cache/cvpzhr/setup/proprimem-visual-pt-20261113/protocol.json')
    hashes[str(ROOT/'source-images.json')] = sha(ROOT/'source-images.json')
    save(ROOT/'selection.json', dict(schema='soc35-shared-grm-training-v1', suites=counts,
         episodes=original['episodes'], source_selection=str(SOURCE), source_selection_sha256=sha(SOURCE),
         selection_policy='Reuse exact prior FoREST/Robometer/ProcVLM 35-per-suite selections',
         source_sha256=hashes, sample_fps=5, include_last_frame=True, augmentation='none',
         validation_overlap=0, minimum_per_task=2, training_started=False,
         stale_metadata_label_notes=metadata_label_notes))
    native_files = [OLD/n for n in ('worker.py', 'common.py', 'grm_prompt.txt')]
    provenance = {}
    for model in MODELS:
        previous = read(f'/cache/cvpzhr/setup/recover-six-model-caches-20261114/{model}-native-provenance.json')
        config = read(RECOVER/model/'config.json')
        assert config['model_path'] == previous['config']['path']
        assert config['official_revision'] == previous['official_revision']
        for p, h in previous['source_sha256'].items(): assert sha(p) == h, p
        for p, h in config['weights_sha256'].items(): assert sha(p) == h, p
        model_files = [p for p in Path(config['model_path']).iterdir()
                       if p.suffix in ('.json', '.txt', '.jinja', '.model') and p.is_file()]
        provenance[model] = dict(config=previous['config'], official_revision=previous['official_revision'],
             weights_sha256=config['weights_sha256'], source_sha256={str(p):sha(p) for p in native_files+model_files},
             max_length=config['max_length'], supervision=config['supervision'])
    save(ROOT/'protocol.json', dict(models=provenance, sample_fps=5, views=VIEWS,
         image_contract='first-main,last-main,initial-three-views,current-three-views',
         future_endpoint_reference=True, labels='unchanged visual cumulative PT',
         missing_views='inherited black RGB64 placeholder; never repair missing frames',
         cache_dtype='bfloat16 pixels, native integer token/grid tensors', cpu_only=True,
         model_forward=False, training_started=False))
    save(ROOT/'initial-resources.json', resource_check())
    print(json.dumps(counts), flush=True)


if __name__ == '__main__': main()
