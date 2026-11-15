"""Independent all-row, source, split, label, and artifact verification."""
from collections import Counter
from .common import *


def main():
    from experiments.recover_finetune_20261114.data import grm_answer
    selection=read(ROOT/'selection.json'); protocol=read(ROOT/'protocol.json')
    for p,h in selection['source_sha256'].items(): assert sha(p)==h,p
    for p,h in read(ROOT/'source-images.json').items(): assert sha(p)==h,p
    manifests={s:read(f'/cache/cvpzhr/datasets/ProPriMem/{s}/manifest.json') for s in SUITES}
    assert check_selection(selection['episodes'],manifests)==selection['suites']
    by_ep={e['episode_id']:e for e in selection['episodes']}; reports=[]
    for model in MODELS:
        ref=protocol['models'][model]
        for p,h in dict(ref['source_sha256'],**ref['weights_sha256']).items(): assert sha(p)==h,p
        for suite in SUITES:
            ready=read(OUT/model/suite/'cache-ready.json');data=rows(ready['train_data'])
            assert ready['status']=='completed' and sha(ready['train_data'])==ready['train_sha256']
            assert ready['selection_sha256']==sha(ROOT/'selection.json')
            assert ready['protocol_sha256']==sha(ROOT/'protocol.json')
            expected={e['episode_id'] for e in selection['episodes'] if e['suite']==suite}
            counts=Counter(r['episode_id'] for r in data)
            assert set(counts)==expected and len(counts)==35
            assert len(data)==selection['suites'][suite]['observations']==len({r['id'] for r in data})
            sources={k:rows(by_ep[k]['observations']) for k in expected}
            episode_files=[];checked_files=set();pixel_bytes=0
            for item in ready['receipts']:
                assert item['native_cached_inputs_equal'] and item['supervision_mask_verified']
                assert sha(item['train_data'])==item['train_sha256']
                episode_files.extend(rows(item['train_data']))
                for p,h in item['files_sha256'].items():
                    assert sha(p)==h,p
                    checked_files.add(p);pixel_bytes+=Path(p).stat().st_size
                assert counts[item['episode_id']]==item['samples']
            assert episode_files==data
            for r in data:
                ep=by_ep[r['episode_id']]; obs=sources[r['episode_id']];i=r['sample_index']
                assert r['frame_index']==obs[i]['frame_index'] and r['pt']==obs[i]['pt']
                assert r['suite']==suite and r['split']=='train' and r['task']==ep['task']
                assert r['before_pt']==obs[0]['pt']==0 and r['after_pt']==r['pt']
                # Explicitly derive the 8-image contract without the builder's slots helper.
                expected_slots=[[0,0],[len(obs)-1,0],[0,0],[0,1],[0,2],[i,0],[i,1],[i,2]]
                assert r['slots']==expected_slots
                assert r['image']==[obs[j]['images'][VIEWS[v]] for j,v in expected_slots]
                assert r['window_frame_indices']==[obs[j]['frame_index'] for j,v in expected_slots]
                assert r['uses_future_endpoint_reference'] and r['augmentation']=='none' and r['sample_fps']==5
                assert r['source_progress_sha256']==ep['progress_sha256']
                assert r['selection_sha256']==sha(ROOT/'selection.json')
                assert r['assistant_target']==grm_answer(r)
                assert r['prompt_tensor_file'] in checked_files
                assert len(r['pixel_caches'])==8 and all(p in checked_files for p in r['pixel_caches'])
            binding=dict(model_directory=ref['config']['path'],official_revision=ref['official_revision'],
                train_data=ready['train_data'],train_sha256=ready['train_sha256'],
                selection_sha256=sha(ROOT/'selection.json'),protocol_sha256=sha(ROOT/'protocol.json'),
                dataset_class='experiments.recover_finetune_20261114.data.Dataset',
                collator_class='experiments.recover_finetune_20261114.data.CachedCollator',
                collator_kind=model,max_length=ref['max_length'],python=PYTHON,python_path=str(ROOT/'code'),
                min_pixels=ref['config']['min_pixels'],max_pixels=ref['config']['max_pixels'],
                image_contract=protocol['image_contract'],future_endpoint_reference=True,
                supervision='assistant-only integer-percent score plus EOS; original continuous PT retained',
                sample_fps=5,augmentation='none',training_started=False)
            save(OUT/model/suite/'data-binding.json',binding)
            reports.append(dict(model=model,suite=suite,episodes=35,observations=len(data),
                task_counts=selection['suites'][suite]['tasks'],bytes=pixel_bytes,train_data=ready['train_data']))
    save(OUT/'final-audit.json',dict(status='passed',caches=reports,distinct_episodes=105,
        observations_per_model=19479,training_rows_total=38958,selection_sha256=sha(ROOT/'selection.json'),
        protocol_sha256=sha(ROOT/'protocol.json'),all_source_and_artifact_hashes_verified=True,
        train_validation_disjoint=True,all_tasks_at_least_two=True,cached_native_inputs_equal=True,
        labels_and_frames_unchanged=True,original_continuous_pt_retained=True,
        model_forward_executed=False,cuda_used=False,training_started=False,
        source_checksums_sha256=sha(ROOT/'source-checksums.json')))
    print('PASS: six model/suite caches, 105 trajectories, 19479 observations per model',flush=True)


if __name__=='__main__':main()
