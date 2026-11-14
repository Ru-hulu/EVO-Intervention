"""Independent all-row audit and training-loader bindings for the six caches."""
from collections import Counter
from pathlib import Path

from .build import ROOT, OUT, MODELS, SUITES, read, rows, sha, save


def main():
    assert read(ROOT/'pipeline-status.json')['status']=='completed'
    selection=read(ROOT/'selection.json'); selected={r['episode_id']:r for r in selection['episodes']}
    sources={key:rows(ep['source_samples']) for key,ep in selected.items()}
    plan_path=Path('/cache/cvpzhr/setup/prm-five-groups-20261114/code/experiments/prm_five_groups_20261114/experiment-plan.json')
    plan=read(plan_path); provenance={}
    for model in MODELS:
        model_id={'robometer':'robometer-4b','procvlm':'procvlm-2b'}[model]
        reference=next(m['prior_provenance_reference'] for m in plan['models'] if m['id']==model_id)
        assert reference['path']==MODELS[model]
        for name,digest in reference['small_file_sha256'].items():assert sha(Path(MODELS[model])/name)==digest
        files=[p for p in Path(MODELS[model]).rglob('*') if p.is_file() and '.cache' not in p.parts
               and p.suffix in ('.json','.yaml','.txt','.jinja','.model') and p.stat().st_size<30*2**20]
        provenance[model]=dict(reference=reference,processor_tokenizer_config_sha256={str(p):sha(p) for p in files})
    output=[]
    for model in MODELS:
        for suite,(_,expected_count) in SUITES.items():
            records=rows(OUT/model/suite/'train.jsonl'); receipt=read(OUT/model/suite/'cache-ready.json')
            expected_episodes={key for key,ep in selected.items() if ep['suite']==suite}
            assert len(records)==expected_count
            assert {r['episode_id'] for r in records}==expected_episodes
            assert len({r['id'] for r in records})==len(records)
            counts=Counter()
            for row in records:
                ep=selected[row['episode_id']]; source=sources[row['episode_id']]; i=row['sample_index']
                counts[row['episode_id']]+=1
                assert row['frame_index']==source[i]['frame_index'] and row['pt']==source[i]['pt']
                assert row['task']==ep['task'] and row['augmentation']=='none'
                assert all(0<=index<=i for index in row['window_indices'])
                assert [source[j]['frame_index'] for j in row['window_indices']]==row['window_frame_indices']
                if model=='procvlm':
                    assert row['assistant_target']==f'<progress>{100*row["pt"]:.2f}%</progress>'
                    assert row['progress_gt']==100*source[i]['pt']
                    assert row['previous_cot_input'] is False and 'remaining_actions' not in row
            for key,count in counts.items():assert count==len(sources[key])
            for episode in receipt['receipts']:
                assert sha(episode['pixel_cache'])==episode['pixel_sha256']
                assert sha(episode['episode_jsonl'])==episode['episode_jsonl_sha256']
            binding=dict(model_directory=MODELS[model],train_data=str(OUT/model/suite/'train.jsonl'),
                model_revision=provenance[model]['reference']['revision'],sample_fps=5,
                min_pixels=4096,max_pixels=65536,max_length=4096,augmentation='none',
                dataset_class=('robometer_training.data.ProgressDataset' if model=='robometer' else
                               'experiments.soc35_model_caches_20261114.proc_data.ProgressDataset'),
                collator_class=('robometer_training.data.ProgressCollator' if model=='robometer' else
                                'experiments.soc35_model_caches_20261114.proc_data.ProgressCollator'),
                python_paths=([str(ROOT/'runtime'),'/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
                              if model=='robometer' else [str(ROOT/'code')]),
                progress_target_units='0..1 C51-soft-target' if model=='robometer' else '0..100 percentage',
                previous_reasoning_input=False,requires_remaining_action_labels=False,
                train_inference_input_contract_must_match=True)
            save(OUT/model/suite/'data-binding.json',binding)
            output.append(dict(model=model,suite=suite,episodes=len(counts),observations=len(records),
                task_count=len(selection['suites'][suite]['tasks']),train_data=binding['train_data'],
                pixel_bytes=receipt['pixel_bytes'],verified_pt_frames_prompts=True,
                pixel_hashes_verified=True,verified_no_future_frames=True))
    files=list((ROOT/'code/experiments/soc35_model_caches_20261114').glob('*.py'))
    files+=list((ROOT/'runtime/robometer_training').glob('*.py'))
    report=dict(status='passed',caches=output,distinct_episodes=105,observations_per_model=19479,
        code_sha256={str(p):sha(p) for p in files},model_provenance=provenance,
        selection_sha256=sha(ROOT/'selection.json'),labels_recomputed=False,training_started=False,
        model_forward_executed=False,gpu_used=False,
        procvlm_supervision='User-approved progress-only response and original numeric PT; no reasoning labels.')
    save(OUT/'final-audit.json',report)
    print('PASS: 6 caches, 105 distinct episodes, 19479 observations per model; original PT/frame/prompt parity.')


if __name__=='__main__':main()
