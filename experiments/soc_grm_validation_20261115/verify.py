"""Independent all-row audit and CPU native-processor equivalence checks."""
import importlib.util
import sys
import time
from collections import Counter
from .common import *


def verify():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    inputs = read(DATA/'inputs.json'); by_key = {r['key']:r for r in inputs['episodes']}
    assert len(by_key) == 84 and len(inputs['registry']) == len(inputs['caches']) == 6
    assert Counter(r['suite'] for r in by_key.values()) == EXPECTED
    assert inputs['source_inputs_sha256'] == sha(SOURCE) and inputs['selection_sha256'] == sha(SELECTION)
    assert inputs['episodes'] == read(SOURCE)['episodes']
    assert not inputs['evaluation_started'] and not inputs['automatic_evaluation_enabled']
    for p,h in {**inputs['protected_files'],**inputs['artifacts_sha256']}.items(): assert sha(p) == h,p
    for p,h in read(DATA/'source-images.json').items(): assert sha(p) == h,p
    labels = {r['key']:r for r in read(DATA/'labels.json')}
    assert set(labels) == set(by_key)
    train_keys = {r['episode_id'] for r in read(TRAIN_SELECTION)['episodes']}
    assert not train_keys & set(by_key)
    for key,row in by_key.items():
        for field in ('suite','frame_indices','timestamps','target_pt'): assert labels[key][field] == row[field]
    sys.path.insert(0,str(NATIVE))
    spec = importlib.util.spec_from_file_location('soc84_grm_native_verifier',NATIVE/'worker.py')
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    summaries = []; checks = []
    for model in MODELS:
        config = read(TRAIN_PROTOCOL)['models'][model]['config']
        assert native.MODELS[model] == config
        processor = AutoProcessor.from_pretrained(config['path'],local_files_only=True)
        options = {k:config[k] for k in ('min_pixels','max_pixels')}
        for k,v in options.items(): setattr(processor.image_processor,k,v)
        shell = object.__new__(native.Generative); shell.kind = 'grm'; shell.processor = processor
        shell.template = (NATIVE/'grm_prompt.txt').read_text()
        for suite in SUITES:
            ckkey = f'{model}-{suite}'; info = inputs['registry'][ckkey]
            binding = read(DATA/model/suite/'data-binding.json')
            for field in ('model','suite','checkpoint','checkpoint_sha256','official_revision','processor_config'):
                assert binding[field] == info[field]
            assert info['suite'] == suite and info['model'] == model
            assert sha(Path(info['checkpoint'])/'trainable-state.pt') == info['checkpoint_sha256']
            assert binding['labels_sha256'] == sha(DATA/'labels.json') and binding['selection_sha256'] == sha(SELECTION)
            data = rows(binding['input_path']); count = Counter(r['key'] for r in data)
            selected = {k:r for k,r in by_key.items() if r['suite'] == suite}
            assert set(count) == set(selected) and len(count) == EXPECTED[suite]
            assert len(data) == binding['observations'] == sum(len(r['frame_indices']) for r in selected.values())
            assert len({r['id'] for r in data}) == len(data)
            native_episodes = {}
            for key,row in selected.items():
                fs = [dict(image=[str(Path(row['image_root'])/f'{f:08d}-{v}.png') if p else None
                                  for v,p in enumerate(row['videos'])]) for f in row['frame_indices']]
                native_episodes[key] = dict(task=row['task_prompt'],frames=fs)
            checked_tasks = set(); max_length = 0
            for r in data:
                row = selected[r['key']]; i = r['sample_index']; n = len(row['frame_indices'])
                assert count[r['key']] == n and r['checkpoint_key'] == ckkey
                assert r['suite'] == suite and r['task_id'] == row['task_id']
                assert r['id'] == f"{row['key']}/{i:06d}"
                assert r['frame_index'] == row['frame_indices'][i] and r['timestamp'] == row['timestamps'][i]
                # This whitelist is the inference boundary: no PT, answer, or training target.
                assert set(r['inference']) == {'task','prompt_path','images'}
                request = r['inference']; ep = native_episodes[r['key']]
                assert request['task'] == row['task_prompt'] and len(request['images']) == 8
                assert request['images'] == native.images_for('grm',ep,i)
                assert sha(request['prompt_path']) == inputs['artifacts_sha256'][request['prompt_path']]
                prompt = read(request['prompt_path']); assert prompt['task'] == row['task_prompt']
                # One middle frame per task plus first/final frames of each suite,
                # including the existing absent-camera cases. No language model is loaded.
                chosen = (row['task_id'] not in checked_tasks and i == n//2) or (
                    row['key'] == next(iter(selected)) and i in (0,n-1))
                if chosen:
                    online = shell.request((ep,i,None))
                    assert online['prompt'] == prompt['prompt']
                    direct = dict(processor(text=[online['prompt']],images=online['multi_modal_data']['image'],
                        return_tensors='pt',**options))
                    cached = dict(processor(text=[prompt['prompt']],images=[native.image(p) for p in request['images']],
                        return_tensors='pt',**options))
                    assert set(direct) == set(cached)
                    assert all(torch.equal(direct[k],cached[k]) for k in direct)
                    assert direct['image_grid_thw'].shape[0] == 8
                    length = direct['input_ids'].shape[1]; assert length < 4096
                    assert all(torch.isfinite(v).all() for v in direct.values())
                    checks.append(dict(model=model,suite=suite,key=row['key'],sample_index=i,
                        native_tensors_equal=True,prompt_tokens=length,missing_view_slots=sum(p is None for p in request['images'])))
                    max_length = max(max_length,length); checked_tasks.add(row['task_id'])
            assert checked_tasks == {r['task_id'] for r in selected.values()}
            summaries.append(dict(model=model,suite=suite,episodes=len(count),observations=len(data),
                task_counts=dict(Counter(r['task_id'] for r in selected.values())),checkpoint=info['checkpoint'],
                checkpoint_sha256=info['checkpoint_sha256'],max_checked_prompt_tokens=max_length))
            print('verified',model,suite,len(count),len(data),flush=True)
    assert not torch.cuda.is_initialized()
    receipt = dict(status='passed',caches=summaries,distinct_trajectories=84,model_trajectories=168,
        observations_per_model=sum(len(r['frame_indices']) for r in by_key.values()),
        inputs_sha256=sha(DATA/'inputs.json'),cpu_native_processor_checks=checks,
        train_validation_disjoint=True,all_task_counts_two=True,all_source_and_png_hashes_verified=True,
        final_checkpoint_and_adapter_tensors_verified=True,labels_and_sampling_unchanged=True,
        future_endpoint_reference=True,model_forward_executed=False,gpu_load_verified=False,
        gpu_used=False,evaluation_started=False,automatic_evaluation_enabled=False)
    save(DATA/'verification.json',receipt); save(ROOT/'verification.json',receipt)
    save(ROOT/'status.json',dict(status='prepared_and_verified_not_started',models=2,suites=3,caches=6,
        distinct_trajectories=84,observations_per_model=receipt['observations_per_model'],
        inputs_sha256=receipt['inputs_sha256'],evaluation_started=False,updated=time.time()))
    return {k:v for k,v in receipt.items() if k not in ('cpu_native_processor_checks','caches')}


if __name__ == '__main__':
    try: print(json.dumps(verify()))
    except BaseException:
        import traceback
        save(ROOT/'status.json',dict(status='verification_failed',evaluation_started=False,error=traceback.format_exc()))
        raise
