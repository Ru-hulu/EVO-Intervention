"""Prepare shared RGB references and model-native input manifests; never infer."""
import csv
import fcntl
import math
import time
from collections import Counter
from .common import *


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'prepare.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(ROOT/'status.json', dict(status='preparing', evaluation_started=False, updated=time.time()))
    code = read(ROOT/'source-checksums.json')
    for p,h in code['files'].items(): assert sha(ROOT/'code'/p) == h, p
    source, selection, training = read(SOURCE), read(SELECTION), read(TRAIN_SELECTION)
    episodes = source['episodes']; assert len(episodes) == len({e['key'] for e in episodes}) == 84
    assert Counter(e['suite'] for e in episodes) == EXPECTED
    assert {e['key'] for e in episodes} == {e['key'] for e in selection['episodes']}
    assert not {e['key'] for e in episodes} & {e['episode_id'] for e in training['episodes']}
    assert set(Counter((e['suite'],e['task_id']) for e in episodes).values()) == {2}
    protected = {}; resolved = {}
    def protect(p, h=None):
        h = h or sha(p); actual = resolve(p, h)
        protected[str(actual)] = h; resolved[str(p)] = dict(path=str(actual), sha256=h)
        return actual
    for p in (SOURCE, SELECTION, TRAIN_SELECTION, TRAIN_PROTOCOL): protect(p)
    for p,h in selection['source_hashes'].items(): protect(p,h)
    manifests = {s:read(resolved[f'/cache/cvpzhr/datasets/ProPriMem/{s}/manifest.json']['path']) for s in SUITES}
    images, labels = {}, []
    for j, row in enumerate(episodes):
        assert row['split'] == 'val' and row['input_fps'] == 5 and not row['input_issues']
        t = next(t for t in manifests[row['suite']]['tasks'] if t['task_id'] == row['task_id'])
        assert row['sample_id'] in t['val'] and row['sample_id'] not in t['train']
        assert not set(t['train']) & set(t['val'])
        for p,h in zip(row['videos'], row['video_hashes']):
            if p: protect(p,h)
        pt = protect(row['progress_path'], row['progress_sha256'])
        with pt.open() as f: truth = list(csv.DictReader(f))
        assert [int(x['frame_index']) for x in truth] == list(range(row['frames']))
        values = [float(x['pt']) for x in truth]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in values)
        assert values[0] == 0 and values[-1] == 1
        assert row['target_pt'] == [values[f] for f in row['frame_indices']]
        assert row['frame_indices'][0] == 0 and row['frame_indices'][-1] == row['frames']-1
        assert row['frame_indices'] == sorted(set(row['frame_indices']))
        assert row['timestamps'] == [f/row['fps'] for f in row['frame_indices']]
        cache = Path(row['image_root']); proof = read(cache/'verification.json')
        protect(cache/'verification.json')
        assert proof['input'] == dict(video_hashes=row['video_hashes'], frame_indices=row['frame_indices'])
        assert proof['lossless'] and proof['decoded_native_frames'] == row['frames']
        expected = {f'{f:08d}-{v}.png' for f in row['frame_indices'] for v,p in enumerate(row['videos']) if p}
        assert set(proof['png_sha256']) == expected
        for name,h in proof['png_sha256'].items():
            p = cache/name; assert sha(p) == h; images[str(p)] = h
        labels.append(dict(key=row['key'],suite=row['suite'],frame_indices=row['frame_indices'],
            timestamps=row['timestamps'],target_pt=row['target_pt'],progress_path=str(pt),
            progress_sha256=row['progress_sha256']))
        save(ROOT/'status.json',dict(status='verifying_sources',verified_episodes=j+1,evaluation_started=False,updated=time.time()))
    artifacts = {}
    artifacts[str(DATA/'labels.json')] = freeze(DATA/'labels.json', labels)
    artifacts[str(DATA/'source-images.json')] = freeze(DATA/'source-images.json', images)
    protocol = read(TRAIN_PROTOCOL); registry = {}; caches = []
    for model in MODELS:
        native = protocol['models'][model]
        for p,h in {**native['source_sha256'],**native['weights_sha256']}.items(): protect(p,h)
        cfg_native = native['config']; processor = AutoProcessor.from_pretrained(cfg_native['path'], local_files_only=True)
        prompts = {}
        for row in episodes:
            key = f"{row['suite']}-{row['task_id']}"
            parts = (NATIVE/'grm_prompt.txt').read_text().format(task=row['task_prompt']).split('<image>')
            assert len(parts) == 9
            content = []
            for i, part in enumerate(parts):
                content.append(dict(type='text',text=part))
                if i < 8: content.append(dict(type='image'))
            prompt = processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            path = DATA/model/'prompts'/f'{key}.json'
            artifacts[str(path)] = freeze(path,dict(task=row['task_prompt'],prompt=prompt))
            prompts[key] = str(path)
        for suite in SUITES:
            folder = TRAIN/model/suite/'formal'; vp = folder/'verification.json'; proof = read(vp)
            ck = Path(proof['checkpoint']); cfg = read(ck/'run_config.json')
            check_checkpoint_metadata(model,suite,proof,cfg,sha(TRAIN_SELECTION))
            assert cfg['official_revision'] == native['official_revision'] and cfg['model_path'] == cfg_native['path']
            assert cfg['processor_config'] == {k:cfg_native[k] for k in ('min_pixels','max_pixels')}
            protect(vp); protect(ck/'run_config.json'); protect(ck/'trainable-state.pt',proof['trainable_state_sha256'])
            for name in ('trainable-state.json','adapter_model.safetensors','adapter_config.json','trainer_state.json'):
                protect(ck/name)
            for name in ('optimizer.pt','scheduler.pt','rng_state_0.pth','rng_state_1.pth'):
                assert (ck/name).stat().st_size > 0
            protect(cfg['train_data'],cfg['train_sha256'])
            train_rows = rows(cfg['train_data']); keys = {r['episode_id'] for r in train_rows}
            assert len(keys) == 35 and len(train_rows) == cfg['samples']
            assert keys == {r['episode_id'] for r in training['episodes'] if r['suite'] == suite}
            assert not keys & {r['key'] for r in episodes}
            state = torch.load(ck/'trainable-state.pt',map_location='cpu',weights_only=True)
            state_digest = tensor_digest(state.items()); meta = read(ck/'trainable-state.json')
            assert set(state) == set(meta['parameters']) and state_digest == meta['sha256']
            assert all(torch.isfinite(t).all() for t in state.values())
            adapter = load_file(str(ck/'adapter_model.safetensors'),device='cpu')
            renamed = {k.replace('.default.','.'):v for k,v in state.items()}
            assert set(renamed) == set(adapter) and all(torch.equal(v,adapter[k]) for k,v in renamed.items())
            ac = read(ck/'adapter_config.json')
            assert ac['base_model_name_or_path'] == cfg['model_path'] and ac['r'] == 16 and ac['lora_alpha'] == 32
            initial = []
            for rank in (0,1):
                ip = folder/f'initialization-rank{rank}.json'; cp = folder/f'completion-rank{rank}.json'
                init, done = read(ip), read(cp); protect(ip); protect(cp)
                assert init['suite'] == suite and init['official_revision'] == cfg['official_revision']
                assert init['initial_adapter'] is None and init['fresh_lora_B_all_zero'] and init['optimizer_reset']
                assert init['selection_sha256'] == cfg['selection_sha256']
                assert done['status'] == 'completed' and done['epoch'] == 1 and not done['pilot']
                assert done['model'] == model and done['checkpoint'] == str(ck) and done['steps'] == proof['steps']
                assert done['final_trainable_sha256'] == state_digest != init['initial_trainable_sha256']
                initial.append(init['initial_trainable_sha256'])
            assert initial[0] == initial[1]
            trainer = read(ck/'trainer_state.json')
            assert trainer['epoch'] == 1 and trainer['global_step'] == proof['steps']
            del state, adapter, renamed
            key = f'{model}-{suite}'
            registry[key] = dict(model=model,suite=suite,checkpoint=str(ck),checkpoint_sha256=proof['trainable_state_sha256'],
                trainable_tensor_sha256=state_digest,adapter_sha256=sha(ck/'adapter_model.safetensors'),
                official_revision=cfg['official_revision'],base=cfg['model_path'],python=cfg['python'],
                processor_config=cfg['processor_config'],proof=str(vp),proof_sha256=sha(vp),
                train_episodes=35,epochs=1,training_fps=5,config_path=str(ck/'run_config.json'),
                uses_future_endpoint_reference=True,gpu_load_verified=False)
            dataset = []
            for row in (r for r in episodes if r['suite'] == suite):
                for i,f in enumerate(row['frame_indices']):
                    dataset.append(dict(id=f"{row['key']}/{i:06d}",key=row['key'],suite=suite,task_id=row['task_id'],
                        sample_index=i,frame_index=f,timestamp=row['timestamps'][i],checkpoint_key=key,
                        inference=native_input(row,i,prompts[f"{suite}-{row['task_id']}"])))
            path = DATA/model/suite/'validation.jsonl'; h = freeze_rows(path,dataset); artifacts[str(path)] = h
            binding = dict(registry[key],input_path=str(path),input_sha256=h,episodes=EXPECTED[suite],observations=len(dataset),
                labels=str(DATA/'labels.json'),labels_sha256=artifacts[str(DATA/'labels.json')],
                image_contract='first main, real last main, initial three BEFORE, current three AFTER',
                missing_view_policy='existing None views become native black RGB64; never repair short streams',
                input_fps=5,include_last_frame=True,selection_sha256=sha(SELECTION),evaluation_started=False)
            bp = DATA/model/suite/'data-binding.json'; artifacts[str(bp)] = freeze(bp,binding)
            caches.append(dict(model=model,suite=suite,path=str(bp),sha256=artifacts[str(bp)],episodes=EXPECTED[suite],observations=len(dataset)))
            print('bound',model,suite,len(dataset),flush=True)
    value = dict(schema='soc84-grm-final-weights-v1',episodes=episodes,registry=registry,caches=caches,
        artifacts_sha256=artifacts,protected_files=protected,resolved_sources=resolved,
        selection_sha256=sha(SELECTION),source_inputs_sha256=sha(SOURCE),source_code_sha256=sha(ROOT/'source-checksums.json'),
        original_training_selection_sha256=sha(TRAIN_SELECTION),input_fps=5,include_last_frame=True,
        future_endpoint_reference=True,labels_never_in_model_input=True,
        cache_format='verified shared lossless RGB plus native 8-image input manifests and prompt text',
        model_forward_executed=False,gpu_used=False,evaluation_started=False,automatic_evaluation_enabled=False)
    freeze(DATA/'inputs.json',value)
    assert not torch.cuda.is_initialized()
    save(ROOT/'status.json',dict(status='prepared_pending_independent_verification',evaluation_started=False,updated=time.time()))
    return dict(caches=6,trajectories=84,observations_per_model=sum(len(r['frame_indices']) for r in episodes))


if __name__ == '__main__':
    try: print(json.dumps(prepare()))
    except BaseException:
        import traceback
        save(ROOT/'status.json',dict(status='preparation_failed',evaluation_started=False,error=traceback.format_exc()))
        raise
