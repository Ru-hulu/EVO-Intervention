"""Native CPU pixels and assistant-only GRM score targets; no model loading."""
import argparse
import fcntl
import importlib.util
import sys
import time
import traceback
from .common import *


def initialize(model):
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    ref = read(ROOT/'protocol.json')['models'][model]
    for p, h in ref['source_sha256'].items(): assert sha(p) == h, p
    cfg = ref['config']
    sys.path.insert(0, str(OLD))
    spec = importlib.util.spec_from_file_location('soc35_native_grm', OLD/'worker.py')
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    assert native.MODELS[model] == cfg
    processor = AutoProcessor.from_pretrained(cfg['path'], local_files_only=True)
    opts = {k:cfg[k] for k in ('min_pixels', 'max_pixels')}
    for k, v in opts.items(): setattr(processor.image_processor, k, v)
    obj = object.__new__(native.Generative); obj.kind = 'grm'; obj.processor = processor
    obj.template = (OLD/'grm_prompt.txt').read_text()
    def prompt(task):
        return obj.request((dict(task=task, frames=[dict(image=[None]*3)]), 0, None))['prompt']
    def encode(task, paths):
        return dict(processor(text=[prompt(task)], images=[native.image(p) for p in paths],
                              return_tensors='pt', **opts))
    return torch, processor, opts, prompt, encode


def build_episode(model, ep, processor, opts, prompt, encode, torch, status_path, completed):
    from PIL import Image
    from safetensors.torch import save_file, load_file
    from experiments.recover_six_caches_20261114.loader import load_input
    from experiments.recover_finetune_20261114.data import CachedCollator, grm_answer
    base = OUT/model
    receipt_path = base/'episodes'/(ep['episode_id']+'.verification.json')
    if receipt_path.exists():
        receipt = read(receipt_path)
        assert receipt['selection_sha256'] == sha(ROOT/'selection.json')
        assert sha(receipt['train_data']) == receipt['train_sha256']
        for p, h in receipt['files_sha256'].items(): assert sha(p) == h, p
        return receipt
    observed = rows(ep['observations']); n = len(observed)
    assert sha(ep['observations']) == ep['observations_sha256']
    pixel_map, grids, file_hashes = {}, {}, {}
    checked_png = read(ROOT/'source-images.json')
    for i, row in enumerate(observed):
        if i % 25 == 0:
            resource_check()
            save(status_path, dict(status='processing', pid=os.getpid(), episode=ep['episode_id'],
                 completed_episodes=completed, current_observation=i, current_total=n, updated=time.time()))
        for v, view in enumerate(VIEWS):
            image = row['images'][view]
            if image in pixel_map: continue
            assert sha(image) == checked_png[image], image
            with Image.open(image) as im:
                encoded = processor.image_processor(images=[im.convert('RGB')], return_tensors='pt', **opts)
            tensors = {k:t.contiguous().to(torch.bfloat16) if t.is_floating_point() else t.contiguous()
                       for k,t in encoded.items() if isinstance(t, torch.Tensor)}
            assert set(tensors) == {'pixel_values','image_grid_thw'}
            assert all(torch.isfinite(t).all() for t in tensors.values())
            path = base/'pixels'/ep['episode_id']/f'{i:06d}-{v}.safetensors'
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            save_file(tensors, str(temporary), metadata=dict(source_png_sha256=checked_png[image], model=model))
            # A failed, incomplete episode is resumable without modifying committed episodes.
            if path.exists():
                assert sha(path) == sha(temporary), f'Conflicting partial cache: {path}'
                temporary.unlink()
            else: temporary.replace(path)
            pixel_map[image] = str(path); grids[image] = tensors['image_grid_thw'].tolist()
            file_hashes[str(path)] = sha(path)
    prepared, prompts, prompt_lengths = [], {}, {}
    collator = CachedCollator(processor.tokenizer, model, 4096)
    chosen = sorted({0, 1, n//2, n-1} & set(range(n)))
    max_length = 0
    for i, r in enumerate(observed):
        pairs = slots(i,n); images = [observed[j]['images'][VIEWS[v]] for j,v in pairs]
        signature = json.dumps([grids[p] for p in images])
        online = None
        if signature not in prompts:
            online = encode(ep['task'],images)
            text = {k:v.squeeze(0).contiguous() for k,v in online.items()
                    if isinstance(v,torch.Tensor) and k not in ('pixel_values','image_grid_thw')}
            tag = hashlib.sha256(signature.encode()).hexdigest()[:16]
            path = base/'prompts'/ep['episode_id']/(tag+'.safetensors')
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(text,str(path),metadata=dict(model=model,task_prompt=ep['task']))
            file_hashes[str(path)] = sha(path); prompts[signature] = str(path)
            prompt_lengths[signature] = text['input_ids'].numel()
        row = dict(id=f'{ep["episode_id"]}/{i:06d}', episode_id=ep['episode_id'], suite=ep['suite'],
            task_id=ep['task_id'], split='train', sample_index=i, frame_index=r['frame_index'],
            timestamp_seconds=r['frame_index']/ep['fps'], pt=r['pt'], before_pt=observed[0]['pt'], after_pt=r['pt'],
            task=ep['task'], image=images, slots=[list(p) for p in pairs],
            window_frame_indices=[observed[j]['frame_index'] for j,v in pairs],
            pixel_caches=[pixel_map[p] for p in images], prompt_tensor_file=prompts[signature],
            missing_current_views=r['missing_views'], source_progress_sha256=ep['progress_sha256'],
            source_samples_sha256=ep['source_samples_sha256'], selection_sha256=sha(ROOT/'selection.json'),
            uses_future_endpoint_reference=True, augmentation='none', sample_fps=5)
        row['assistant_target'] = grm_answer(row)
        suffix = processor.tokenizer.encode(row['assistant_target']+processor.tokenizer.eos_token,add_special_tokens=False)
        length = prompt_lengths[signature] + len(suffix)
        assert length <= 4096, 'Native inputs must not be truncated'
        max_length = max(max_length, length)
        if i in chosen:
            if online is None: online = encode(ep['task'], images)
            cached, supervision = load_input(row)
            assert set(cached) == set(online)
            for k in cached:
                assert torch.equal(cached[k], online[k].to(cached[k].dtype)), (model,ep['episode_id'],i,k)
            training = collator([row])
            active = training['labels'][0] != -100
            assert processor.tokenizer.decode(training['labels'][0][active],skip_special_tokens=False) == row['assistant_target']+processor.tokenizer.eos_token
            assert training['labels'][0,:cached['input_ids'].shape[1]].eq(-100).all()
            rendered = prompt(ep['task'])
            prefix_ids = processor.tokenizer.encode(rendered,add_special_tokens=False)
            full_ids = processor.tokenizer.encode(rendered+row['assistant_target']+processor.tokenizer.eos_token,add_special_tokens=False)
            assert full_ids[:len(prefix_ids)] == prefix_ids and full_ids[len(prefix_ids):] == suffix
            assert not any(k in cached for k in ('pt','progress_gt','labels','before_pt','after_pt'))
            max_length = max(max_length,training['input_ids'].shape[1])
        prepared.append(row)
    # Boundary merging and padding are checked on a mixed pair, not just single inputs.
    mixed = collator([prepared[0], prepared[-1]])
    assert mixed['labels'].shape[0] == 2
    target = base/'episodes'/(ep['episode_id']+'.jsonl')
    write_rows(target, prepared)
    receipt = dict(episode_id=ep['episode_id'],suite=ep['suite'],samples=n,
        train_data=str(target),train_sha256=sha(target),selection_sha256=sha(ROOT/'selection.json'),
        files_sha256=file_hashes,pixel_bytes=sum(Path(p).stat().st_size for p in file_hashes),
        native_cached_inputs_equal=True,checked_indices=chosen,supervision_mask_verified=True,
        max_checked_length=max_length,cuda_used=False,model_forward_executed=False)
    save(receipt_path,receipt)
    return receipt


def main(model,suite,smoke=False):
    status = ROOT/f'{model}-{suite}-status.json'
    lock = (ROOT/f'{model}-{suite}.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch,processor,opts,prompt,encode = initialize(model)
    selected=read(ROOT/'selection.json'); episodes=[e for e in selected['episodes'] if e['suite']==suite]
    if smoke: episodes=episodes[:1]
    receipts=[]
    try:
        for ep in episodes:
            receipts.append(build_episode(model,ep,processor,opts,prompt,encode,torch,status,len(receipts)))
            print(model,suite,len(receipts),ep['episode_id'],ep['samples'],flush=True)
        if not smoke:
            data=[r for receipt in receipts for r in rows(receipt['train_data'])]
            assert len(receipts)==35 and len(data)==selected['suites'][suite]['observations']
            train=OUT/model/suite/'train.jsonl';write_rows(train,data)
            save(OUT/model/suite/'cache-ready.json',dict(status='completed',model=model,suite=suite,
                 episodes=35,observations=len(data),task_counts=selected['suites'][suite]['tasks'],
                 selection_sha256=sha(ROOT/'selection.json'),protocol_sha256=sha(ROOT/'protocol.json'),
                 train_data=str(train),train_sha256=sha(train),receipts=receipts,
                 training_started=False,cuda_used=False,model_forward_executed=False))
        assert not torch.cuda.is_initialized()
        save(status,dict(status='smoke_passed' if smoke else 'completed',pid=os.getpid(),
             completed_episodes=len(receipts),observations=sum(r['samples'] for r in receipts),updated=time.time()))
    except Exception:
        save(status,dict(status='failed',pid=os.getpid(),completed_episodes=len(receipts),
             error=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--suite',choices=SUITES,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();main(a.model,a.suite,a.smoke)
