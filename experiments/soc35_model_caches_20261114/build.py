"""Freeze the three completed FoREST selections; CPU-only model-native caches."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114')
OUT = Path('/cache/cvpzhr/datasets/robometer-procvlm-soc35-5fps-20261114')
OLD = Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')
VIEWS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
SUITES = {'State': ('state-10pct', 6381), 'Order': ('order-35', 8783), 'Count': ('count-35', 4315)}
MODELS = {'robometer': '/cache/cvpzhr/models/Robometer-4B',
          'procvlm': '/cache/robot-baselines/models/ProcVLM-2B'}


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)); temp.replace(path)


def freeze():
    from PIL import Image
    assert not (ROOT / 'selection.json').exists(), 'Selection already frozen'
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    black = OUT / 'images/_missing_rgb64.png'
    if not black.exists(): Image.new('RGB', (64, 64), (0, 0, 0)).save(black)
    selection = {'schema': 'soc35-exact-forest-selection-v1', 'sample_fps': 5,
        'augmentation': 'none', 'training_started': False, 'suites': {}, 'episodes': []}
    for suite, (tag, expected) in SUITES.items():
        setup = Path(f'/cache/cvpzhr/setup/forest-{tag}-legacy-e1-20261114')
        status = read(setup / 'training-status.json'); config = read(setup / 'training.config.json')
        output = Path(config['output_dir']); saved_config = read(output / 'config.json')
        assert status['status'] == 'completed' and status['final_checkpoint_verified']
        assert Path(status['checkpoint']).is_file()
        assert config['sample_files'] == saved_config['sample_files']
        assert config['data_signature'] == saved_config['data_signature']
        assert len(config['sample_files']) == len(set(config['sample_files'])) == 35
        suite_records = []
        for directory in config['sample_files']:
            source = Path(directory); summary = read(source / 'summary.json'); observed = rows(source / 'samples.jsonl')
            info = summary['source']; verification = read(source / 'cache-verification.json')
            sample_hash = sha(source / 'samples.jsonl')
            assert info['suite'] == suite and info['split'] == 'train'
            assert sample_hash == verification['samples_sha256']
            assert sha(info['progress_path']) == info['progress_sha256'] == verification['progress_sha256']
            assert len(observed) == summary['num_samples'] == verification['samples']
            assert summary['history_source']['sample_fps'] == 5
            with Path(info['progress_path']).open() as stream:
                pt = {int(r['frame_index']): float(r['pt']) for r in csv.DictReader(stream)}
            image_records, image_hash = [], hashlib.sha256()
            for i, r in enumerate(observed):
                assert r['pt'] == pt[r['frame_index']] and 0 <= r['pt'] <= 1
                assert r.get('kind', 'original') == 'original'
                if i: assert r['frame_index'] > observed[i-1]['frame_index']
                images = {}
                for view in VIEWS:
                    rel = r['images'][view]
                    if rel is None:
                        images[view] = str(black); image_hash.update(f'{i}/{view}:missing\n'.encode()); continue
                    rel = Path(rel); assert not rel.is_absolute() and '..' not in rel.parts
                    origin = source / rel; dest = OUT / 'images' / summary['episode_id'] / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists(): assert os.path.samefile(origin, dest)
                    else: os.link(origin, dest)
                    images[view] = str(dest)
                    image_hash.update(f'{i}/{view}:{sha(origin)}\n'.encode())
                assert r['images']['cam_high'] is not None
                image_records.append({'sample_index': i, 'frame_index': r['frame_index'], 'pt': r['pt'],
                    'images': images, 'missing_views': [v for v in VIEWS if r['images'][v] is None]})
            target = OUT / 'observations' / (summary['episode_id'] + '.jsonl')
            write_rows(target, image_records)
            record = dict(episode_id=summary['episode_id'], suite=suite, task_id=info['task_id'],
                sample_id=info['sample_id'], task=summary['task_prompt'], fps=info['fps'], samples=len(observed),
                source_samples=str(source/'samples.jsonl'), source_samples_sha256=sample_hash,
                source_summary_sha256=sha(source/'summary.json'), progress_path=info['progress_path'],
                progress_sha256=info['progress_sha256'], observations=str(target), observations_sha256=sha(target),
                source_image_sequence_sha256=image_hash.hexdigest(), image_storage='hardlinks-to-verified-FoREST-RGB')
            suite_records.append(record); selection['episodes'].append(record)
        assert sum(r['samples'] for r in suite_records) == expected
        selection['suites'][suite] = dict(episodes=35, observations=expected,
            tasks=dict(Counter(r['task_id'] for r in suite_records)),
            forest_config=str(output/'config.json'), forest_config_sha256=sha(output/'config.json'),
            forest_checkpoint=status['checkpoint'], forest_checkpoint_sha256=status['checkpoint_sha256'],
            checkpoint_verification_source=str(setup/'training-status.json'), data_signature=config['data_signature'])
    save(ROOT/'selection.json', selection)
    print(json.dumps(selection['suites'], indent=2), flush=True)


def initialize(model):
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    if model == 'robometer':
        sys.path[:0] = [str(ROOT/'runtime'), '/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
        from robometer_training.model import load_processor
        from robometer_training.data import ProgressCollator
        processor = load_processor(MODELS[model], min_pixels=4096, max_pixels=65536)
    else:
        from transformers import AutoProcessor
        from .proc_data import ProgressCollator
        processor = AutoProcessor.from_pretrained(MODELS[model], local_files_only=True, min_pixels=4096, max_pixels=65536)
    return torch, processor, ProgressCollator(processor, 4096)


def make_records(model, ep, observed, target):
    if model == 'robometer':
        from robometer_training.uniform import prefix_indices, POLICY
    else:
        from .proc_data import answer_text, user_prompt, POLICY
    records = []
    for i, r in enumerate(observed):
        temporal = prefix_indices(i) if model == 'robometer' else [max(0, j) for j in range(i-7, i+1)]
        indices = temporal[:-1] + [i]*3 if model == 'robometer' else temporal
        views = ['cam_high']*8 + VIEWS[1:] if model == 'robometer' else ['cam_high']*8
        assert max(indices) == i
        record = dict(id=f'{ep["episode_id"]}/{i:08d}', episode_id=ep['episode_id'], suite=ep['suite'],
            task_id=ep['task_id'], split='train', sample_index=i, frame_index=r['frame_index'],
            source_frame_index=r['frame_index'], source_sample_index=i, kind='original',
            timestamp_seconds=r['frame_index']/ep['fps'], task=ep['task'], pt=r['pt'],
            forest_reference_pt=r['pt'], pt_policy='unchanged-original', augmentation='none',
            image=[observed[j]['images'][v] for j,v in zip(indices,views)],
            window_indices=indices, window_frame_indices=[observed[j]['frame_index'] for j in indices],
            input_views=views, pixel_cache=str(target), missing_current_views=r['missing_views'])
        if model == 'robometer':
            record.update(input_contract='history7-main-current3-v1', temporal_sampling=POLICY,
                temporal_indices=temporal, history_sample_indices=temporal[:-1], observation_indices=indices,
                padding='repeat_current_frame_on_right')
        else:
            record.update(supervision_policy=POLICY, supervision_eligible=True, previous_cot_input=False,
                assistant_target=answer_text(r['pt']), progress_gt=100*r['pt'],
                conversations=[{'from':'human','value':'<image>\n'*8+user_prompt(ep['task'])},
                               {'from':'gpt','value':answer_text(r['pt'])}])
        records.append(record)
    return records


def build_episode(model, ep, torch, processor, collator):
    from PIL import Image
    from safetensors.torch import save_file
    observed = rows(ep['observations'])
    assert sha(ep['source_samples']) == ep['source_samples_sha256']
    assert sha(ep['observations']) == ep['observations_sha256']
    target = OUT/model/'pixels'/(ep['episode_id']+'.safetensors')
    receipt_path = OUT/model/'episodes'/(ep['episode_id']+'.verification.json')
    if receipt_path.exists():
        receipt=read(receipt_path)
        assert sha(target)==receipt['pixel_sha256']
        assert receipt['source_samples_sha256']==ep['source_samples_sha256']
        assert sha(receipt['episode_jsonl'])==receipt['episode_jsonl_sha256']
        return receipt
    tensors = {}
    for view in VIEWS if model == 'robometer' else VIEWS[:1]:
        pixels, grids = [], []
        for start in range(0, len(observed), 16):
            images=[]
            for row in observed[start:start+16]:
                with Image.open(row['images'][view]) as image: images.append(image.convert('RGB'))
            encoded = processor.image_processor(images=images, return_tensors='pt', min_pixels=4096, max_pixels=65536)
            grid=encoded['image_grid_thw']
            assert torch.equal(grid,grid[:1].expand_as(grid)), (ep['episode_id'],view)
            patches=int(grid[0].prod())
            pixels.append(encoded['pixel_values'].reshape(len(images),patches,-1).to(torch.bfloat16))
            grids.append(grid)
        tensors['pixels_'+view]=torch.cat(pixels).contiguous()
        tensors['grids_'+view]=torch.cat(grids).contiguous()
        assert torch.equal(tensors['grids_'+view], tensors['grids_'+view][:1].expand_as(tensors['grids_'+view]))
        assert torch.isfinite(tensors['pixels_'+view]).all()
    records=make_records(model,ep,observed,target)
    raw_first={k:v for k,v in records[0].items() if k!='pixel_cache'}
    canonical=collator.encode([raw_first],supervised=False)
    tensors.update({'input_'+k:v[0].contiguous() for k,v in canonical.items() if k not in ('pixel_values','image_grid_thw')})
    assert set(tensors) >= {'input_input_ids','input_attention_mask'}
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix('.partial')
    save_file(tensors,str(partial),metadata=dict(model=model,source_samples_sha256=ep['source_samples_sha256'],
        augmentation='none',min_pixels='4096',max_pixels='65536',selection_sha256=sha(ROOT/'selection.json')))
    partial.replace(target); del tensors
    chosen=sorted({0,min(1,len(records)-1),min(6,len(records)-1),min(8,len(records)-1),len(records)//2,len(records)-1})
    max_length=0
    for start in range(0,len(chosen),2):
        selected=[records[i] for i in chosen[start:start+2]]
        for supervised in (False,True):
            cached=collator.encode(selected,supervised=supervised)
            online=collator.encode([{k:v for k,v in r.items() if k!='pixel_cache'} for r in selected],supervised=supervised)
            assert set(cached)==set(online)
            for k in cached:
                a,b=cached[k],online[k]
                if k=='pixel_values': a,b=a.to(torch.bfloat16),b.to(torch.bfloat16)
                assert torch.equal(a,b),(model,ep['episode_id'],chosen[start:start+2],supervised,k)
            max_length=max(max_length,int(cached['input_ids'].shape[1]))
        if model=='procvlm':
            from .proc_data import answer_text,chat_prompt
            tok=processor.tokenizer
            for row in selected:
                # Verify assistant-only mask and the exact full-sequence token boundary.
                batch=collator.encode([row],supervised=True)
                supervised_ids=batch['labels'][0][batch['labels'][0]!=-100].tolist()
                assert tok.decode(supervised_ids,skip_special_tokens=False)==answer_text(row['pt'])+tok.eos_token
                raw=collator.encode([{k:v for k,v in row.items() if k!='pixel_cache'}],supervised=False)
                prompt_only=tok.encode(chat_prompt(processor,row['task']),add_special_tokens=False)
                full=tok.encode(chat_prompt(processor,row['task'])+answer_text(row['pt'])+tok.eos_token,add_special_tokens=False)
                assert full[len(prompt_only):]==supervised_ids
                assert not any(key in raw for key in ['progress_gt','labels'])
    if model=='robometer':
        from robometer_training.data import _episode_cache
        _episode_cache.cache_clear()
    else:
        from .proc_data import episode_cache
        episode_cache.cache_clear()
    dest=OUT/model/'episodes'/(ep['episode_id']+'.jsonl');write_rows(dest,records)
    receipt=dict(episode_id=ep['episode_id'],suite=ep['suite'],samples=len(records),
        pixel_cache=str(target),pixel_bytes=target.stat().st_size,pixel_sha256=sha(target),
        episode_jsonl=str(dest),episode_jsonl_sha256=sha(dest),
        source_samples_sha256=ep['source_samples_sha256'],cached_online_equal=True,
        checked_indices=chosen,max_checked_sequence_length=max_length,pt_unchanged=True,cuda_used=False)
    save(receipt_path,receipt);return receipt


def main(model,suite,smoke=False):
    torch,processor,collator=initialize(model)
    selection=read(ROOT/'selection.json'); episodes=[e for e in selection['episodes'] if e['suite']==suite]
    if smoke: episodes=episodes[:1]
    status_path=ROOT/f'{model}-{suite}-status.json'
    provenance={'model':model,'model_directory':MODELS[model],'min_pixels':4096,'max_pixels':65536,
        'pixel_policy':'existing-model-finetuning-settings','processor_class':type(processor).__name__,
        'processor_image_config':processor.image_processor.to_dict(), 'input_only_cpu':True,
        'source_config_sha256':sha(Path(MODELS[model])/'config.json')}
    save(OUT/model/suite/'processor-provenance.json',provenance)
    receipts=[];started=time.time()
    try:
        for e in episodes:
            save(status_path,dict(status='processing',pid=os.getpid(),episode=e['episode_id'],
                completed_episodes=len(receipts),total_episodes=len(episodes),started=started,updated=time.time()))
            receipts.append(build_episode(model,e,torch,processor,collator))
            print(model,suite,len(receipts),e['episode_id'],e['samples'],flush=True)
        if not smoke:
            records=[row for receipt in receipts for row in rows(receipt['episode_jsonl'])]
            assert len(records)==SUITES[suite][1] and len(receipts)==35
            by_ep={e['episode_id']:rows(e['source_samples']) for e in episodes}
            for r in records:
                source=by_ep[r['episode_id']][r['sample_index']]
                assert r['pt']==source['pt'] and r['frame_index']==source['frame_index']
            dest=OUT/model/suite/'train.jsonl';write_rows(dest,records)
            if model=='robometer':from robometer_training.data import ProgressDataset
            else:from .proc_data import ProgressDataset
            assert len(ProgressDataset(dest))==len(records)
            report=dict(status='completed',model=model,suite=suite,episodes=35,observations=len(records),
                task_counts=selection['suites'][suite]['tasks'],selection_sha256=sha(ROOT/'selection.json'),
                train_jsonl=str(dest),train_sha256=sha(dest),original_pt_and_frames_match=True,
                no_augmentation=True,all_samples_supervision_eligible=True,
                cached_online_inputs_equal=True,model_forward_executed=False,training_started=False,
                pixel_bytes=sum(r['pixel_bytes'] for r in receipts),cpu_only=True,
                supervision='continuous-PT-C51-soft-target' if model=='robometer' else 'progress-only-text-and-continuous-PT',
                receipts=receipts,finished=time.time())
            save(OUT/model/suite/'cache-ready.json',report)
        save(status_path,dict(status='smoke_passed' if smoke else 'completed',pid=os.getpid(),
            completed_episodes=len(receipts),observations=sum(r['samples'] for r in receipts),
            started=started,finished=time.time()))
    except Exception:
        save(status_path,dict(status='failed',pid=os.getpid(),completed_episodes=len(receipts),
                             error=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--model',choices=MODELS)
    p.add_argument('--suite',choices=SUITES);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:main(a.model,a.suite,a.smoke)
