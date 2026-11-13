"""Load each official checkpoint and run synthetic-only smoke inference.

No validation dataset, PT labels, manifests, or evaluation output files are read
or written. Synthetic decoding is capped at 16 tokens solely for this check.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from common import MODELS, ROOT, read, sha, write

CHECKS = Path('/cache/cvpzhr/setup/zeroshot-rebuild-20261113/environment/smoke')


def run_model(name):
    import torch
    from PIL import Image, ImageDraw
    from worker import Generative, Robometer, VLAC, INITIAL_HISTORY
    CHECKS.mkdir(parents=True, exist_ok=True)
    provenance = read(ROOT / 'model-provenance.json')[name]
    cfg = MODELS[name]
    state = {'model': name, 'checkpoint': cfg['path'], 'official_revision': provenance['revision'],
             'status': 'loading', 'pid': os.getpid(), 'started_at': time.time(),
             'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'input_source': 'synthetic images only',
             'dataset_evaluation_started': False, 'local_adapter_loaded': False,
             'smoke_script_sha256': sha(Path(__file__)), 'worker_sha256': sha(ROOT / 'worker.py'),
             'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers']}}
    path = CHECKS / f'{name}.json'
    write(path, state)
    try:
        torch.set_num_threads(4)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        episode = {'task': 'Move the red block to the blue area.', 'frames': []}
        folder = CHECKS / name
        folder.mkdir(exist_ok=True)
        for i in range(10):
            paths = []
            for view in range(3):
                im = Image.new('RGB', (224, 224), (40 + view * 20, 80, 120))
                draw = ImageDraw.Draw(im)
                draw.rectangle((150, 50, 210, 170), fill=(0, 0, 220))
                draw.rectangle((10 + i * 10, 90, 40 + i * 10, 120), fill=(230, 20, 20))
                file = folder / f'{i}-{view}.png'
                im.save(file); paths.append(str(file))
            episode['frames'].append({'image': paths})
        kind = cfg['kind']
        engine = Robometer(cfg) if kind == 'robometer' else VLAC(cfg) if kind == 'vlac' else Generative(cfg)
        state.update(status='loaded', loaded_at=time.time())
        write(path, state)
        if kind == 'robometer':
            from worker import images_for, image
            inputs = engine.raw_batch(engine.processor, [(episode['task'],
                       [image(p) for p in images_for(kind, episode, 9)])])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = engine.logits(engine.model, {k: v.to('cuda') for k, v in inputs.items()})
                assert torch.isfinite(logits).all()
                values, responses = engine.values(logits).cpu().tolist(), [None]
        else:
            if kind == 'vlac':
                engine.model.max_tokens = 16
                engine.model.set_config()
                state['versions']['ms-swift'] = importlib.metadata.version('ms-swift')
            else:
                from vllm import SamplingParams
                engine.sampling = SamplingParams(temperature=0, max_tokens=16)
                state['versions']['vllm'] = importlib.metadata.version('vllm')
            values, responses = engine.infer([(episode, 9, INITIAL_HISTORY)])
            # Arbitrary synthetic scenes can yield unparseable task progress;
            # this checks the native engine, not accuracy or response parsing.
            assert len(responses) == 1 and isinstance(responses[0], str) and responses[0]
        state.update(status='passed', finished_at=time.time(), native_forward_passed=True,
                     synthetic_values=values, synthetic_raw_responses=responses,
                     config_sha256=sha(Path(cfg['path']) / 'config.json'))
        write(path, state)
        del engine
        gc.collect()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), finished_at=time.time())
        write(path, state)
        raise


def run_queue():
    import fcntl
    CHECKS.mkdir(parents=True, exist_ok=True)
    lock = (CHECKS / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from pipeline import gpu_free
    # Two generative workers plus Robometer and VLAC exercise all four cards.
    order = ['robometer-4b', 'vlac-2b', 'robobrain25-4b', 'procvlm-2b',
             'robobrain25-8b', 'dopamine-3b', 'dopamine-8b', 'r2vlm-alfred-rl', 'vlac-8b']
    done, failures, live, handles = [], [], {}, []
    for name in list(order):
        p = CHECKS / f'{name}.json'
        if p.exists():
            old = read(p)
            if old['status'] not in ('passed', 'failed'):
                proc = Path('/proc') / str(old['pid'])
                if proc.exists() and (proc / 'stat').read_text().split()[2] != 'Z':
                    raise RuntimeError('Previous smoke process still alive: ' + name)
            if old['status'] == 'passed' and old.get('worker_sha256') == sha(ROOT / 'worker.py'):
                done.append(name); order.remove(name)
    while order or live:
        for gpu, (name, process) in list(live.items()):
            rc = process.poll()
            if rc is None:
                continue
            (done if rc == 0 and read(CHECKS / f'{name}.json')['status'] == 'passed' else failures).append(name)
            del live[gpu]
        for gpu in range(4):
            if not order or gpu in live or not gpu_free(gpu):
                continue
            def installed(model):
                kind = MODELS[model]['kind']
                group = kind if kind in ('robometer', 'vlac') else 'generative'
                state = CHECKS.parent / f'{group}-status.json'
                return state.exists() and read(state)['status'] == 'installed'
            eligible = [model for model in order if installed(model)]
            if not eligible:
                continue
            name = eligible[0]
            order.remove(name)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', HF_HOME='/cache/cvpzhr/cache/huggingface',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONNOUSERSITE='1',
                       VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH', None)
            log = (CHECKS / f'{name}.log').open('a'); handles.append(log)
            p = subprocess.Popen([MODELS[name]['python'], str(Path(__file__)), '--model', name],
                                 env=env, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
            live[gpu] = (name, p)
        write(CHECKS / 'status.json', dict(status='running' if order or live else 'failed' if failures else 'passed',
              passed=done, failed=failures, pending=order,
              active={str(g): {'model': n, 'pid': p.pid} for g, (n, p) in live.items()},
              dataset_evaluation_started=False, updated_at=time.time()))
        if order or live:
            time.sleep(5)
    for handle in handles:
        handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS))
    args = parser.parse_args()
    run_model(args.model) if args.model else run_queue()
