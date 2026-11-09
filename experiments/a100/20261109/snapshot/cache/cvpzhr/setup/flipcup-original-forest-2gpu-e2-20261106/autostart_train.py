"""Two-GPU, two-epoch Forest fine-tuning with half-epoch rolling checkpoints."""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

SETUP = Path(__file__).resolve().parent
CACHE = Path('/cache/cvpzhr/datasets/forest-flipcup-normal-recovery-aug-5fps-20261106')
CACHE_JOB = Path('/cache/cvpzhr/setup/flipcup-original-forest-cache-20261106')
OUTPUT = Path('/cache/cvpzhr/outputs/forest-flipcup-normal-recovery-2gpu-e2-20261106')
PYTHON = '/cache/cvpzhr/envs/forest/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def cache_ready():
    state = read(CACHE_JOB/'status.json')
    if state['phase'].endswith('failed'):
        raise RuntimeError(f'Cache job failed: {state["phase"]}')
    if state['phase'] != 'completed':
        return False
    report, index = read(CACHE/'validation-report.json'), read(CACHE/'index.json')
    assert report['status'] == 'completed' and report['index_sha256'] == index['index_sha256']
    assert report['episode_counts'] == {'train': 53, 'validation': 14}
    assert report['observations'] == {'train': 19841, 'validation': 3727}
    assert all(report[key] for key in ['sources_unchanged', 'histories_causal', 'labels_follow_source_frames',
                                      'three_views_synchronized', 'validation_unaugmented'])
    for split, expected in [('train', 53), ('validation', 14)]:
        paths = read(CACHE/f'{split}.sample_files.json')
        assert len(paths) == expected
        assert all((Path(p)/'summary.json').is_file() for p in paths)
    assert not set(read(CACHE/'train.sample_files.json')) & set(read(CACHE/'validation.sample_files.json'))
    return True


def gpus_available():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = {int(line.split(',')[0]):int(line.split(',')[1]) for line in output.splitlines()}
    return all(memory.get(i, 81920) < 1024 for i in range(2)), memory


def make_config():
    index = read(CACHE/'index.json')
    reference = read(CACHE/'reference-config.json')
    checkpoint = index['checkpoint']
    assert checkpoint['format'] == 'mem-lite-streamforest-v1' and checkpoint['epoch'] == 1
    assert sha256(checkpoint['path']) == checkpoint['sha256']
    config = {key: copy.deepcopy(value) for key, value in reference.items()
              if key not in ['data_signature', 'validation_data_signature', 'validation_sample_files',
                             'validation_suite_by_episode', 'best_validation_mae_pp', 'best_validation_step', 'initialization']}
    config.update(sample_files=read(CACHE/'train.sample_files.json'), output_dir=str(OUTPUT),
                  init_checkpoint=checkpoint['path'], epochs=2, max_steps=None,
                  batch_size=4, gradient_accumulation_steps=2, num_workers=4,
                  learning_rate=1e-4, log_every_steps=10, save_every_steps=None, save_every_epochs=1,
                  keep_only_latest=True, rolling_checkpoints=True, validation_interval_epochs=0.5,
                  validation=dict(sample_files=read(CACHE/'validation.sample_files.json'),
                                  expected_samples=3727, batch_size=4, num_workers=2))
    assert not config['model'].get('cross_attention_layers')
    write(SETUP/'config.json', config)
    return config


def smoke_subset(split, size):
    index = read(CACHE/'index.json')
    paths = []
    for kind in ['normal', 'recovery']:
        record = max((r for r in index['records'] if r['split'] == split and r['dataset_kind'] == kind),
                     key=lambda r:r['augmented_samples'])
        source = Path(record['output_dir'])
        output = SETUP/'smoke-data'/split/kind
        if output.exists():
            raise RuntimeError(f'Smoke directory already exists: {output}')
        output.mkdir(parents=True)
        rows = [json.loads(line) for line in (source/'samples.jsonl').read_text().splitlines()][-size:]
        assert len(rows) == size
        # Start this short probe as a fresh stream; all following histories retain their real causal predecessor.
        rows[0]['history'] = None
        for name in ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'history']:
            (output/name).symlink_to(source/name, target_is_directory=True)
        summary = read(source/'summary.json')
        summary.update(episode_id=f'smoke-{split}-{kind}', num_samples=size)
        write(output/'summary.json', summary)
        (output/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        paths.append(str(output))
    return paths


def main():
    lock = (SETUP/'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    state = dict(pid=os.getpid(), started_at=started, training_started=False,
                 cache_dir=str(CACHE), output_dir=str(OUTPUT))
    env = dict(os.environ, PYTHONPATH=str(SETUP/'code'), CUDA_VISIBLE_DEVICES='0,1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def status(phase, **updates):
        state.update(phase=phase, updated_at=time.time(), elapsed_seconds=time.time()-started, **updates)
        write(SETUP/'status.json', state)

    def launch(config, log_name):
        with (SETUP/log_name).open('a') as log:
            process = subprocess.Popen([PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone',
                                        '--nproc_per_node=2', '-m', 'mem_lite.train_stage3', '--config', str(config)],
                                       cwd=SETUP/'code', env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
        state['launcher_pid'] = process.pid
        write(SETUP/'status.json', state)
        return process.wait()

    try:
        status('waiting_for_cache')
        while not cache_ready():
            status('waiting_for_cache', cache_phase=read(CACHE_JOB/'status.json')['phase'])
            time.sleep(20)
        status('waiting_for_gpus')
        while True:
            ready, memory = gpus_available()
            if ready:
                break
            status('waiting_for_gpus', memory_used_mib=memory)
            time.sleep(20)
        if OUTPUT.exists():
            raise RuntimeError(f'Refusing to mix with an existing training output: {OUTPUT}')
        config = make_config()
        train_paths, val_paths = smoke_subset('train', 16), smoke_subset('validation', 4)
        passed = False
        for batch in [4, 2, 1]:
            smoke = copy.deepcopy(config)
            smoke.update(sample_files=train_paths, output_dir=str(SETUP/f'smoke-b{batch}'),
                         epochs=1, max_steps=2, batch_size=batch, gradient_accumulation_steps=8//batch,
                         log_every_steps=1, num_workers=0, persistent_workers=False,
                         validation=dict(sample_files=val_paths, expected_samples=8, batch_size=batch, num_workers=0))
            path = SETUP/f'smoke-b{batch}.config.json'
            write(path, smoke)
            status('gpu_smoke', batch_size=batch)
            code = launch(path, f'smoke-b{batch}.log')
            if code:
                log = (SETUP/f'smoke-b{batch}.log').read_text()
                if 'out of memory' in log.lower() and batch > 1:
                    continue
                raise RuntimeError(f'Two-GPU smoke failed (batch {batch}); see smoke-b{batch}.log')
            import torch
            saved = torch.load(SETUP/f'smoke-b{batch}/last.pt', map_location='cpu', weights_only=True, mmap=True)
            assert saved['format'] == 'mem-lite-streamforest-v1' and saved['step'] == 2
            assert saved['config']['initialization']['epoch'] == 1
            assert saved['config']['initialization']['step'] == 19934
            assert all(float(v['step']) == 2 for v in saved['optimizer']['state'].values())
            metrics = [json.loads(line) for line in (SETUP/f'smoke-b{batch}/validation.jsonl').read_text().splitlines()]
            assert len(metrics) == 2 and all(row['num_samples'] == 8 and math.isfinite(row['loss']) for row in metrics)
            assert all(len(row['groups']) == 2 for row in metrics)
            assert [row['epoch'] for row in metrics] == [0.5, 1.0]
            assert (SETUP/f'smoke-b{batch}/last.pt').is_symlink()
            assert not (SETUP/f'smoke-b{batch}/step-000001.pt').exists()
            assert (SETUP/f'smoke-b{batch}/step-000002.pt').is_file()
            del saved
            config.update(batch_size=batch, gradient_accumulation_steps=8//batch)
            config['validation']['batch_size'] = batch
            write(SETUP/'config.json', config)
            write(SETUP/'smoke-passed.json', dict(passed=True, world_size=2, batch_size=batch,
                  gradient_accumulation_steps=8//batch, checkpoint_epoch=1, optimizer_reset=True,
                  validation_events=2, observations_per_validation=8, rolling_checkpoint_deletion_verified=True))
            passed = True
            break
        if not passed:
            raise RuntimeError('No batch size passed smoke test')
        for temporary in SETUP.glob('smoke-b[124]'):
            if temporary.is_dir() and not temporary.is_symlink():
                shutil.rmtree(temporary)
        shutil.rmtree(SETUP/'smoke-data')
        status('training', training_started=True, batch_size=config['batch_size'],
               gradient_accumulation_steps=config['gradient_accumulation_steps'], epochs=2,
               validation_interval_epochs=0.5, gpu_ids=[0, 1])
        code = launch(SETUP/'config.json', 'train.log')
        status('completed' if code == 0 else 'training_failed', returncode=code)
        return code
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    sys.exit(main())
