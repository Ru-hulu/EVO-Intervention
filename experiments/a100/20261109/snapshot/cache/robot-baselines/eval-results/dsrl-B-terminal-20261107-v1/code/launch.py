"""Launch one isolated A/B smoke run with real native episodes and saved-actor eval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import yaml

ROOT = Path('/cache/robot-baselines')
CODE = Path(__file__).resolve().parent
IMPRINT = ROOT/'repos/imprint'
EVAL = IMPRINT/'benchmarks/rmbench/eval'
ADAPTER = EVAL/'runtime_overlay/adapter'
RMBENCH = ROOT/'dependencies/EventVLA/RoboTwin-Mem'
FOREST_CODE = Path('/cache/cvpzhr/setup/forest-s01-decline-20261106-4gpu/code')
UPSTREAM = '4fe3476230befaea6ec6fd84c5a8a24eae049669'


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt('Stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    p = argparse.ArgumentParser()
    p.add_argument('--reward-mode', choices=('terminal', 'terminal_progress'), required=True)
    p.add_argument('--policy-gpu', default='0')
    p.add_argument('--forest-gpu', default='1')
    p.add_argument('--sim-gpu', default=None)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-starts', type=int, default=4)
    p.add_argument('--updates-per-step', type=int, default=1)
    p.add_argument('--port', type=int, default=18400)
    p.add_argument('--output', required=True)
    p.add_argument('--train-episodes', type=int, default=1)
    p.add_argument('--eval-episodes', type=int, default=1)
    p.add_argument('--train-seed', type=int, default=310000)
    p.add_argument('--eval-seed', type=int, default=320000)
    p.add_argument('--forest-checkpoint', default='/cache/cvpzhr/outputs/forest-s01-critical-sampling-20261106/best.pt')
    a = p.parse_args()
    if min(a.train_episodes, a.eval_episodes, a.batch_size, a.learning_starts, a.updates_per_step) < 1:
        p.error('Episode counts and learner settings must be positive')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Run outputs are immutable; choose a new output path')
    out.mkdir(parents=True)
    shutil.copytree(CODE, out/'code', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    run_code = out/'code'
    # Pin the reward snapshot so ongoing FOREST training cannot change rewards.
    checkpoint = out/'forest.pt'
    if a.reward_mode == 'terminal_progress':
        shutil.copyfile(a.forest_checkpoint, checkpoint)
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    else:
        sha = None
    manifest = dict(vars(a), robometer_commit=UPSTREAM, forest_sha256=sha,
        task='cover_blocks', architecture='recurrent', execution_horizon=50, renderer_spp=8,
        instruction_type='unseen', native_step_limit=True,
        actor_observation='frozen IMPRINT memory read + recent visual + language + proprio + remaining horizon',
        reward='native terminal success + optional FOREST progress difference', gamma=1.0,
        scope='online DSRL; report actual episode counts; no claim of improvement without baseline')
    dump(out/'manifest.json', manifest)
    config = dict(adapter_module='robomme_gated_delta_optimized_tcp_policy',
                  checkpoint=str(ROOT/'models/IMPRINT/simulation/rmbench/recurrent/stage2/step-100000'),
                  config_name='imprint_rmbench_recurrent_memft',
                  history_config='cosmo-wan-gated-delta-rmbench-all9-dinov3-memft-optimized.yaml',
                  prediction_horizon=50, seed=42)
    (out/'policy.yaml').write_text(yaml.safe_dump(config))
    pythonpath = ':'.join(map(str, [run_code, ADAPTER, EVAL/'src', IMPRINT/'src', IMPRINT/'packages/openpi-client/src']))
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        HF_HOME=str(ROOT/'.cache/huggingface'), HF_HUB_CACHE=str(ROOT/'.cache/huggingface/hub'),
        OPENPI_DATA_HOME=str(ROOT/'.cache/openpi'))
    policy_env = dict(env, CUDA_VISIBLE_DEVICES=a.policy_gpu,
        EVAL_CONSOLIDATE_ALL='1', EVAL_VISUAL_PROFILE='dinov3',
        DINOV3_MODEL_PATH=str(ROOT/'models/DINOv3-ViT-B16'),
        MEMFT_CAUSAL_BUFFER_REQUIRED='1', MEMFT_LEGACY_EMPTY_RECENT='0',
        MEMFT_MEMORY_CAMERA_ORDER='head_camera,left_camera,right_camera',
        MEMFT_EXPECTED_MODEL_TYPE='pi05', MEMFT_DETERMINISTIC_INFERENCE='0',
        MEMORY_LANGUAGE_SOURCE='live', IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG='1',
        RMBENCH_POLICY_INPUT_COLOR_ORDER='bgr', RMBENCH_PREDICTION_HORIZON='50',
        MEMFT_EXPECTED_MEMORY_ARCHITECTURE='gated_delta_hybrid_v1',
        MEMFT_EXPECTED_MEMORY_TRAIN_SCAN='chunkwise_wy',
        MEMFT_EXPECTED_MEMORY_INFERENCE='incremental_recurrent')
    processes, logs = [], []

    def start(name, interpreter, script, arguments, environment, port=None):
        log = (out/(name+'.log')).open('w'); logs.append(log)
        proc = subprocess.Popen([str(interpreter), '-u', str(run_code/script), *map(str, arguments)],
                    env=environment, cwd=RMBENCH if name.startswith('sim') else IMPRINT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        dump(out/'processes.json', dict(controller=os.getpid(), children=[p.pid for p in processes]))
        if port is not None:
            deadline = time.monotonic()+1200
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'{name} exited {proc.returncode}; see {name}.log')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        return proc
                except OSError:
                    time.sleep(1)
            raise TimeoutError(name+' startup')
        return proc

    try:
        start('policy', ROOT/'envs/imprint/bin/python', 'policy_service.py',
              ['--config', out/'policy.yaml', '--port', a.port+1], policy_env, a.port+1)
        if a.reward_mode == 'terminal_progress':
            start('forest', Path('/cache/cvpzhr/envs/forest/bin/python'), 'forest_service.py',
                  ['--checkpoint', checkpoint, '--port', a.port+2],
                  dict(env, CUDA_VISIBLE_DEVICES=a.forest_gpu, PYTHONPATH=str(FOREST_CODE)+':'+pythonpath), a.port+2)
        start('learner', ROOT/'envs/imprint-dsrl/bin/python', 'learner_service.py',
              ['--port', a.port, '--policy-port', a.port+1, '--forest-port', a.port+2,
               '--output', out/'learner', '--forest-checkpoint', checkpoint,
               '--robometer-commit', UPSTREAM, '--reward-mode', a.reward_mode,
               '--batch-size', a.batch_size, '--learning-starts', a.learning_starts,
               '--updates-per-step', a.updates_per_step],
              dict(env, CUDA_VISIBLE_DEVICES='',
                   PYTHONPATH=str(ROOT/'repos/robometer-policy-learning')+':'+pythonpath), a.port)
        sys.path.insert(0, str(ADAPTER))
        from rpc import Client
        client = Client(a.port)
        for mode, count, seed in [('train', a.train_episodes, a.train_seed), ('eval', a.eval_episodes, a.eval_seed)]:
            client.call('set_mode', {'mode': mode})
            phase = out/mode
            phase.mkdir()
            arguments = dict(task_name='cover_blocks', task_config='demo_clean', ckpt_setting='stage2',
                policy_name='dsrl_adapter', instruction_type='unseen', seed=0, port=a.port,
                execution_horizon=50, start_seed=seed, end_seed=None, test_num=count, result_dir=str(phase))
            dump(phase/'evaluator_args.json', arguments)
            # Same scoped three-camera collector already used for IMPRINT data.
            sim_env = dict(env, CUDA_VISIBLE_DEVICES=a.sim_gpu or a.policy_gpu,
                PYTHONPATH=pythonpath+':'+str(RMBENCH/'script')+':'+str(ROOT/'setup/imprint-extra-20261105'),
                RMBENCH_RENDER_DEVICE='cuda:0', RMBENCH_POLICY_INPUT_COLOR_ORDER='bgr',
                RMBENCH_MODEL_CLIENT_TIMEOUT='1800', VLA_RENDER_SPP='8')
            sim = start('sim-'+mode, ROOT/'envs/rmbench/bin/python', 'sim_entry.py',
                [EVAL/'src/rmbench_l40_eval/runtime/eval_policy_four_view.py',
                 '--canonical-eval-policy', RMBENCH/'script/eval_policy_client.py',
                 '--canonical-working-directory', RMBENCH,
                 '--user-args-json', phase/'evaluator_args.json', '--video-output', phase,
                 '--task_name', 'cover_blocks', '--execution_horizon', '50', '--start_seed', seed], sim_env)
            while sim.poll() is None:
                for proc in processes[:-1]:
                    if proc.poll() is not None and proc.returncode != 0:
                        raise RuntimeError('A service failed; see logs')
                time.sleep(2)
            if sim.returncode:
                raise RuntimeError(f'Simulator {mode} failed: {sim.returncode}')
        client.close()
        events = [json.loads(s) for s in (out/'learner/events.jsonl').read_text().splitlines()]
        episodes = [e for e in events if e['event'] == 'episode_complete']
        train = [e for e in episodes if e['mode'] == 'train']
        evaluation = [e for e in episodes if e['mode'] == 'eval']
        if len(train) != a.train_episodes or len(evaluation) != a.eval_episodes:
            raise RuntimeError('Missing complete benchmark episodes')
        if not train[-1]['actor_changed'] or train[-1]['updates'] == 0:
            raise RuntimeError('Actor never updated')
        dump(out/'result.json', dict(status='complete', reward_mode=a.reward_mode,
            train=train, evaluation=evaluation, eval_successes=sum(e['success'] for e in evaluation),
            eval_episodes=len(evaluation), eval_success_rate=sum(e['success'] for e in evaluation)/len(evaluation),
            checkpoint_reload_verified=any(e['event']=='reload_verified' for e in events)))
    except BaseException as error:
        dump(out/'error.json', dict(error=repr(error)))
        raise
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
