"""Serial online DSRL coordinator, serving RMBench's existing policy protocol."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from core import build_algorithm, reward_components, observation, save, load
from rpc import Client
from serve_robomme_tcp import TCPPolicyServer

CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def actor_digest(actor):
    digest = hashlib.sha256()
    for tensor in actor.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Learner:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.policy = Client(args.policy_port)
        self.forest = Client(args.forest_port) if args.reward_mode == 'terminal_progress' else None
        self.policy_info = self.policy.call('metadata')
        self.algorithm = None
        self.mode = 'train'
        self.episode = -1
        self.total_steps = 0
        self.pending = None
        self.score = None
        self.finished = True
        self.initial_actor = None
        self.log('ready', policy=self.policy_info, reward_mode=args.reward_mode)

    def log(self, event, **values):
        row = dict(event=event, time=time.time(), **values)
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def set_mode(self, payload):
        if not self.finished:
            raise RuntimeError('Cannot change mode during an unfinished episode')
        if payload['mode'] not in ('train', 'eval'):
            raise ValueError('Unknown mode')
        self.mode = payload['mode']
        if self.mode == 'eval':
            if self.algorithm is None or self.algorithm._n_updates == 0:
                raise RuntimeError('No trained actor to evaluate')
            before = actor_digest(self.algorithm.actor)
            metadata = load(self.algorithm, self.output / 'last.pt')
            if actor_digest(self.algorithm.actor) != before:
                raise RuntimeError('Saved/reloaded actor differs')
            self.log('reload_verified', actor_sha256=before, metadata=metadata)
        return {'mode': self.mode}

    def warmup_runtime(self):
        return self.policy.call('warmup_runtime')

    def get_runtime_provenance(self):
        return dict(self.policy_info['provenance'], dsrl_mode=self.mode,
                    reward_mode=self.args.reward_mode,
                    forest_checkpoint=self.args.forest_checkpoint if self.forest else None,
                    robometer_commit=self.args.robometer_commit)

    def reset_model(self):
        if not self.finished:
            raise RuntimeError('Unfinished episode: do not silently mix replay across resets')
        self.policy.call('reset_model')
        if self.forest:
            self.forest.call('reset_model')
        self.episode += 1
        self.frame = 0
        self.chunk = 0
        self.pending = self.score = None
        self.features = None
        self.finished = False
        self.episode_return = 0.0
        self.episode_started = time.monotonic()
        return {'episode': self.episode}

    def score_frames(self, observations):
        frames = []
        for obs in observations:
            frames.append({'frame': self.frame, 'head': obs['observation']['head_camera']['rgb']})
            self.frame += 1
        last = observations[-1]
        return self.forest.call('observe', dict(frames=frames, episode=str(self.episode),
                    prompt=self.prompt, images=[last['observation'][k]['rgb'] for k in CAMERAS]))

    def eval(self, payload):
        if self.pending is not None or self.finished:
            raise RuntimeError('Action requested in an invalid episode state')
        obs = payload['observation']
        self.prompt = payload['prompt']
        limit = int(payload['step_limit'])
        if self.features is None:
            self.features = self.policy.call('features', payload)
        if self.forest and self.score is None:
            self.score = self.score_frames([obs])
            self.initial_progress = self.score['progress']
        current = observation(self.features, obs['joint_action']['vector'],
                              max(0, 1 - payload['action_step'] / limit))
        if self.algorithm is None:
            self.algorithm = build_algorithm(len(current['features']), self.policy_info['noise_dim'],
                batch_size=self.args.batch_size, learning_starts=self.args.learning_starts,
                updates_per_step=self.args.updates_per_step)
            if self.args.resume:
                metadata = load(self.algorithm, Path(self.args.resume))
                if metadata['reward_mode'] != self.args.reward_mode or metadata['robometer_commit'] != self.args.robometer_commit:
                    raise ValueError('Resume reward mode or upstream version mismatch')
                if actor_digest(self.algorithm.actor) != metadata['actor_sha256']:
                    raise ValueError('Resume actor checksum mismatch')
                self.total_steps = metadata['total_steps']
                self.episode = metadata['episode'] + 1
                self.log('resumed', checkpoint=self.args.resume, total_steps=self.total_steps,
                         updates=self.algorithm._n_updates, replay_size=len(self.algorithm.buffer),
                         next_episode=self.episode, actor_sha256=actor_digest(self.algorithm.actor))
            self.initial_actor = actor_digest(self.algorithm.actor)
        with torch.no_grad():
            inputs = {k: torch.from_numpy(v[None]) for k, v in current.items()}
            mean, log_std, kwargs = self.algorithm.actor.get_action_dist_params(inputs)
            dist = self.algorithm.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
            noise = dist.get_actions(deterministic=self.mode == 'eval')[0].numpy()
        self.pending = dict(obs=current, noise=noise, progress=self.score['progress'] if self.score else None)
        return self.policy.call('eval', dict(prompt=self.prompt, observation=obs, noise=noise))

    def add_buffer(self, payload):
        started = time.monotonic()
        if self.pending is None:
            raise RuntimeError('No action pending')
        observations = payload['observations']
        result = self.policy.call('add_buffer', dict(observations=observations, actions=payload['actions']))
        if self.forest:
            self.score = self.score_frames(observations)
        self.features = self.policy.call('features', dict(prompt=self.prompt, observation=observations[-1]))
        remaining = max(0, 1 - payload['action_step'] / payload['step_limit'])
        next_obs = observation(self.features, observations[-1]['joint_action']['vector'], remaining)
        # Success and the native finite-horizon deadline terminate the task.
        # fail_flag is audit-only: it neither creates reward nor changes stopping.
        done = bool(payload['success'] or payload['action_step'] >= payload['step_limit'])
        terminal_reward, dense_reward = reward_components(self.args.reward_mode, payload['success'], done,
                    self.pending['progress'], self.score['progress'] if self.score else None)
        reward = terminal_reward + dense_reward
        self.total_steps += int(self.mode == 'train')
        self.chunk += 1
        self.episode_return += reward
        metrics = {}
        if self.mode == 'train':
            self.algorithm.buffer.add(obs=self.pending['obs'], action=self.pending['noise'], reward=reward,
                       next_obs=next_obs, done=done, truncated=False,
                       episode_id=self.episode, step_in_episode=self.chunk - 1)
            metrics = self.algorithm.train_step(rollout_step=self.total_steps)
            if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))):
                raise ValueError('Non-finite SAC metrics')
        self.log('transition', mode=self.mode, episode=self.episode, chunk=self.chunk,
                 action_step=payload['action_step'], step_limit=payload['step_limit'],
                 progress_before=self.pending['progress'], progress=self.score['progress'] if self.score else None,
                 reward=reward, terminal_reward=terminal_reward, dense_reward=dense_reward,
                 done=done, native_success=bool(payload['success']),
                 native_fail_flag=bool(payload.get('fail_flag', False)),
                 history_tokens=self.score['history_tokens'] if self.score else None, updates=self.algorithm._n_updates,
                 seconds=time.monotonic()-started, metrics=metrics)
        self.pending = None
        if self.mode == 'train' and done:
            self.checkpoint()
        if done:
            self.finished = True
            expected = float(payload['success'])
            if self.forest:
                expected += self.score['progress'] - self.initial_progress
            if abs(self.episode_return - expected) > 1e-5:
                raise RuntimeError('Reward accounting does not telescope')
            self.log('episode_complete', mode=self.mode, episode=self.episode, seed=payload['seed'],
                     prompt=self.prompt, success=bool(payload['success']), steps=payload['action_step'],
                     chunks=self.chunk, return_=self.episode_return,
                     seconds=time.monotonic()-self.episode_started,
                     actor_changed=actor_digest(self.algorithm.actor) != self.initial_actor,
                     updates=self.algorithm._n_updates)
        return result

    def checkpoint(self):
        metadata = dict(total_steps=self.total_steps, episode=self.episode,
                        actor_sha256=actor_digest(self.algorithm.actor),
                        forest_checkpoint=self.args.forest_checkpoint if self.forest else None,
                        robometer_commit=self.args.robometer_commit, reward_mode=self.args.reward_mode,
                        reward='terminal_success + optional(p_next - p); gamma=1')
        save(self.algorithm, self.output / 'last.pt', metadata)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('port', 'policy-port', 'forest-port'):
        p.add_argument('--'+name, required=True, type=int)
    for name in ('output', 'forest-checkpoint', 'robometer-commit'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--reward-mode', choices=('terminal', 'terminal_progress'), required=True)
    p.add_argument('--resume', default='')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-starts', type=int, default=4)
    p.add_argument('--updates-per-step', type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(4)
    TCPPolicyServer(Learner(a), host='127.0.0.1', port=a.port).serve_forever()
