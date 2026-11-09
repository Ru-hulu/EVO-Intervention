"""Evaluation-only coordinator: no replay writes, optimizer steps or checkpoint writes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from core import build_algorithm, observation, load
from learner_service import Learner, actor_digest
from serve_robomme_tcp import TCPPolicyServer


class Evaluator(Learner):
    def set_mode(self, payload):
        if payload['mode'] != 'eval' or not self.finished:
            raise ValueError('Evaluation-only service')
        self.mode = 'eval'
        return {'mode': self.mode}

    def eval(self, payload):
        if self.pending is not None or self.finished:
            raise RuntimeError('Invalid episode state')
        self.prompt = payload['prompt']
        obs = payload['observation']
        self.pending = True
        if self.args.evaluation_kind == 'original':
            # No steering noise: preserve original policy sampling and memory path.
            return self.policy.call('eval', dict(prompt=self.prompt, observation=obs))
        if self.features is None:
            self.features = self.policy.call('features', payload)
        current = observation(self.features, obs['joint_action']['vector'],
                              max(0, 1-payload['action_step']/payload['step_limit']))
        if self.algorithm is None:
            self.algorithm = build_algorithm(len(current['features']), self.policy_info['noise_dim'],
                batch_size=self.args.batch_size, learning_starts=self.args.learning_starts,
                updates_per_step=self.args.updates_per_step)
            metadata = load(self.algorithm, Path(self.args.resume))
            if metadata['reward_mode'] != 'terminal' or metadata['robometer_commit'] != self.args.robometer_commit:
                raise ValueError('Checkpoint provenance mismatch')
            self.initial_actor = actor_digest(self.algorithm.actor)
            if self.initial_actor != metadata['actor_sha256']:
                raise ValueError('Actor checksum mismatch')
            self.initial_updates = self.algorithm._n_updates
            self.log('reload_verified', actor_sha256=self.initial_actor, metadata=metadata)
        with torch.no_grad():
            inputs = {k: torch.from_numpy(v[None]) for k,v in current.items()}
            mean, log_std, kwargs = self.algorithm.actor.get_action_dist_params(inputs)
            dist = self.algorithm.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
            noise = dist.get_actions(deterministic=True)[0].numpy()
        return self.policy.call('eval', dict(prompt=self.prompt, observation=obs, noise=noise))

    def add_buffer(self, payload):
        if self.pending is None:
            raise RuntimeError('No action pending')
        observations = payload['observations']
        result = self.policy.call('add_buffer', dict(observations=observations, actions=payload['actions']))
        if self.args.evaluation_kind == 'trained':
            self.features = self.policy.call('features', dict(prompt=self.prompt, observation=observations[-1]))
        done = bool(payload['success'] or payload['action_step'] >= payload['step_limit'])
        self.chunk += 1
        self.pending = None
        reward = float(done and payload['success'])
        self.episode_return += reward
        updates = self.algorithm._n_updates if self.algorithm else 0
        self.log('transition', mode='eval', episode=self.episode, chunk=self.chunk,
                 action_step=payload['action_step'], step_limit=payload['step_limit'],
                 progress=None, progress_before=None, reward=reward, dense_reward=0,
                 terminal_reward=reward, done=done, native_success=bool(payload['success']),
                 updates=updates)
        if done:
            import time
            if self.algorithm:
                assert actor_digest(self.algorithm.actor) == self.initial_actor
                assert updates == self.initial_updates
            assert self.episode_return == float(payload['success'])
            self.finished = True
            self.log('episode_complete', mode='eval', episode=self.episode, seed=payload['seed'],
                     prompt=self.prompt, success=bool(payload['success']), steps=payload['action_step'],
                     chunks=self.chunk, return_=self.episode_return, updates=updates,
                     seconds=time.monotonic()-self.episode_started, actor_changed=False)
        return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('port','policy-port','forest-port'):
        p.add_argument('--'+name, required=True, type=int)
    for name in ('output','forest-checkpoint','robometer-commit','reward-mode','resume','evaluation-kind'):
        p.add_argument('--'+name, required=True)
    for name in ('batch-size','learning-starts','updates-per-step'):
        p.add_argument('--'+name, required=True, type=int)
    args=p.parse_args()
    assert args.reward_mode == 'terminal'
    torch.set_num_threads(4)
    TCPPolicyServer(Evaluator(args), host='127.0.0.1', port=args.port).serve_forever()
