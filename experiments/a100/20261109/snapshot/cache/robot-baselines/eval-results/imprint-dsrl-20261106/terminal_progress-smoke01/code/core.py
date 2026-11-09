"""RoboMeter's SAC/MLP/replay components; no reimplementation of the RL loss."""
import random

import gymnasium as gym
import numpy as np
import torch

from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.modules.mlp import MLPActor, MLPActorConfig, MLPCritic, MLPCriticConfig


def progress_reward(previous, current):
    values = np.asarray([previous, current])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Progress must be finite and in [0, 1]')
    return float(current - previous)


def reward_components(mode, success, done, previous=None, current=None):
    terminal = float(bool(success) and bool(done))
    if mode == 'terminal':
        dense = 0.0
    elif mode == 'terminal_progress':
        dense = progress_reward(previous, current)
    else:
        raise ValueError('Unknown reward mode')
    return terminal, dense


def build_algorithm(feature_dim, noise_dim, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observations = gym.spaces.Dict({'features': gym.spaces.Box(-np.inf, np.inf, (feature_dim,), np.float32)})
    actions = gym.spaces.Box(-1, 1, (noise_dim,), np.float32)
    actor = MLPActor(MLPActorConfig(observation_space=observations, action_space=actions,
                    hidden_dims=(256, 256), use_layer_norm=True, use_tanh_output=True,
                    log_std_init=-0.5))
    critic = MLPCritic(MLPCriticConfig(observation_space=observations, action_space=actions,
                      hidden_dims=(256, 256), use_layer_norm=True))
    buffer = ReplayBuffer(capacity=10000)
    # Finite-horizon return is success (+ p_final - p_initial in dense mode). The native
    # benchmark deadline is terminal, not an infinite-horizon time-limit truncation.
    algorithm = SAC(SACConfig(actor=actor, critic=critic, buffer=buffer,
                    gamma=1.0, compute_chunked_gamma=False, pooled_critic_features=False,
                    batch_size=8, learning_starts=4, num_critics=2,
                    num_critic_updates_per_actor_update=1, num_updates_per_train_step=1,
                    actor_optimizer_lr=1e-4, critic_optimizer_lr=1e-4,
                    ent_coef=0.01, train_actor_with_kl_divergence=False,
                    train_actor_with_entropy=True, train_critic_with_entropy=True,
                    clip_grad_norm=1.0))
    return algorithm


def observation(score, state, fraction_remaining):
    feature = np.asarray(score['feature'], dtype=np.float32)
    feature = feature / max(float(np.sqrt(np.mean(feature ** 2))), 1e-6)
    return {'features': np.concatenate((feature, np.asarray(state, dtype=np.float32),
                                        [fraction_remaining])).astype(np.float32)}


def save(algorithm, path, metadata):
    state = {name: getattr(algorithm, name).state_dict() for name in
             ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer')}
    state.update(metadata=metadata, updates=algorithm._n_updates,
                 replay=algorithm.buffer.get_all_transitions(),
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate())
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load(algorithm, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    for name in ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer'):
        getattr(algorithm, name).load_state_dict(state[name])
    algorithm._n_updates = state['updates']
    algorithm.buffer.clear()
    for transition in state['replay']:
        algorithm.buffer.add(obs=transition.obs, action=transition.action, reward=transition.reward,
                             next_obs=transition.next_obs, done=transition.done, truncated=transition.truncated,
                             episode_id=transition.episode_id, step_in_episode=transition.step_in_episode)
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    return state['metadata']
