"""Exercise actual upstream SAC updates and the integration's reward contract."""
import numpy as np
import pytest
import torch

from core import build_algorithm, progress_reward, reward_components, save, load


def test_reward_preserves_failure_recovery_and_zero():
    p = [0, .2, .6, .3, .5, 0]
    rewards = [progress_reward(a, b) for a, b in zip(p, p[1:])]
    assert rewards == pytest.approx([.2, .4, -.3, .2, -.5])
    assert sum(rewards) == pytest.approx(p[-1] - p[0])
    assert progress_reward(.4, .4) == 0
    with pytest.raises(ValueError):
        progress_reward(.5, float('nan'))


def test_upstream_sac_updates_and_roundtrip(tmp_path):
    torch.set_num_threads(2)
    algorithm = build_algorithm(18, 32)
    original = {k: v.clone() for k, v in algorithm.actor.state_dict().items()}
    rng = np.random.default_rng(4)
    for i in range(12):
        algorithm.buffer.add(obs={'features': rng.normal(size=18).astype('float32')},
            action=rng.uniform(-1, 1, 32).astype('float32'), reward=float(rng.uniform(-.2,.2)),
            next_obs={'features': rng.normal(size=18).astype('float32')},
            done=i % 4 == 3, truncated=False, episode_id=i//4, step_in_episode=i%4)
    # Benchmark deadline must actually mask bootstrapping in upstream replay.
    terminal = algorithm.buffer._batch_transitions([algorithm.buffer.get_all_transitions()[3]])
    assert terminal['done'][0]
    metrics = algorithm.train_step(rollout_step=12)
    assert metrics and all(np.isfinite(v) for v in metrics.values() if isinstance(v, (float,int)))
    assert any(not torch.equal(original[k], v) for k, v in algorithm.actor.state_dict().items())
    before = {k:v.clone() for k,v in algorithm.actor.state_dict().items()}
    path = tmp_path/'last.pt'
    save(algorithm, path, {'test':True})
    restored = build_algorithm(18, 32, seed=7)
    assert load(restored, path) == {'test':True}
    assert len(restored.buffer) == 12
    assert all(torch.equal(before[k],v) for k,v in restored.actor.state_dict().items())
    # Resuming loads optimizer state onto the actual restored parameters.
    restored.train_step(rollout_step=13)
    assert any(not torch.equal(before[k],v) for k,v in restored.actor.state_dict().items())


def test_sparse_dense_reward_ablation():
    assert reward_components('terminal', False, False) == (0, 0)
    assert reward_components('terminal', False, True) == (0, 0)
    assert reward_components('terminal', True, True) == (1, 0)
    assert reward_components('terminal_progress', False, False, .7, .3) == pytest.approx((0, -.4))
    assert reward_components('terminal_progress', True, True, .9, 1) == pytest.approx((1, .1))


def test_simulator_entry_is_safe_for_spawn():
    import runpy
    from pathlib import Path
    # Spawned CuRobo planners import __mp_main__; they must not rerun the evaluator.
    runpy.run_path(str(Path(__file__).with_name('sim_entry.py')), run_name='__mp_main__')
