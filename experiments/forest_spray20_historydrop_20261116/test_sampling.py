import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from .prepare import failure_windows, make_draws
from mem_lite.datasets.training_draws import DrawDataset, DrawSampler, load_training_draws


def test_failure_boundaries_include_non_falling_failure_and_union():
    a = dict(fps=30, total_frames=300, stages=[dict(outcome='completed', source_start_frame=0, source_end_frame=50),
        dict(outcome='failed_return_bottle_left_on_table', source_start_frame=40, source_end_frame=59),
        dict(outcome='failed_bottle_fell', source_start_frame=280, source_end_frame=299)])
    windows = failure_windows(a, 2)
    assert [(w['start'], w['end']) for w in windows] == [(0,119),(220,299)]
    with pytest.raises(ValueError): failure_windows(dict(a, stages=[]), 2)


def test_weighted_draw_distribution_and_independent_dropout():
    samples = [dict(episode=str(i//1000), near_anomaly=i < 1000) for i in range(10_002)]
    total, epochs = make_draws(samples, 2, 4, 42, 3, .5)
    assert total == 10_004
    assert (total, epochs) == make_draws(samples, 2, 4, 42, 3, .5)
    assert epochs[0]['draws'] != epochs[1]['draws']
    for epoch in epochs:
        # 10% source frames at 3x weight should occupy about 25% of draws.
        assert .23 < epoch['near_anomaly_draws']/total < .27
        for near in (True, False):
            group = [d for d in epoch['draws'] if samples[d['index']]['near_anomaly'] == near]
            assert .46 < sum(d['drop_history'] for d in group)/len(group) < .54


def test_four_ranks_cover_exact_frozen_draws_and_resume_suffix():
    samples = [dict(episode='e', near_anomaly=i%2 == 0) for i in range(22)]
    total, epochs = make_draws(samples, 2, 4, 42, 3, .5)
    plan = dict(samples=samples, epochs=epochs, world_size=4, draws_per_epoch=total)
    samplers = [DrawSampler(plan, rank) for rank in range(4)]
    for epoch in (0, 1):
        for sampler in samplers: sampler.set_epoch(epoch)
        shards = [list(s) for s in samplers]
        assert all(len(s) == total//4 for s in shards)
        recovered = [x for row in zip(*shards) for x in row]
        assert [(i,d) for i,d,_ in recovered] == [(d['index'],d['drop_history']) for d in epochs[epoch]['draws']]
        assert list(samplers[2])[2:] == shards[2][2:]


def test_fivefold_weight_increases_event_draws_without_changing_dropout_budget():
    samples = [dict(episode='e', near_anomaly=i < 1000) for i in range(10_002)]
    total, three = make_draws(samples, 2, 4, 42, 3, .5)
    _, five = make_draws(samples, 2, 4, 42, 5, .5)
    for a,b in zip(three,five):
        assert .34 < b['near_anomaly_draws']/total < .38
        assert b['near_anomaly_draws'] > a['near_anomaly_draws']
        assert [d['drop_history'] for d in a['draws']] == [d['drop_history'] for d in b['draws']]


class FakeDataset:
    signature, history_dim, history_capacity, history_source = 'signature', 2, 2, {}
    episodes = [(Path('/cache/train/episode'), {})]
    samples = [(0, dict(frame_index=6), None)]
    row = dict(history_tokens=object(), images=[object(),object(),object()], target_pt=.2, task_prompt='goal')
    def __len__(self): return 1
    def __getitem__(self, index): assert index == 0; return self.row


def test_drop_removes_history_before_collation_without_mutating_source():
    base = FakeDataset(); dataset = DrawDataset(base)
    drop = dataset[(0, True, True)]; keep = dataset[(0, False, True)]
    assert drop['history_tokens'] is None
    assert keep['history_tokens'] is base.row['history_tokens']
    assert base.row['history_tokens'] is not None
    assert drop['images'] is base.row['images'] and drop['target_pt'] == .2 and drop['task_prompt'] == 'goal'
    assert drop['sampling_audit'] == [1,1,1,1] and keep['sampling_audit'] == [1,0,1,0]


def test_plan_hash_world_and_sample_order_are_bound(tmp_path):
    path = tmp_path/'plan.json'
    plan = dict(format='forest-training-draws-v1', world_size=4, draws_per_epoch=4, data_signature='signature',
        samples=[dict(episode='episode', frame_index=6, near_anomaly=True)],
        epochs=[dict(draws=[dict(index=0,drop_history=True)]*4)])
    def config():
        path.write_text(json.dumps(plan))
        return dict(epochs=1, training_draws=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    c = config(); load_training_draws(FakeDataset(), c, 4, 0)
    with pytest.raises(ValueError): load_training_draws(FakeDataset(), c, 2, 0)
    path.write_text('{}')
    with pytest.raises(ValueError): load_training_draws(FakeDataset(), c, 4, 0)
    plan['samples'][0]['frame_index'] = 7
    with pytest.raises(ValueError): load_training_draws(FakeDataset(), config(), 4, 0)
