import json
import pytest

from experiments.prm_five_groups_20261114 import model_cancellation as cancellation
from experiments.prm_five_groups_20261114 import followup_schedule as schedule
from experiments.prm_five_groups_20261114 import experiment_completion as completion


def test_only_explicit_authorized_cancellation_amends_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(cancellation, 'EVAL', tmp_path)
    path = tmp_path / 'r2vlm-user-cancellation.json'
    assert cancellation.cancelled_models() == set()
    receipt = dict(status='cancelled_by_user', model='r2vlm-alfred-rl',
                   scope='all evaluation groups in this experiment',
                   preserve_existing_predictions=True, user_authorized=False)
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):cancellation.cancelled_models()
    receipt['user_authorized'] = True
    path.write_text(json.dumps(receipt))
    assert cancellation.cancelled_models() == {'r2vlm-alfred-rl'}


def test_cancelled_model_does_not_block_handoff_or_hide_other_missing_results(monkeypatch):
    cancelled = {'r2vlm-alfred-rl'}
    monkeypatch.setattr(schedule, 'cancelled_models', lambda: cancelled)
    states = {f'{m}-group1-{s}': dict(status='completed', input_blockers=[])
              for m in schedule.OFFICIAL for s in range(4)}
    report = {'models': {m: dict(missing_episodes=[], overall=dict(completed_episodes=385, invalid_frames=0))
                         for m in schedule.OFFICIAL}}
    for s in range(4):states[f'r2vlm-alfred-rl-group1-{s}']['status'] = 'cancelled_by_user'
    report['models']['r2vlm-alfred-rl'] = dict(missing_episodes=['unfinished'], overall=dict(completed_episodes=85, invalid_frames=0))
    assert schedule.official_jobs_terminal(states)
    verified = schedule.verify_official_handoff(report, states)
    assert len(verified) == 5 and 'r2vlm-alfred-rl' not in verified
    report['models']['procvlm-2b']['missing_episodes'] = ['unexplained']
    with pytest.raises(AssertionError):schedule.verify_official_handoff(report, states)
    states['procvlm-2b-group1-0']['status'] = 'failed'
    assert not schedule.official_jobs_terminal(states)
    states['procvlm-2b-group1-0']['status'] = 'cancelled_by_user'
    assert not schedule.official_jobs_terminal(states)


def test_cancelled_jobs_are_excluded_but_other_dependencies_remain():
    result = completion.assess(
        {'r2vlm-alfred-rl-group1-0': {'status':'cancelled_by_user'},
         'forest-state-group1-0': {'status':'pending'}},
        {}, {}, {}, False, {'r2vlm-alfred-rl'})
    assert result['unfinished_jobs'] == ['forest-state-group1-0']
    assert not result['all_experiments_verified']
    assert result['missing_reports']
    assert not any(k.endswith('/r2vlm-alfred-rl') for k in result['missing_reports'])
