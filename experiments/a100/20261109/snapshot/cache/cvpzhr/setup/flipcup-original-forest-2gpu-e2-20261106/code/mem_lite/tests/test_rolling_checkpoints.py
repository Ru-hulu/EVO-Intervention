"""A new checkpoint must be readable and evaluated before the previous one is deleted."""
import json

import pytest
import torch

from mem_lite.periodic_validation import validation_points
from mem_lite.rolling_checkpoint import commit_evaluated_checkpoint, save_rolling_checkpoint


def state(step):
    return {'format':'mem-lite-streamforest-v1', 'step':step, 'config':{},
            'model':{key:{'weight':torch.tensor([float(step)])}
                     for key in ['m3_lora', 'history_projection', 'progress_head']}}


def evaluated(root, step):
    target = root/'validation'/f'step-{step:06d}'
    target.mkdir(parents=True, exist_ok=True)
    (target/'metrics.json').write_text(json.dumps({'step':step, 'checkpoint':str(root/f'step-{step:06d}.pt')}))
    (root/'validation-status.json').write_text(json.dumps({'phase':'completed', 'step':step}))


def test_previous_survives_until_new_evaluation_finishes(tmp_path):
    base = tmp_path/'base.pt'
    base.write_bytes(b'initialization must remain unchanged')
    save_rolling_checkpoint(state(1), tmp_path, 'step-000001.pt')
    evaluated(tmp_path, 1)
    commit_evaluated_checkpoint(tmp_path, 'step-000001.pt', 1)
    save_rolling_checkpoint(state(2), tmp_path, 'step-000002.pt')
    assert (tmp_path/'step-000001.pt').exists()
    assert (tmp_path/'last.pt').resolve() == tmp_path/'step-000001.pt'
    with pytest.raises(FileNotFoundError):
        commit_evaluated_checkpoint(tmp_path, 'step-000002.pt', 2)
    assert (tmp_path/'step-000001.pt').exists()
    evaluated(tmp_path, 2)
    commit_evaluated_checkpoint(tmp_path, 'step-000002.pt', 2)
    assert (tmp_path/'last.pt').is_symlink()
    assert (tmp_path/'last.pt').resolve() == tmp_path/'step-000002.pt'
    assert not (tmp_path/'step-000001.pt').exists()
    assert torch.load(tmp_path/'last.pt', weights_only=True)['step'] == 2
    assert base.read_bytes() == b'initialization must remain unchanged'
    assert (tmp_path/'validation/step-000001/metrics.json').exists()


def test_failed_new_write_does_not_delete_last_good_checkpoint(tmp_path, monkeypatch):
    save_rolling_checkpoint(state(1), tmp_path, 'step-000001.pt')
    evaluated(tmp_path, 1)
    commit_evaluated_checkpoint(tmp_path, 'step-000001.pt', 1)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        save_rolling_checkpoint(state(2), tmp_path, 'step-000002.pt')
    assert (tmp_path/'step-000001.pt').exists()
    assert (tmp_path/'last.pt').resolve() == tmp_path/'step-000001.pt'


def test_incomplete_validation_and_external_paths_cannot_trigger_pruning(tmp_path):
    save_rolling_checkpoint(state(1), tmp_path, 'step-000001.pt')
    evaluated(tmp_path, 1)
    commit_evaluated_checkpoint(tmp_path, 'step-000001.pt', 1)
    save_rolling_checkpoint(state(2), tmp_path, 'step-000002.pt')
    evaluated(tmp_path, 2)
    (tmp_path/'validation-status.json').write_text(json.dumps({'phase':'running', 'step':2}))
    with pytest.raises(ValueError, match='not completed'):
        commit_evaluated_checkpoint(tmp_path, 'step-000002.pt', 2)
    with pytest.raises(ValueError, match='filename'):
        save_rolling_checkpoint(state(2), tmp_path, '../step-000002.pt')
    assert (tmp_path/'step-000001.pt').exists()


def test_half_epoch_schedule_has_four_evaluations_in_two_epochs():
    points = validation_points(1241, .5)
    assert points == {621:.5, 1241:1.0}
    assert [epoch*1241+step for epoch in range(2) for step in points] == [621,1241,1862,2482]
