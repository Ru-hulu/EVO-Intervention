import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.prm_five_groups_20261114 import state_acceleration as maintenance
from experiments.prm_five_groups_20261114.verify_acceleration import compare


def probe_pair(tmp_path, change=None):
    c=dict(initial_trainable_sha256='same',config={'batch_size':1,'gradient_accumulation_steps':9},
        peak_allocated_gib=10.5,peak_reserved_gib=11.,cases=[dict(index=i,sequence_length=1280,
        pt=torch.tensor([.4]),loss=torch.tensor(2.),gradients={'head':torch.tensor([1.,2.,3.])},
        seconds=2.) for i in range(6)])
    d=copy.deepcopy(c)
    for r in d['cases']:r['seconds']=.5
    if change:change(d)
    torch.save(c,tmp_path/'original.pt');torch.save(d,tmp_path/'accelerated.pt')
    return compare(tmp_path)


def test_equal_native_probe_passes(tmp_path):
    r=probe_pair(tmp_path)
    assert r['status']=='passed' and r['warm_median_speedup']==4


@pytest.mark.parametrize('change',[
    lambda d:d['cases'][0].update(pt=torch.tensor([.41])),
    lambda d:d['cases'][0].update(gradients={'head':torch.tensor([-1.,-2.,-3.])}),
    lambda d:d.update(peak_allocated_gib=17.),
    lambda d:d['cases'][0].update(loss=torch.tensor(2.2)),
])
def test_bad_numerics_or_memory_are_not_accepted(tmp_path,change):
    assert probe_pair(tmp_path,change)['status']=='failed'


def test_config_change_is_rejected(tmp_path):
    with pytest.raises(AssertionError):
        probe_pair(tmp_path,lambda d:d.update(config={'batch_size':3}))


def test_only_live_authorized_state_maintenance_keeps_existing_evaluations(tmp_path,monkeypatch):
    folder=tmp_path/'state-acceleration';folder.mkdir()
    r=dict(status='testing',user_authorized=True,suite='State',gpus=[0,1],supervisor={'pid':7})
    (folder/'request.json').write_text(json.dumps(r))
    monkeypatch.setattr(maintenance,'matches',lambda x:True)
    assert maintenance.permits_existing_evaluation(tmp_path,0)
    assert maintenance.permits_existing_evaluation(tmp_path,1)
    assert not maintenance.permits_existing_evaluation(tmp_path,2)
    monkeypatch.setattr(maintenance,'matches',lambda x:False)
    assert not maintenance.permits_existing_evaluation(tmp_path,0)


def test_maintenance_allows_existing_but_not_new_gpu_jobs(tmp_path,monkeypatch):
    from experiments.prm_five_groups_20261114 import eval_dispatch as dispatch
    monkeypatch.setattr(dispatch,'BASE',tmp_path)
    monkeypatch.setattr(dispatch,'reserved',lambda *_:False)
    monkeypatch.setattr(dispatch,'pause_reasons',lambda *_:[])
    monkeypatch.setattr(maintenance,'permits_existing_evaluation',lambda b,g:g in [0,1])
    training={'updated':10,'groups':[{'suite':'State','gpus':[0,1],'phase':'training_maintenance'}]}
    assert dispatch.permitted_phase(training,0,11)
    assert not dispatch.eligible_gpu(training,0,{}, {},{},0,11)
    training['groups'][0].update(phase='training',pid=99999999)
    assert dispatch.permitted_phase(training,0,11)  # State exited before status refresh.
    assert not dispatch.eligible_gpu(training,0,{}, {},{},0,11)


def resume_fixture(base):
    from experiments.prm_five_groups_20261114.prepare_cache import sha
    (base/'State').mkdir();(base/'state-acceleration').mkdir()
    c=dict(batch_size=1,gradient_accumulation_steps=9,learning_rate=1e-4,
           model={'gradient_checkpointing':True},run_protocol={'gpus':[0,1]})
    config=base/'State/training.config.json';config.write_text(json.dumps(c))
    checkpoint=base/'state-acceleration/state-resume.pt'
    ck=dict(format='mem-lite-streamforest-v1',epoch=0,step=2100,next_batch=18900,
            config=dict(c,total_optimizer_steps=3830),optimizer={'state':{'p':1}},
            scheduler={'last_epoch':2100},rng_by_rank=[{'rank':0},{'rank':1}])
    torch.save(ck,checkpoint)
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(config),
                backend='original',reason='acceleration_validation_failed_restore_original_training'),ck


def test_resume_retains_checkpoint_protocol_and_two_rank_state(tmp_path):
    value,_=resume_fixture(tmp_path)
    _,c,checkpoint,python=maintenance.validate_resume(tmp_path,value)
    assert c['batch_size']==1 and c['gradient_accumulation_steps']==9
    assert checkpoint==value['checkpoint'] and python.endswith('/forest-labels/bin/python')


def test_changed_checkpoint_protocol_is_rejected(tmp_path):
    from experiments.prm_five_groups_20261114.prepare_cache import sha
    value,ck=resume_fixture(tmp_path);ck['config']['learning_rate']=1e-3
    torch.save(ck,value['checkpoint']);value['checkpoint_sha256']=sha(value['checkpoint'])
    with pytest.raises(AssertionError):maintenance.validate_resume(tmp_path,value)


def test_accelerated_resume_requires_all_gpu_gates(tmp_path):
    value,_=resume_fixture(tmp_path);value['backend']='accelerated'
    proof=dict(status='passed',checkpoint_sha256=value['checkpoint_sha256'],
               config_sha256=value['config_sha256'],numerical_passed=True,
               memory_passed=True,two_rank_smoke_passed=False)
    (tmp_path/'state-acceleration/verification.json').write_text(json.dumps(proof))
    with pytest.raises(AssertionError):maintenance.validate_resume(tmp_path,value)
