import json
from pathlib import Path

import pytest

from experiments.prm_five_groups_20261114 import evaluation_priority as priority
from experiments.prm_five_groups_20261114 import eval_dispatch as dispatch


def test_reservation_requires_explicit_authorization(tmp_path):
    assert not priority.reserved(tmp_path, 2)
    p=tmp_path/'evaluation-priority.json'
    p.write_text(json.dumps({'status':'active','user_authorized':True,'reserved_gpus':[2,3],'paused_suites':['Count']}))
    assert priority.reserved(tmp_path,2) and not priority.reserved(tmp_path,0)
    p.write_text(json.dumps({'status':'active','user_authorized':False}))
    with pytest.raises(AssertionError):priority.policy(tmp_path)


def test_adoption_waits_for_parent_and_all_ranks(monkeypatch,tmp_path):
    live={10,11,12}
    monkeypatch.setattr(priority,'matches',lambda r:r['pid'] in live)
    record={'parent':{'pid':10},'ranks':[{'pid':11},{'pid':12}], 'config':{'output_dir':str(tmp_path)}}
    adopted=priority.AdoptedTraining(record)
    assert adopted.poll() is None
    live.remove(10)
    assert adopted.poll() is None
    live.clear()
    assert adopted.poll()==-1
    (tmp_path/'epoch-01.pt').touch()
    assert adopted.poll()==0  # controller still requires full checkpoint verification
    with pytest.raises(AssertionError):priority.AdoptedTraining(record)


def test_reserved_cuda_proof_requires_fresh_prediction_stable_singleton():
    gate=priority.ReservedEvaluationOwnership()
    ident={'pid':123,'start_ticks':8}
    for t in [100,120,140,160]:
        assert not gate.observe(2,ident,{'status':'running','updated':t,'completed_samples':10},[999],t)
    assert gate.observe(2,ident,{'status':'running','updated':170,'completed_samples':11},[999],170)
    assert not gate.observe(2,ident,{'status':'running','updated':180,'completed_samples':12},[999,888],180)
    assert not gate.observe(2,ident,{'status':'running','updated':190,'completed_samples':13},[999],190)
    assert not gate.observe(2,ident,{'status':'running','updated':190,'completed_samples':13},[999],230)


def test_worker_replacement_resets_reserved_proof():
    gate=priority.ReservedEvaluationOwnership()
    for t in [0,20,40,60]:
        ready=gate.observe(3,{'pid':1,'start_ticks':5},{'status':'running','updated':t,'completed_samples':t},[9],t)
    assert ready
    assert not gate.observe(3,{'pid':1,'start_ticks':6},{'status':'running','updated':70,'completed_samples':71},[9],70)


def test_dispatch_does_not_admit_reserved_card_before_training_exits(monkeypatch):
    monkeypatch.setattr(dispatch,'reserved',lambda base,gpu:gpu==2)
    monkeypatch.setattr(dispatch,'pause_reasons',lambda *args:[])
    training={'updated':100,'groups':[{'gpus':[2,3],'phase':'training'}]}
    class Gate:
        def permits(self,*args):return True
    args=(training,2,{}, {2:[123]}, {'status':'running'},0,110)
    assert not dispatch.eligible_gpu(*args,reservation_ownership=Gate(),minors={2:6})
    training['groups'][0]['phase']='evaluation_reserved'
    assert dispatch.eligible_gpu(*args,reservation_ownership=Gate(),minors={2:6})
    assert not dispatch.eligible_gpu(*args,reservation_ownership=None,minors={2:6})
