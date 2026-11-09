"""Verify original-policy evaluation bypasses steering and cannot train."""
import importlib
import sys
import types
import pytest


def test_original_evaluation_preserves_policy_calls(monkeypatch):
    monkeypatch.setitem(sys.modules, 'serve_robomme_tcp', types.SimpleNamespace(TCPPolicyServer=object))
    monkeypatch.setitem(sys.modules, 'rpc', types.SimpleNamespace(Client=object))
    module = importlib.import_module('evaluation_service')
    e = module.Evaluator.__new__(module.Evaluator)
    calls=[]
    class Policy:
        def call(self, name, payload):
            calls.append((name,payload))
            return {'ok':True}
    e.policy=Policy()
    e.args=types.SimpleNamespace(evaluation_kind='original')
    e.algorithm=None
    e.pending=None
    e.finished=True
    e.set_mode({'mode':'eval'})
    with pytest.raises(ValueError):e.set_mode({'mode':'train'})
    e.finished=False
    e.chunk=0
    e.episode=0
    e.episode_return=0
    e.episode_started=0
    logs=[]
    e.log=lambda event,**values:logs.append((event,values))
    e.eval({'prompt':'instruction','observation':{'image':'current'}})
    assert calls == [('eval',{'prompt':'instruction','observation':{'image':'current'}})]
    assert 'noise' not in calls[0][1]
    e.add_buffer({'observations':[{'image':'next'}], 'actions':[1], 'success':True,
                  'action_step':1,'step_limit':10,'seed':350000})
    assert calls[1][0]=='add_buffer'  # IMPRINT history still receives executed actions.
    assert e.finished and e.episode_return==1
    assert logs[-1][1]['actor_changed'] is False
    assert logs[-1][1]['updates']==0
