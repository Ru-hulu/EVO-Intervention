from copy import deepcopy
import pytest
from experiments.soc_disrupted_eval_20261115.test_disrupted import fixture, info, FakeBackend
from experiments.soc_disrupted_eval_20261115.common import identity, read, save, digest
from experiments.soc_disrupted_eval_20261115.engine import run_episode as native_episode, SafetyPause
from .engine import run_episode, validate_execution, VECTORS
from .tune import compare
from .memo import Memoized
from experiments.soc_disrupted_eval_20261115.protocol import model_episode,image_slots


class BatchOracle(FakeBackend):
    def predict_many(self, steps):return [self.predict(i) for i in steps]


@pytest.mark.parametrize('model',['forest','procvlm-2b','robometer-4b'])
@pytest.mark.parametrize('kind',['static_only','rewind_only'])
def test_atomic_batch_pause_resume_matches_sequential_state_and_outputs(model,kind,tmp_path):
    row,entry=fixture(kind);ident=identity('source',info(model),row,entry)
    reference=native_episode(FakeBackend(model),row,entry,ident,tmp_path/'native.json')
    backend=BatchOracle(model);destination=tmp_path/'optimized.json';execution={'batch_size':8,'receipt':'test'}
    def stop():
        if len(backend.predict_calls)>=24:raise SafetyPause()
    with pytest.raises(SafetyPause):
        run_episode(backend,row,entry,ident,destination,execution,stop)
    partial=read(destination.with_suffix('.partial.json'))
    assert len(partial['predictions'])==24
    resumed=run_episode(BatchOracle(model),row,entry,ident,destination,execution)
    for k in VECTORS:
        assert digest(resumed[k])==digest(reference[k])
        assert digest(resumed[k][:24])==digest(partial[k])
    validate_execution(resumed,execution)
    assert resumed['metrics']==reference['metrics']


def test_existing_native_completed_file_is_not_rewritten(tmp_path):
    row,entry=fixture();ident=identity('source',info('forest'),row,entry);path=tmp_path/'a.json'
    native_episode(FakeBackend('forest'),row,entry,ident,path);before=path.read_bytes()
    backend=BatchOracle('forest');run_episode(backend,row,entry,ident,path,{'batch_size':16})
    assert path.read_bytes()==before and not backend.predict_calls


def test_mixed_native_prefix_and_batched_tail_preserve_all_original_fields(tmp_path):
    row,entry=fixture();ident=identity('source',info('forest'),row,entry)
    reference=native_episode(FakeBackend('forest'),row,entry,ident,tmp_path/'baseline.json')
    prefix=deepcopy(reference);prefix.pop('metrics');prefix['status']='partial'
    for k in VECTORS:prefix[k]=prefix[k][:27]
    path=tmp_path/'mixed.json';save(path.with_suffix('.partial.json'),prefix)
    execution={'batch_size':4};result=run_episode(BatchOracle('forest'),row,entry,ident,path,execution)
    for k in VECTORS:assert digest(result[k])==digest(reference[k])
    validate_execution(result,execution)
    assert result['execution_segments'][0]['end']==27


def test_equivalence_rejects_invalid_masks_inputs_and_large_numeric_error():
    a=[{'prediction':.3,'input_audit':{'slots':[1,2]}}]
    assert compare(a,a)['max_abs_error']==0
    b=deepcopy(a);b[0]['prediction']=None
    assert not compare(a,b)['invalid_positions_equal']
    b=deepcopy(a);b[0]['input_audit']['slots']=[2,3]
    assert not compare(a,b)['input_equivalence_verified']
    b=deepcopy(a);b[0]['prediction']=.31
    assert compare(a,b)['max_abs_error']>.002


def test_execution_audit_rejects_wrong_batch_or_missing_steps(tmp_path):
    row,entry=fixture();ident=identity('source',info('forest'),row,entry)
    execution={'batch_size':8};result=run_episode(BatchOracle('forest'),row,entry,ident,tmp_path/'a.json',execution)
    wrong=deepcopy(result);wrong['execution_segments'][0]['end']-=1
    with pytest.raises(AssertionError):validate_execution(wrong,execution)
    with pytest.raises(AssertionError):validate_execution(result,{'batch_size':16})


def test_cooperative_pause_flushes_short_uncommitted_write_group(tmp_path):
    row,entry=fixture();ident=identity('source',info('forest'),row,entry);backend=BatchOracle('forest')
    path=tmp_path/'paused.json'
    def check():
        if len(backend.predict_calls)==3:raise SafetyPause()
    with pytest.raises(SafetyPause):run_episode(backend,row,entry,ident,path,{'batch_size':1},check)
    assert len(read(path.with_suffix('.partial.json'))['predictions'])==3


class PureVisualOracle(BatchOracle):
    batch_size=1
    def predict_many(self,steps):
        result=super().predict_many(steps)
        for step,item in zip(steps,result):
            value=int(digest((self.episode['task'],[self.episode['images'][i][v]
                for i,v in image_slots(self.episode,step,self.name)]))[:8],16)%1000/1000
            item.update(prediction=value,raw_response=str(value))
        return result


@pytest.mark.parametrize('model',['procvlm-2b','robometer-4b'])
def test_memo_requires_identical_task_and_every_history_image_and_keeps_audits(model):
    row,entry=fixture('static_only');ep=model_episode(row,entry)
    native=PureVisualOracle(model);native.begin(ep)
    reference=[native.predict_many([i])[0] for i in range(len(ep['images']))]
    cached=Memoized(PureVisualOracle(model),True);cached.begin(ep)
    first=[cached.predict_many([i])[0] for i in range(len(ep['images']))]
    misses=cached.misses
    cached.begin(ep);second=[cached.predict_many([i])[0] for i in range(len(ep['images']))]
    assert first==reference==second and cached.misses==misses
    changed=deepcopy(ep);changed['task']='Different instruction';cached.begin(changed);cached.predict_many([0])
    assert cached.misses==misses+1
    changed['images'][0][0]='changed-main.png';cached.begin(changed);cached.predict_many([0])
    assert cached.misses==misses+2


def test_memo_rejects_stateful_forest():
    with pytest.raises(AssertionError):Memoized(PureVisualOracle('forest'),True)
