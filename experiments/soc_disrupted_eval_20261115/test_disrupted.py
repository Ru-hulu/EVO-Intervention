from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import pytest
from .common import digest, identity, read, save, jobs, validate_canary
from .protocol import timeline, model_episode, image_slots, validate_record
from .metrics import cycle_triplets, score
from .engine import run_episode, resume_canary, SafetyPause
from .report import summarize, seed_summary
from .guard import idle_safe, validate_ticket
from .controller import queue
from experiments.prm_five_groups_20261114.stream_manifests import playback_map


def fixture(kind='rewind_only', rate='30'):
    fps=Fraction(rate);frames=[round(i*fps/5) for i in range(50)]
    row=dict(key='State/S01/episode_1',suite='State',task_id='S01',frames=frames[-1]+1,
        source_rate=rate,frame_indices=frames,timestamps=[float(Fraction(i)/fps) for i in frames],
        target_pt=[i/49 for i in range(50)],progress_sha256='label',input_issues=[],
        videos=['a','b',None],video_hashes=['videoa','videob',None],image_root='/images',task_prompt='Stack blocks')
    entry=playback_map(row,kind,42,[dict(after_source_step=20,duration_steps=10),dict(after_source_step=24,duration_steps=15)])
    return row,entry


def info(model):
    return dict(model=model,suite='State',checkpoint_sha256='checkpoint')


class FakeBackend:
    """State-sensitive oracle: replaying images changes state, reading PT cannot."""
    def __init__(self,model):
        self.name=model;self.predict_calls=[];self.rebuild_calls=[];self.starts=0
        self.audit=dict(**info(model),memory_enabled=True,strict_final_state=True,epoch=1,
            loaded_trainable_state_sha256='tensor',future_endpoint_reference=False,previous_text_used=False)
    def begin(self,episode):
        assert set(episode)=={'key','task','positions','images'}
        self.episode=episode;self.state=0;self.starts+=1
    def advance(self,i):
        self.state=(self.state+int(Path(self.episode['images'][i][0]).stem.split('-')[0]))%101
    def rebuild_step(self,i):
        self.rebuild_calls.append(i)
        if self.name=='forest':self.advance(i)
    def predict(self,i):
        self.predict_calls.append(i)
        slots=image_slots(self.episode,i,self.name)
        if self.name=='forest':self.advance(i);value=self.state/101
        else:value=sum(j for j,v in slots)%101/101
        return dict(prediction=value,raw_response=str(value),input_audit=dict(playback_step=i,
            playback_position=self.episode['positions'][i],input_playback_slots=slots,input_images=len(slots)))


@pytest.mark.parametrize('rate',['30','25','50/3','10'])
def test_playback_clock_always_increases_and_preserves_original_clean_positions(rate):
    row,entry=fixture(rate=rate);clock=timeline(row,entry)
    assert clock['playback_positions'][:21]==row['frame_indices'][:21]
    assert all(a<b for a,b in zip(clock['playback_positions'],clock['playback_positions'][1:]))
    assert entry['source_frame_indices'][21]>entry['source_frame_indices'][22]
    assert clock['playback_positions'][21]<clock['playback_positions'][22]
    assert [entry['source_step_indices'][i] for i in clock['original_positions']]==list(range(50))


def test_visual_windows_follow_presented_history_without_future_or_label_inputs():
    row,entry=fixture();e=model_episode(row,entry)
    assert set(e)=={'key','task','images','positions'}
    assert image_slots(e,0,'procvlm-2b')==[(0,0)]*8
    assert image_slots(e,24,'procvlm-2b')==[(i,0) for i in range(17,25)]
    assert len(image_slots(e,24,'robometer-4b'))==10
    assert image_slots(e,24,'robometer-4b')[-3:]==[(24,0),(24,1),(24,2)]
    for m in ('forest','robometer-4b','procvlm-2b'):
        assert all(i<=24 for i,v in image_slots(e,24,m))
    assert e['images'][24][0].endswith(f"{entry['source_frame_indices'][24]:08d}-0.png")


def test_event_local_triplets_handle_overlapping_loops_and_omit_turning_endpoints():
    row,entry=fixture();events=cycle_triplets(row,entry);original=timeline(row,entry)['original_positions']
    assert [len(e['points']) for e in events]==[9,14]
    for event in events:
        for point in event['points']:
            i,j,k=point['prediction_indices'];q=point['source_step']
            assert i==original[q] and i<j<k and q not in event['excluded_turning_source_steps']
            assert [entry['source_step_indices'][s] for s in (i,j,k)]==[q]*3
    a=next(p for p in events[0]['points'] if p['source_step']==15)
    b=next(p for p in events[1]['points'] if p['source_step']==15)
    assert a['prediction_indices'][0]==b['prediction_indices'][0]
    assert a['prediction_indices'][1:]!=b['prediction_indices'][1:]


def test_circle_all_three_differences_divided_by_three_and_pooled_by_timestamps():
    row,entry=fixture();pred=[0.]*len(entry['target_pt'])
    points=cycle_triplets(row,entry)[0]['points']
    i,j,k=points[0]['prediction_indices'];pred[j]=.3;pred[k]=.6
    result=score(pred,row,entry)
    assert result['events'][0]['points'][0]['k']==pytest.approx(.4)
    assert result['circle_consistency']==pytest.approx(.4/23)
    assert result['expected_triplets']==23


@pytest.mark.parametrize('kind',['static_only','rewind_only'])
def test_perfect_source_lookup_has_zero_disruption_error(kind):
    row,entry=fixture(kind);result=score(entry['target_pt'],row,entry)
    assert result['pearson']==pytest.approx(1) and result['spearman']==pytest.approx(1)
    metric='static_mae' if kind=='static_only' else 'circle_consistency'
    assert result[metric]==pytest.approx(0,abs=1e-15)


def test_static_error_is_deviation_from_own_mean_and_equal_segment_weight():
    row,entry=fixture('static_only');pred=[.9]*len(entry['target_pt'])
    a,b=entry['events'];pred[a['first_inserted_step']:a['last_inserted_step']+1]=[0,1]*5
    pred[b['first_inserted_step']:b['last_inserted_step']+1]=[.2]*15
    result=score(pred,row,entry)
    assert result['events'][0]['segment_mean']==.5
    assert result['static_mae']==pytest.approx(.25)
    assert result['events'][0]['prediction_indices'][0]==a['first_inserted_step']
    assert len(result['events'][0]['prediction_indices'])==10


@pytest.mark.parametrize('kind',['static_only','rewind_only'])
def test_invalid_points_are_not_imputed_or_scored_as_valid_subsets(kind):
    row,entry=fixture(kind);pred=list(entry['target_pt']);pred[0]=None
    result=score(pred,row,entry);assert result['pearson'] is None
    metric='static_mae' if kind=='static_only' else 'circle_consistency'
    assert result[metric]==pytest.approx(0,abs=1e-15)
    pred[entry['events'][0]['first_inserted_step']]=None
    result=score(pred,row,entry);assert result[metric] is None and result['invalid_frames']==2


@pytest.mark.parametrize('kind',['static_only','rewind_only'])
def test_constant_curve_has_undefined_correlation_but_defined_disruption_error(kind):
    row,entry=fixture(kind);result=score([.5]*len(entry['target_pt']),row,entry)
    assert result['pearson'] is None and result['correlation_undefined_reason']=='constant_prediction'
    assert result['static_mae' if kind=='static_only' else 'circle_consistency']==0


@pytest.mark.parametrize('model',['forest','procvlm-2b','robometer-4b'])
@pytest.mark.parametrize('kind',['static_only','rewind_only'])
def test_real_engine_interrupt_resume_preserves_prefix_and_memory(model,kind,tmp_path):
    row,entry=fixture(kind);ident=identity('source',info(model),row,entry)
    baseline=run_episode(FakeBackend(model),row,entry,ident,tmp_path/'baseline.json')
    backend=FakeBackend(model);path=tmp_path/'interrupted.json';cut=27
    def pause():
        if len(backend.predict_calls)>=cut:raise SafetyPause()
    with pytest.raises(SafetyPause):run_episode(backend,row,entry,ident,path,pause)
    prefix=read(path.with_suffix('.partial.json'));saved=deepcopy(prefix['predictions'])
    second=FakeBackend(model);resumed=run_episode(second,row,entry,ident,path)
    assert second.predict_calls==list(range(cut,len(entry['target_pt'])))
    assert second.rebuild_calls==list(range(cut))
    assert resumed['predictions']==baseline['predictions'] and resumed['predictions'][:cut]==saved
    assert resumed['raw_responses']==baseline['raw_responses']
    again=run_episode(second,row,entry,ident,path);assert again['predictions']==resumed['predictions']


def test_canary_compares_resumes_in_both_events_and_resets_episode_memory(tmp_path):
    row,entry=fixture();backend=FakeBackend('forest');ident=identity('source',info('forest'),row,entry)
    checks=resume_canary(backend,row,entry,ident,tmp_path)
    assert len(checks)==2 and all(c['max_abs_difference']==0 for c in checks)
    assert backend.starts==3


@pytest.mark.parametrize('field',['target_pt','source_frame_indices','raw_responses','identity','input_audits','initialization'])
def test_record_tampering_rejected(field,tmp_path):
    row,entry=fixture();ident=identity('source',info('forest'),row,entry)
    result=run_episode(FakeBackend('forest'),row,entry,ident,tmp_path/'result.json')
    result=deepcopy(result)
    if field=='identity':result[field]['checkpoint_sha256']='other'
    elif field=='initialization':result[field]['loaded_trainable_state_sha256']='other'
    elif field=='input_audits':result[field][5]['input_playback_slots']=[(20,0)]
    elif field=='raw_responses':result[field].pop()
    else:result[field][0]=999
    with pytest.raises(AssertionError):validate_record(result,row,entry,ident,complete=True)


def test_incomplete_suites_do_not_publish_subset_means_and_seed_summary_needs_three():
    row,entry=fixture();m=score(entry['target_pt'],row,entry);a={'pearson','spearman','circle_consistency'}
    s=summarize([m],2,a);assert not s['complete'] and s['pearson']['defined_trajectory_mean'] is None
    s=summarize([m],1,a);assert s['complete']
    assert seed_summary([s,s],a)['pearson']['mean'] is None
    assert seed_summary([s,s,s],a)['pearson']['mean']==pytest.approx(1)
    assert seed_summary([s,s,s],a)['pearson']['sample_std']==0
    constant=score([.5]*len(entry['target_pt']),row,entry)
    s=summarize([m,constant],2,a);assert s['pearson']['defined']==1 and s['pearson']['undefined']==1


def test_idle_admission_rejects_unknown_contexts_pressure_and_oom():
    sample=dict(gpus={0:dict(temperature=60,total_mib=81920,used_mib=1000)},resident_and_shmem=10,
        nonreclaimable=20,memory_limit=100,under_oom=0,oom_kill=0,disk_free=200*2**30)
    assert idle_safe(sample,{0:[]},0,0)
    assert not idle_safe(sample,{0:[12345]},0,0)
    assert not idle_safe(dict(sample,oom_kill=1),{0:[]},0,0)
    assert not idle_safe(dict(sample,nonreclaimable=90),{0:[]},0,0)


def test_explicit_ticket_and_one_gpu_task_per_model_suite():
    parent=dict(pid=1,start_ticks=2)
    ticket=dict(token='secret',job_id='job',controller=parent,issued=10,inputs_sha256='source')
    validate_ticket(ticket,'job','secret',parent,20)
    with pytest.raises(AssertionError):validate_ticket(ticket,'job','wrong',parent,20)
    with pytest.raises(AssertionError):validate_ticket(ticket,'job','secret',parent,300)
    q=queue({});assert len(q)==len(jobs({}))==9
    assert len({(v['model'],v['suite']) for v in q.values()})==9
    assert all(v['phase']=='suite' and v['shard']==0 and v['shards']==1 for v in q.values())
    audit=dict(checkpoint_sha256='correct',suite='State')
    proof=dict(status='passed',inputs_sha256='source',initialization=audit,gpu_forward_tested=True)
    validate_canary(proof,'source',audit)
    with pytest.raises(AssertionError):validate_canary(proof,'different',audit)
    with pytest.raises(AssertionError):validate_canary(proof,'source',dict(audit,suite='Count'))
    with pytest.raises(AssertionError):validate_canary(dict(proof,gpu_forward_tested=False),'source',audit)


def test_nonfinite_raw_is_retained_as_text_and_invalid_is_not_clipped(tmp_path):
    row,entry=fixture();backend=FakeBackend('forest');original=backend.predict
    def invalid(i):
        value=original(i)
        if i==0:value.update(prediction=float('nan'),raw_response={'logits':[float('nan')]})
        if i==1:value.update(prediction=1.2,raw_response='1.2')
        return value
    backend.predict=invalid
    result=run_episode(backend,row,entry,identity('source',info('forest'),row,entry),tmp_path/'r.json')
    assert result['predictions'][:2]==[None,None]
    assert result['raw_responses'][0]['raw_response']['logits'][0]=={'nonfinite':'nan'}
    assert result['raw_responses'][1]['invalid_prediction']=='1.2'


def test_full_report_audits_completed_records_and_refuses_wrong_prediction(tmp_path,monkeypatch):
    from . import report as reporter
    from .common import CONDITIONS, sha
    from .engine import empty_record
    setup=tmp_path/'setup';out=tmp_path/'out';setup.mkdir()
    row,_=fixture();conditions={};entries={}
    for condition in CONDITIONS:
        kind,seed=condition.split('-seed')
        entry=playback_map(row,kind,int(seed),[dict(after_source_step=20,duration_steps=10),dict(after_source_step=24,duration_steps=15)])
        path=tmp_path/f'{condition}.json';save(path,dict(episodes=[entry]))
        conditions[condition]=dict(path=str(path));entries[condition]=entry
    source=dict(episodes=[row],registry={'forest-State':info('forest')},conditions=conditions,
        expected_curves=6,expected_observations=sum(len(e['target_pt']) for e in entries.values()),protocol={})
    save(setup/'inputs.json',source);inputs_sha=sha(setup/'inputs.json')
    monkeypatch.setattr(reporter,'ROOT',setup);monkeypatch.setattr(reporter,'OUT',out)
    monkeypatch.setattr(reporter,'MODELS',('forest',));monkeypatch.setattr(reporter,'SUITES',('State',))
    monkeypatch.setattr(reporter,'verify_contract',lambda value:None)
    def path_for(model,condition,key):return out/condition/f'{model}.json'
    monkeypatch.setattr(reporter,'result_path',path_for)
    audit=FakeBackend('forest').audit
    save(setup/'canary/forest-State.json',dict(status='passed',gpu_forward_tested=True,inputs_sha256=inputs_sha,initialization=audit))
    for condition,entry in entries.items():
        result=empty_record(row,entry,identity(inputs_sha,info('forest'),row,entry),audit)
        episode=model_episode(row,entry)
        result.update(predictions=entry['target_pt'],raw_responses=[None]*len(entry['target_pt']),
            input_audits=[dict(playback_step=i,playback_position=episode['positions'][i],
                input_playback_slots=image_slots(episode,i,'forest'),input_images=3) for i in range(len(entry['target_pt']))],
            source_frame_indices=entry['source_frame_indices'],playback_timestamps=entry['playback_timestamps'],
            playback_positions=episode['positions'],target_pt=entry['target_pt'],status='completed',metrics=score(entry['target_pt'],row,entry))
        save(path_for('forest',condition,row['key']),result)
    report=reporter.report();assert report['status']=='completed' and report['present_curves']==6
    assert len(report['across_seeds'])==2 and len(report['episodes'])==6
    path=path_for('forest',CONDITIONS[0],row['key']);value=read(path);value['predictions'][0]=.8;save(path,value)
    with pytest.raises(AssertionError):reporter.report()
