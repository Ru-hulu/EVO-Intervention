from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.prm_five_groups_20261114 import experiment_inputs as inputs
from experiments.prm_five_groups_20261114 import experiment_report as report
from experiments.prm_five_groups_20261114.experiment_backends import robometer_paths, playback_positions, ForestBackend, OfficialBackend
from experiments.prm_five_groups_20261114.experiment_worker import execute_episode, check_prefix, name_for
from experiments.prm_five_groups_20261114.followup_schedule import planned_followups, followup_ready, verify_official_handoff, OFFICIAL
from experiments.prm_five_groups_20261114.recover_metrics import score_recover, validate_events
from experiments.prm_five_groups_20261114.prepare_cache import save, sha
from experiments.prm_five_groups_20261114.stream_manifests import playback_map, fps_map, digest
from experiments.prm_five_groups_20261114.test_stream_runtime import source


def test_official_fpl_ties_no_descent_and_rzero_exclusion():
    event = dict(success=True, failure_onset_frame=1, trr_role='r_positive', abnormal_plateaus=[])
    score = score_recover([0,1,1,.2,.2,.9], [0,.2,.4,.6,.8,1], list(range(6)), list(range(6)), 6, event)
    assert score['fpl_details']['predicted_onset_frame'] == 2
    assert score['fpl'] == pytest.approx(.2)
    score = score_recover([.5]*6, [0]*6, list(range(6)), list(range(6)), 6, event)
    assert score['fpl'] == .8 and score['fpl_details']['no_descent_penalty']
    event['trr_role'] = 'r_zero'
    assert score_recover([0]*6,[0]*6,list(range(6)),list(range(6)),6,event)['metric_reasons']['fpl'] == 'official_r_zero_exclusion'


def test_success_endpoint_is_duration_based_and_invalid_segments_never_imputed():
    event = dict(success=True, failure_onset_frame=1, trr_role='r_positive',
                 abnormal_plateaus=[dict(cause='repeated_action',start_frame=1,end_frame=3)])
    score = score_recover([None,.2,.3,.4,.6,.8], [0,.2,.2,.2,.7,1], list(range(6)), [0,20,40,98,99,100], 6, event)
    assert score['successful_endpoint_score'] == pytest.approx(.6)
    assert score['fpl'] is None
    assert score['abnormal_plateau_absolute_mae'] == pytest.approx(.1)
    assert score['abnormal_plateau_drift_mae'] == pytest.approx(.1)
    score = score_recover([0,.2,None,.4,.6,.8], [0,.2,.2,.2,.7,1], list(range(6)), [0,20,40,98,99,100], 6, event)
    assert score['abnormal_plateau_absolute_mae'] is None
    assert score['plateau_segments'][0]['reason'] == 'invalid_predictions'
    assert score_recover([1]*6,[1]*6,list(range(6)),list(range(6)),6,None)['successful_endpoint_score'] is None


def test_event_provenance_rejects_waiting_nonflat_and_overlap(tmp_path):
    evidence = tmp_path/'source.json'; save(evidence, {'annotated':True})
    row = dict(key='Recover/R01/x',video_hashes=['video'],progress_sha256='pt',frames=6)
    event = dict(schema='prm-recover-events-v1', review_status='verified',key=row['key'],
        video_sha256='video',progress_sha256='pt',total_frames=6,success=True,failure_onset_frame=1,
        trr_role='r_positive',abnormal_plateaus=[dict(start_frame=1,end_frame=3,cause='wrong_action')],
        annotation_provenance=[dict(path=str(evidence),sha256=sha(evidence))])
    validate_events(event,row,dict(enumerate([0,.3,.3,.3,.7,1])))
    for mutation in ('normal_waiting','overlap','nonflat'):
        bad=deepcopy(event); truth=[0,.3,.3,.3,.7,1]
        if mutation=='normal_waiting':bad['abnormal_plateaus'][0]['cause']='normal_waiting'
        elif mutation=='overlap':bad['abnormal_plateaus']*=2
        else:truth[2]=.31
        with pytest.raises(AssertionError):validate_events(bad,row,dict(enumerate(truth)))


@pytest.mark.parametrize('variant',['no_memory','static_only','rewind_only'])
def test_robometer_three_vs_ten_images_and_forest_playback_positions(variant):
    row=source(); entry=playback_map(row,variant,None if variant=='no_memory' else 42)
    root=Path('/pixels'); positions=playback_positions(row,entry)
    assert all(a<b for a,b in zip(positions,positions[1:]))
    for i in range(len(entry['target_pt'])):
        paths=robometer_paths(row,entry,root,i)
        assert len(paths)==(3 if variant=='no_memory' else 10)
        assert paths[-3:]==[root/f'{entry["source_frame_indices"][i]:08d}-{v}.png' for v in range(3)]
        if variant!='no_memory':
            assert all(int(p.name.split('-')[0]) in entry['source_frame_indices'][:i+1] for p in paths[:7])
    if variant=='no_memory':assert positions==row['frame_indices']


def test_forest_restore_advances_memory_without_language_calls(monkeypatch):
    import experiments.prm_five_groups_20261114.experiment_backends as backend_module
    row=source(); entry=playback_map(row,'rewind_only',42); calls=[]
    runtime=SimpleNamespace(reset=lambda:calls.append(('reset',)),
        advance_memory=lambda images,position,key,prompt:calls.append(('memory',position)),
        predict=lambda images,position,key,prompt:dict(prediction=.4,raw_nonfinite=None,history_length=12))
    backend=ForestBackend.__new__(ForestBackend);backend.runtime=runtime
    monkeypatch.setattr(backend_module,'images_at',lambda root,row,index:['pixels',index])
    assert backend.begin(row,entry,Path('/images'),{'predictions':[.1]*22},lambda:False)
    assert [x[0] for x in calls]==['reset']+['memory']*22
    assert [x[1] for x in calls[1:]]==playback_positions(row,entry)[:22]
    assert backend.predict(22,[],[])==(.4,None,12)


def test_native_r2_reset_and_vlac_recurrence():
    row=source(); entry=playback_map(row,'no_memory'); requests=[]
    native=SimpleNamespace(INITIAL_HISTORY='initial',image=SimpleNamespace(cache_clear=lambda:None))
    predictor=SimpleNamespace(infer=lambda batch:(requests.append(batch) or ([.5],['<think>new</think>'])))
    backend=OfficialBackend.__new__(OfficialBackend);backend.native=native;backend.predictor=predictor;backend.kind='r2'
    backend.begin(row,entry,Path('/images'),{},lambda:False);backend.predict(2,[.1,.2],['old','<think>prior</think>'])
    assert requests[-1][0][2]=='initial' and 'target_pt' not in requests[-1][0][0]
    backend.kind='vlac';entry=fps_map(row,dict(enumerate(i/180 for i in range(181))),20)
    backend.begin(row,entry,Path('/images'),{},lambda:False)
    assert backend.predict(0,[],[])==(0.,None,None)
    assert backend.predict(1,[.2],[None])[0]==pytest.approx(.6)


class FakeBackend:
    def __init__(self):self.calls=[];self.rebuilt=[]
    def begin(self,row,entry,root,prefix,pause):
        self.rebuilt=list(range(len(prefix['predictions'])));return not pause()
    def predict(self,i,pred,raw):
        self.calls.append(i);return (None if i==4 else i/100),None,i


@pytest.mark.parametrize('variant',['no_memory','static_only','rewind_only','fps20'])
def test_unified_execution_resume_is_exact_and_completed_files_immutable(tmp_path,variant):
    row=source();entry=(fps_map(row,dict(enumerate(i/180 for i in range(181))),20) if variant=='fps20'
                        else playback_map(row,variant,None if variant=='no_memory' else 42))
    identity=dict(model='forest',model_signature='weights',plan_sha256='plan')
    audit=tmp_path/'audit.json';save(audit,identity)
    result=tmp_path/'result.json';partial=tmp_path/'partial.json';backend=FakeBackend()
    status,_=execute_episode(backend,row,entry,identity,Path('/images'),result,partial,audit,lambda:len(backend.calls)==9)
    assert status=='paused' and not result.exists()
    old=json.loads(partial.read_text());resumed=FakeBackend()
    status,record=execute_episode(resumed,row,entry,identity,Path('/images'),result,partial,audit,lambda:False)
    assert resumed.rebuilt==list(range(9)) and resumed.calls==list(range(9,len(entry['target_pt'])))
    assert record['predictions'][:9]==old['predictions'] and record['predictions'][4] is None
    original=result.read_bytes();unused=FakeBackend()
    assert execute_episode(unused,row,entry,identity,Path('/images'),result,partial,audit,lambda:False)[0]=='existing'
    assert not unused.calls and result.read_bytes()==original
    corrupt=deepcopy(record);corrupt['target_pt'][0]+=.1
    with pytest.raises(AssertionError):check_prefix(corrupt,identity,entry,complete=True)


def test_swapped_instruction_requires_its_own_full_native_truth(tmp_path):
    row=source();row['video_hashes']=['video',None,None]
    evidence=tmp_path/'evidence';evidence.write_text('manual labels')
    label=dict(schema='prm-instruction-pt-v1',review_status='verified',key=row['key'],
        video_sha256='video',task_prompt='Different order',frame_indices=list(range(181)),
        target_pt=[0.]*181,annotation_method='explicit instruction-conditioned annotation',
        annotation_provenance=[dict(path=str(evidence),sha256=sha(evidence))])
    assert inputs.counterfactual_truth(label,row,'Different order')==[0.]*181
    with pytest.raises(AssertionError):inputs.counterfactual_truth(label,row,row['task_prompt'])
    label['frame_indices']=label['frame_indices'][:-1]
    with pytest.raises(AssertionError):inputs.counterfactual_truth(label,row,'Different order')


def test_all_followup_jobs_and_dependency_canary_gates(tmp_path):
    jobs=planned_followups();assert len(jobs)==440
    assert len([j for j in jobs if j['group']==2])==168
    assert len([j for j in jobs if j['group']==3])==28
    assert len([j for j in jobs if j['group']==4])==64
    assert len([j for j in jobs if j['group']==5])==180
    job=next(j for j in jobs if j['model']=='robometer-4b' and j['shard']==0)
    readiness={'conditions':{f'group2/{job["condition"]}':{'status':'ready','manifest_sha256':'manifest'}}}
    assert not followup_ready(job,readiness,False,tmp_path)[0]
    assert followup_ready(job,readiness,True,tmp_path)[0]
    sibling=dict(job,shard=1)
    assert not followup_ready(sibling,readiness,True,tmp_path)[0]
    result=tmp_path/'result';result.write_text('finished real trajectory')
    proof=dict(status='passed',complete_real_episode=True,model=job['model'],group=2,
        condition=job['condition'],suite=None,manifest_sha256='manifest',result_path=str(result),result_sha256=sha(result))
    save(tmp_path/'canaries'/f'{job["canary_name"]}.json',proof)
    assert followup_ready(sibling,readiness,True,tmp_path)[0]
    result.write_text('changed')
    with pytest.raises(AssertionError):followup_ready(sibling,readiness,True,tmp_path)


def test_first_group_handoff_does_not_treat_missing_results_as_done():
    states={f'{m}-group1-{i}':dict(status='completed',input_blockers=[]) for m in OFFICIAL for i in range(4)}
    data={'models':{m:dict(missing_episodes=[],overall=dict(completed_episodes=385,invalid_frames=0)) for m in OFFICIAL}}
    assert len(verify_official_handoff(data,states))==6
    data['models']['robometer-4b']['missing_episodes']=['known-input-gap']
    data['models']['robometer-4b']['overall']['completed_episodes']=384
    with pytest.raises(AssertionError):verify_official_handoff(data,states)
    states['robometer-4b-group1-3']['input_blockers']=[{'key':'known-input-gap'}]
    assert verify_official_handoff(data,states)['robometer-4b']['missing']==['known-input-gap']


def test_report_retains_missing_denominators_and_success_is_not_accuracy():
    rows=[dict(recover=score_recover([.2]*10,[.3]*10,list(range(10)),list(range(10)),10,None))]
    summary=report.event_summary(rows,True)
    assert summary['successful_endpoint_score']['mean'] is None
    assert summary['successful_endpoint_score']['reasons']=={'missing_verified_event_annotation':1}
