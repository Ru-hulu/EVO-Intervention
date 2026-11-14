"""CPU-only integration of persistence, auditing, dependency and handoff boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.prm_five_groups_20261114 import experiment_report as report
from experiments.prm_five_groups_20261114 import experiment_completion as completion
from experiments.prm_five_groups_20261114 import eval_handoff as handoff
from experiments.prm_five_groups_20261114 import controller, followup_schedule
from experiments.prm_five_groups_20261114.prepare_cache import save, sha
from experiments.prm_five_groups_20261114.experiment_worker import execute_episode
from experiments.prm_five_groups_20261114.stream_manifests import fps_map, digest
from experiments.prm_five_groups_20261114.test_stream_runtime import source


class TextBackend:
    def begin(self,row,entry,root,prefix,pause):return True
    def predict(self,i,pred,raw):return (None if i==4 else i/50),'unparsed' if i==4 else str(i/50),None


def test_worker_report_pipeline_rejects_tampering_and_retains_full_vs_common_denominators(tmp_path,monkeypatch):
    output=tmp_path/'output';old=tmp_path/'old';plan=tmp_path/'plan.json'
    save(plan,{'models':[dict(id='procvlm-2b',prior_provenance_reference=dict(path='/official',revision='rev',small_file_sha256={}))]})
    save(old/'model-download-catalog.json',{'procvlm-2b':dict(revision='rev',files=[])})
    monkeypatch.setattr(report,'OLD',old)
    rows={};entries=[]
    for number in range(2):
        row=source();row['key']=f'State/S01/ep{number}';row['suite']='State';row['task_id']='S01'
        video=tmp_path/f'video{number}';video.write_text('synthetic byte provenance; no inference')
        labels=tmp_path/f'labels{number}.csv';labels.write_text('frame_index,pt\n'+''.join(f'{i},{i/180}\n' for i in range(181)))
        row.update(videos=[str(video),None,None],video_hashes=[sha(video),None,None],progress_path=str(labels),progress_sha256=sha(labels))
        entry=fps_map(row,dict(enumerate(i/180 for i in range(181))),5)
        rows[row['key']]=row;entries.append(entry)
    manifest=dict(episodes=entries,expected_episodes=2)
    monkeypatch.setattr(report,'verified_condition',lambda *args:(manifest,rows,'manifest'))
    backend_audit=dict(model='procvlm-2b',native_model_loaded=True,uses_future_endpoint_reference=False,
        checkpoint='/official',official_revision='rev',adapter_loaded=False,official_file_sha256={})
    signature=digest(backend_audit);audit=tmp_path/'audit.json'
    save(audit,dict(backend_audit,model_signature=signature,plan_sha256=sha(plan),manifest_sha256='manifest',
        group=5,condition='fps5',memory_enabled=True,worker_code_sha256='test-worker'))
    paths=[]
    for entry in entries:
        row=rows[entry['key']]
        identity=dict(model='procvlm-2b',group=5,condition='fps5',suite='State',task_id='S01',key=row['key'],
            input_fps=5,task_prompt=row['task_prompt'],plan_sha256=sha(plan),manifest_sha256='manifest',
            map_sha256=entry['map_sha256'],progress_sha256=row['progress_sha256'],source_video_sha256=row['video_hashes'],
            memory_enabled=True,batch=1,memory_fraction=.3,uses_future_endpoint_reference=False,model_signature=signature)
        result,partial=report.paths(output,5,'fps5','procvlm-2b',row['key']);paths.append(result)
        execute_episode(TextBackend(),row,entry,identity,tmp_path,result,partial,audit,lambda:False)
        r=report.aggregate_condition(5,'fps5','procvlm-2b',output=output,plan_path=plan)
        summary=r['by_suite']['State']
        assert summary['coverage_complete'] is (len(paths)==2)
        assert summary['defined_trajectory_means']['pearson'] is None
    assert summary['invalid_frames']==2 and summary['invalid_episodes']==2
    assert summary['common_integer_seconds']['invalid_frames']==0
    assert summary['common_integer_seconds']['defined_trajectory_means']['pearson']==pytest.approx(1.)
    assert len(list(output.rglob('*.svg')))==2 and list(output.rglob('episode-metrics.csv'))
    report.comparison(output);assert (output/'followup-comparison.md').exists()
    record=json.loads(paths[0].read_text());record['identity']['task_prompt']='changed instruction';save(paths[0],record)
    with pytest.raises(AssertionError):report.aggregate_condition(5,'fps5','procvlm-2b',output=output,plan_path=plan)


def test_completion_never_hides_annotation_or_coverage_gaps(monkeypatch):
    monkeypatch.setattr(completion,'CONDITIONS',{3:('recover',)})
    monkeypatch.setattr(completion,'MODELS',('procvlm-2b',))
    training={'training_completed':dict.fromkeys(['State','Order','Count','Recover'],{})}
    states={'one':{'status':'completed'}};readiness={'conditions':{'group3/recover':{'status':'ready'}}}
    reports={'group3/recover/procvlm-2b':dict(coverage_complete=True,by_episode=[
        dict(key='Recover/R01/one',recover=dict(metric_reasons={'fpl':'missing_explicit_failure_onset_or_role'}))])}
    assert not completion.assess(states,training,readiness,reports,True)['all_experiments_verified']
    reports['group3/recover/procvlm-2b']['by_episode'][0]['recover']['metric_reasons']={'fpl':'no_annotated_failure'}
    assert completion.assess(states,training,readiness,reports,True)['all_experiments_verified']
    reports['group3/recover/procvlm-2b']['coverage_complete']=False
    assert not completion.assess(states,training,readiness,reports,True)['all_experiments_verified']


def test_adoption_waits_for_all_recorded_children_and_rejects_pid_reuse(tmp_path,monkeypatch):
    record=dict(receipt=dict(pid=10,name='job'),identity=dict(pid=10,start_ticks=1),
                process_tree=[dict(pid=10,start_ticks=1),dict(pid=11,start_ticks=2)])
    live={(10,1),(11,2)}
    monkeypatch.setattr(handoff,'matches',lambda r,*args:(r['pid'],r['start_ticks']) in live)
    monkeypatch.setattr(handoff,'verify_command',lambda *args:dict(start_ticks=1))
    monkeypatch.setattr(handoff,'descendants',lambda *args:record['process_tree'])
    worker=handoff.AdoptedEvaluation(record,tmp_path)
    save(tmp_path/'job.json',dict(status='completed'))
    assert worker.poll() is None
    live.remove((10,1));assert worker.poll() is None
    live.remove((11,2));live.add((10,999))
    assert worker.poll()==0  # An unrelated reuse is never adopted.
    save(tmp_path/'job.json',dict(status='running'))
    with pytest.raises(AssertionError):handoff.AdoptedEvaluation(record,tmp_path)


def test_count_resume_requires_original_configuration_and_rank_rng(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(controller,'BASE',tmp_path)
    config=dict(data_signature='data',history_source={'native':True},model={},run_protocol={},
        batch_size=1,gradient_accumulation_steps=9,learning_rate=1e-4,epochs=1,seed=42,
        output_dir='/existing',sample_files=[])
    save(tmp_path/'Count/training.config.json',config)
    save(tmp_path/'evaluation/group1-official-handoff.json',{'status':'passed'})
    checkpoint=tmp_path/'count-resume.pt'
    def prepare(rng):
        torch.save(dict(format='mem-lite-streamforest-v1',step=1200,optimizer={'state':{1:{}}},
            scheduler={'last_epoch':1200},rng_by_rank=rng,config=config),checkpoint)
        monkeypatch.setattr(followup_schedule,'COUNT_CHECKPOINT',checkpoint)
        monkeypatch.setattr(followup_schedule,'COUNT_SHA',sha(checkpoint))
        save(tmp_path/'Count/resume-authorization.json',dict(status='authorized_after_group1_official_audit',
            gpus=[2,3],step=1200,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            handoff_sha256=sha(tmp_path/'evaluation/group1-official-handoff.json'),
            config_sha256=sha(tmp_path/'Count/training.config.json')))
    prepare([{},{}]);assert controller.resume_configuration('Count',[2,3])[2]==str(checkpoint)
    with pytest.raises(AssertionError):controller.resume_configuration('Count',[0,1])
    prepare([{}])
    with pytest.raises(AssertionError):controller.resume_configuration('Count',[2,3])
    prepare([{},{}]);save(tmp_path/'Count/training.config.json',dict(config,epochs=2))
    with pytest.raises(AssertionError):controller.resume_configuration('Count',[2,3])


def test_cpu_upgrade_refuses_gpu_mapped_process_or_different_module(tmp_path):
    from experiments.prm_five_groups_20261114.test_gpu_ownership import fake_process
    from experiments.prm_five_groups_20261114.upgrade_controllers import cpu_identity
    module='experiments.prm_five_groups_20261114.eval_dispatch'
    fake_process(tmp_path,10,1,command=b'python\0-m\0'+module.encode()+b'\0run\0')
    assert cpu_identity(10,module,tmp_path)['pid']==10
    with pytest.raises(AssertionError):cpu_identity(10,'unrelated',tmp_path)
    (tmp_path/'10/maps').write_text('mapped /dev/nvidia4\n')
    with pytest.raises(AssertionError):cpu_identity(10,module,tmp_path)
