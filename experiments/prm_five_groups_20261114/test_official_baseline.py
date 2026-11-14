from pathlib import Path
import pytest
from experiments.prm_five_groups_20261114.official_baseline import (
    native_module,native_episode,input_blockers,next_history,vlac_next)
from experiments.prm_five_groups_20261114 import eval_dispatch as dispatch


def test_native_input_mapping_and_truth_isolation():
    native=native_module()
    row={'task_prompt':'task','videos':['main','left','right'],'frame_indices':list(range(12)),
         'target_pt':[42]*12,'input_issues':[{'view':1,'reason':'view_frame_count_mismatch'}]}
    episode=native_episode(row,Path('/test'),'grm')
    assert set(episode)=={'task','frames'} and '42' not in str(episode)
    fs=episode['frames'];cur=fs[9]['image']
    assert native.images_for('grm',episode,9)==[fs[0]['image'][0],fs[-1]['image'][0],*fs[0]['image'],*cur]
    assert native.images_for('proc',episode,2)==[fs[0]['image'][0]]*6+[fs[1]['image'][0],fs[2]['image'][0]]
    assert native.images_for('r2',episode,9)==[cur[0]]
    assert native.images_for('vlac',episode,9)==[fs[8]['image'][0],cur[0]]
    assert input_blockers(row,'grm') and not input_blockers(row,'proc')
    main=native_episode(row,Path('/test'),'proc')
    assert all(f['image'][1:]==[None,None] for f in main['frames'])


def test_native_parser_and_memory_do_not_fill_failed_outputs():
    native=native_module()
    assert native.parse('<score>-20</score>','score',-100,100)==-.2
    assert native.parse('unknown','progress',0,100) is None
    assert native.parse('<progress>120</progress>','progress',0,100) is None
    assert next_history('<think>new memory</think><answer>20</answer>')=='new memory'
    assert next_history('malformed but retained')=='malformed but retained'
    assert vlac_next(.2,.5)==pytest.approx(.6)
    assert vlac_next(.6,-.5)==pytest.approx(.4)
    assert vlac_next(.2,None) is None and vlac_next(None,.5) is None


def test_dispatch_waits_for_robometer_and_unknown_gpu_processes(monkeypatch):
    monkeypatch.setattr(dispatch,'alive',lambda pid:pid==10)
    monkeypatch.setattr(dispatch,'descends_from',lambda pid,parent:pid==11 and parent==10)
    training={'updated':100,'groups':[{'gpus':[0,1],'phase':'training','pid':10}]}
    sample={'gpus':{0:{'used_mib':12327,'total_mib':81920,'temperature':40}},'memory_limit':100,
            'memory_working_set':30,'resident_and_shmem':10,'under_oom':0,'oom_kill':0,'disk_free':1000*2**30}
    assert not dispatch.eligible_gpu(training,0,sample,{0:[11]},{'status':'paused_by_resource_guard'},0,110)
    done={'status':'completed','pid':99}
    assert dispatch.eligible_gpu(training,0,sample,{0:[11]},done,0,110)
    assert not dispatch.eligible_gpu(training,0,sample,{0:[11,123]},done,0,110)
    assert not dispatch.eligible_gpu(training,0,sample,{0:[11]},done,0,300)
    training['groups'][0]['phase']='cache'
    assert not dispatch.eligible_gpu(training,0,sample,{0:[]},done,0,110)
    training['training_completed']={s:{} for s in ['State','Order','Count','Recover']}
    assert dispatch.eligible_gpu(training,0,sample,{0:[]},done,0,10000)


def test_new_model_must_finish_real_episode_before_replication():
    job=dispatch.spec('vlac-2b',1)
    assert dispatch.canary_ready(dispatch.spec('vlac-2b',0),{})
    assert not dispatch.canary_ready(job,{})
    first={'status':'running','completed_episodes':0}
    states={'vlac-2b-group1-0':first}
    assert not dispatch.canary_ready(job,states)
    first['completed_episodes']=1
    assert dispatch.canary_ready(job,states)
    first['status']='failed'
    assert not dispatch.canary_ready(job,states)


def test_dispatch_plan_keeps_all_baselines_and_gates_each_forest_suite():
    jobs=dispatch.planned_jobs()
    assert len(jobs)==len({j['name'] for j in jobs})==32
    forest=[j for j in jobs if j['model']=='forest']
    assert len(forest)==12 and {j['suite'] for j in forest}=={'State','Order','Count'}
    assert all(not dispatch.checkpoint_ready(j,{'training_completed':{}}) for j in forest)
    state=dispatch.forest_spec('State',1);order=dispatch.forest_spec('Order',1)
    statuses={'forest-state-group1-0':{'status':'running','completed_episodes':1}}
    assert dispatch.canary_ready(state,statuses)
    assert not dispatch.canary_ready(order,statuses)
    assert state['args']==['--suite','State','--shard','1']
    assert all(dispatch.checkpoint_ready(j,{}) for j in jobs if j['model']!='forest')
