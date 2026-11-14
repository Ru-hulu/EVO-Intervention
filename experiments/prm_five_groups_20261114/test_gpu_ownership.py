import copy
import json
from pathlib import Path
import pytest
from experiments.prm_five_groups_20261114.gpu_ownership import GroupOwnership,training_identity,latest_progress


IDENTITY={'suite':'State','gpus':[0,1],'parent':{'pid':10,'start_ticks':30},
          'ranks':[{'pid':11,'start_ticks':31},{'pid':12,'start_ticks':32}]}
PROCESSES={0:[1001],1:[1002]}


def advance(owner,identity=IDENTITY,processes=PROCESSES):
    for t in range(100,161,10):
        owner.observe(identity,{'step':10 if t<160 else 20,'timestamp':90 if t<160 else 155},processes,t)


def test_singletons_require_new_collective_and_continuous_observations():
    o=GroupOwnership();g={'suite':'State','pid':10,'gpus':[0,1]}
    o.observe(IDENTITY,{'step':10,'timestamp':90},PROCESSES,100)
    assert not o.permits(g,0,PROCESSES)
    o.observe(IDENTITY,{'step':20,'timestamp':155},PROCESSES,160)
    assert not o.permits(g,0,PROCESSES) # 60s unobserved gap resets evidence
    advance(o)
    assert o.permits(g,0,PROCESSES) and o.permits(g,1,PROCESSES)


def test_multiple_or_unknown_contexts_are_never_admitted():
    o=GroupOwnership();g={'suite':'State','pid':10,'gpus':[0,1]}
    advance(o,processes={0:[1001,4001],1:[1002]})
    assert not o.permits(g,0,PROCESSES)
    advance(o)
    assert not o.permits(g,0,{0:[1001,9001],1:[1002]})
    assert not o.permits(g,0,{0:[9001],1:[1002]})
    assert o.permits(g,1,{0:[1001,9001],1:[1002]})


def test_pid_reuse_rank_restart_or_context_loss_invalidates_proof():
    o=GroupOwnership();advance(o)
    changed=copy.deepcopy(IDENTITY);changed['ranks'][0]['start_ticks']+=1
    o.observe(changed,{'step':30,'timestamp':170},PROCESSES,180)
    assert not o.proofs
    o=GroupOwnership();advance(o)
    o.observe(IDENTITY,{'step':30,'timestamp':170},{0:[],1:[1002]},180)
    assert not o.proofs


def test_old_logs_and_same_context_on_two_devices_do_not_prove_ddp():
    o=GroupOwnership()
    for t in range(100,181,10):o.observe(IDENTITY,{'step':10,'timestamp':90},PROCESSES,t)
    assert not o.proofs
    advance(o,processes={0:[1001],1:[1001]})
    assert not o.proofs


def fake_process(root,pid,parent,command=b'',env=b'',maps='',start=50):
    p=root/str(pid);p.mkdir();fields=['S',str(parent)]+['0']*17+[str(start)]
    (p/'stat').write_text(f'{pid} (python) '+' '.join(fields))
    (p/'cmdline').write_bytes(command);(p/'environ').write_bytes(env);(p/'maps').write_text(maps)
    (p/'task'/str(pid)).mkdir(parents=True);(p/'task'/str(pid)/'children').write_text('')
    return p


def test_reads_exact_rank_tree_configuration_and_device_minors(tmp_path):
    root=tmp_path/'proc';root.mkdir();base=tmp_path/'setup';group={'suite':'State','gpus':[0,1],'pid':10}
    p=fake_process(root,10,1);(p/'task/10/children').write_text('11 12')
    for rank,pid in enumerate([11,12]):
        fake_process(root,pid,10,b'python\0-m\0mem_lite.train_stage3\0--config\0'+str(base/'State/training.config.json').encode()+b'\0',
                     f'CUDA_VISIBLE_DEVICES=0,1\0LOCAL_RANK={rank}\0RANK={rank}\0WORLD_SIZE=2\0TOKEN=never_record\0'.encode(),
                     f'fake /dev/nvidia{4+rank}\n')
    result=training_identity(group,base,{0:4,1:5},root)
    assert [r['rank'] for r in result['ranks']]==[0,1] and 'never_record' not in json.dumps(result)
    (root/'12/maps').write_text('fake /dev/nvidia6\n')
    with pytest.raises(ValueError):training_identity(group,base,{0:4,1:5},root)


def test_partial_log_line_is_ignored_but_stale_or_noncollective_record_rejected(tmp_path):
    p=tmp_path/'forest-state-qwen35-clip-depth-e1-20261114';p.mkdir();f=p/'train.jsonl'
    row={'kind':'train','epoch':1,'step':20,'timestamp':150,'peak_allocated_gib_per_gpu':[10,10]}
    f.write_text(json.dumps(row)+'\n{"partial":')
    assert latest_progress('State',160,tmp_path)=={'step':20,'timestamp':150}
    with pytest.raises(ValueError):latest_progress('State',1000,tmp_path)
    row['peak_allocated_gib_per_gpu']=[10];f.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):latest_progress('State',160,tmp_path)
