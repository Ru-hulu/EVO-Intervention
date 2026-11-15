import hashlib
import json
import math
from pathlib import Path
from experiments.soc_grm_validation_20261115.common import read, rows, sha, save, freeze, resolve, SOURCE, SELECTION, EXPECTED
from experiments.forest_soc84_baseline_20261115.run import model_input
from experiments.soc_disrupted_eval_20261115.protocol import image_slots

ROOT=Path('/cache/cvpzhr/setup/soc-five-baseline-20261115')
OUT=Path('/cache/cvpzhr/evaluations/soc-five-baseline-20261115')
FOREST=Path('/cache/cvpzhr/setup/forest-soc84-baseline-20261115')
FOREST_OUT=Path('/cache/cvpzhr/evaluations/forest-soc84-baseline-20261115')
GRM=Path('/cache/cvpzhr/datasets/soc-grm-validation-5fps-20261115')
PYTHON='/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
MODELS=('robobrain25-4b','dopamine-3b','procvlm-2b','robometer-4b','forest')
SUITES=('State','Order','Count')
REPLICAS={'robobrain25-4b':2,'dopamine-3b':2,'procvlm-2b':4,'robometer-4b':2}
PACKAGE='experiments.soc_five_baseline_20261115'


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def result_path(model,key):return OUT/model/'episodes'/(key.replace('/','__')+'.json')


def partition(episodes,n):
    parts=[[] for _ in range(n)];loads=[0]*n
    for r in sorted(episodes,key=lambda r:(-len(r['frame_indices']),r['key'])):
        i=min(range(n),key=lambda i:loads[i]);parts[i].append(r['key']);loads[i]+=len(r['frame_indices'])
    return parts


def slots(model,step,total):
    if model in ('robobrain25-4b','dopamine-3b'):
        return [(0,0),(total-1,0),(0,0),(0,1),(0,2),(step,0),(step,1),(step,2)]
    return image_slots(dict(images=[None]*total),step,model)


def identity(inputs,row,model):
    return dict(inputs_sha256=sha(ROOT/'inputs.json'),model=model,suite=row['suite'],key=row['key'],
        checkpoint_sha256=inputs['registry'][f'{model}-{row["suite"]}']['checkpoint_sha256'],condition='baseline_memory_5fps')


def validate(record,row,expected,complete=False):
    assert record['identity']==expected
    n=len(record['predictions']);assert n<=len(row['frame_indices'])
    if complete:assert n==len(row['frame_indices']) and record['status']=='completed'
    for k in ('frame_indices','timestamps','target_pt'):assert record[k]==row[k][:n]
    assert len(record['raw_responses'])==len(record['input_audits'])==n
    assert record['source_video_sha256']==row['video_hashes'] and record['progress_sha256']==row['progress_sha256']
    assert all(x is None or (type(x) in (float,int) and math.isfinite(x)) for x in record['predictions'])
    a=record['initialization'];assert digest(a)==record['initialization_sha256'] and a['strict_final_state'] and a['epoch']==1
    for k in ('model','suite','checkpoint_sha256'):assert a[k]==expected[k]
    assert a['future_endpoint_reference']==(expected['model'] in ('robobrain25-4b','dopamine-3b'))
    for i,audit in enumerate(record['input_audits']):
        ss=slots(expected['model'],i,len(row['frame_indices']))
        assert audit['sample_index']==i and audit['frame_index']==row['frame_indices'][i]
        assert [tuple(x) for x in audit['slots']]==ss
    return n
