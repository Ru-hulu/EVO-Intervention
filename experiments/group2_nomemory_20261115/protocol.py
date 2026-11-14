"""Only the current observation crosses the inference boundary; labels stay outside."""
import math
import re
from pathlib import Path
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics

def current_sample(row,index,model):
    f=row['frame_indices'][index];views=[0] if model=='procvlm-2b' else [0,1,2]
    return dict(task=row['task_prompt'],images=[str(Path(row['image_root'])/f'{f:08d}-{v}.png')
        if row['videos'][v] else None for v in views])

def current_robometer_messages(native_messages):
    message=native_messages[0];content=message['content']
    start=next(i for i,c in enumerate(content) if c.get('text','').startswith('Current observation,'))
    selected=[content[0]]+content[start:]
    assert sum(x['type']=='image' for x in selected)==3
    assert sum(x.get('text')=='<|prog_token|>' for x in selected)==1
    assert not any('Historical observations' in x.get('text','') for x in selected)
    return [dict(role=message['role'],content=selected)]

def parse_progress(raw):
    match=re.fullmatch(r'\s*<progress>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*</progress>\s*',raw)
    if not match:return None
    value=float(match[1])/100
    return value if math.isfinite(value) and 0<=value<=1 else None

def endpoint_score(predictions):
    assert predictions
    count=max(1,math.ceil(.02*len(predictions)));tail=predictions[-count:]
    valid=all(x is not None and math.isfinite(x) for x in tail)
    return dict(endpoint_score=math.fsum(tail)/count if valid else None,
        endpoint_indices=list(range(len(predictions)-count,len(predictions))),
        endpoint_reason=None if valid else 'invalid_prediction_in_last_2_percent')

def metrics(predictions,row,success=None):
    out=trajectory_metrics(predictions,row['target_pt'],row['timestamps'])
    out.update(endpoint_score(predictions));out['success']=success
    out['SA']=out['endpoint_score'] if success is True else None
    out['SA_reason']=(out['endpoint_reason'] if success is True else
        'not_successful' if success is False else 'success_eligibility_not_confirmed')
    return out

def validate_record(record,row,identity,complete=False):
    assert record['identity']==identity
    n=len(record['predictions']);assert n<=len(row['frame_indices'])
    if complete:assert n==len(row['frame_indices'])
    assert record['frame_indices']==row['frame_indices'][:n]
    assert record['timestamps']==row['timestamps'][:n] and record['target_pt']==row['target_pt'][:n]
    assert len(record['raw_responses'])==len(record['input_token_counts'])==n
    assert all(x is None or (isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1) for x in record['predictions'])
    assert record['history_length']==0 and record['memory_enabled'] is False
    return n
