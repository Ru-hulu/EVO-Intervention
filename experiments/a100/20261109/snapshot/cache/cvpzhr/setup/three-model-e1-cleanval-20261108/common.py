"""Shared paths, immutable evaluation plan, and per-episode Pearson aggregation."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib, json, math, os
ROOT = Path(__file__).resolve().parent
OUT = Path('/cache/cvpzhr/evaluations/forest-robometer-e1-augval-pearson-20261105')
CACHE = Path('/cache/cvpzhr/datasets/forest-val-static-jitter-seed42-20261105')
BASE = Path('/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-20261103')
RB_CACHE = Path('/cache/cvpzhr/datasets/robometer-proprimem-cap50-5fps-h7-current3-static-jitter-seed42-20261104')
F_CODE = Path('/cache/cvpzhr/setup/forest-original-cap50-train-20261104/code')
R_CODE = Path('/cache/cvpzhr/code/robometer-current3-w8-e1-20261104')
R_SETUP = Path('/cache/cvpzhr/setup/robometer-current3-w8-e1-20261104')
R_OUT = Path('/cache/cvpzhr/outputs/robometer-cap50-h7-current3-static-jitter-e1-20261104')
F_CKPT = OUT/'forest-epoch1.pt'
R_CKPT = R_OUT/'checkpoint-9967'
F_PY = '/cache/cvpzhr/envs/forest/bin/python'
R_PY = '/cache/cvpzhr/envs/robometer/bin/python'
def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def plan(): return read(ROOT/'plan.json')
def assigned(p, rank): return [e for e in p['episodes'] if e['worker']==rank]
def check_inputs(p):
    for path,digest in p['protected_hashes'].items():
        assert sha(path)==digest, 'Input changed: '+path
    assert sha(F_CKPT)==p['forest_checkpoint_sha256']
    for e in p['episodes']:
        assert sha(e['timeline'])==e['timeline_sha256']
        assert sha(e['robometer_rows'])==e['robometer_rows_sha256']
def pearson(x,y):
    if len(x)<2:return None,'fewer_than_two_samples'
    mx=math.fsum(x)/len(x);my=math.fsum(y)/len(y)
    dx=[v-mx for v in x];dy=[v-my for v in y]
    xx=math.fsum(v*v for v in dx);yy=math.fsum(v*v for v in dy)
    if xx==0:return None,'constant_prediction'
    if yy==0:return None,'constant_target'
    value=math.fsum(a*b for a,b in zip(dx,dy))/math.sqrt(xx*yy)
    return max(-1.,min(1.,value)),None
def summarize(records):
    groups=defaultdict(list)
    for r in records:groups[r['episode_id']].append(r)
    per=[]
    for key,rr in groups.items():
        r,reason=pearson([x['predicted_pt'] for x in rr],[x['target_pt'] for x in rr])
        per.append(dict(episode_id=key,suite=rr[0]['suite'],task=rr[0]['task'],
                        samples=len(rr),pearson=r,undefined_reason=reason,
                        mae_pp=100*math.fsum(abs(x['predicted_pt']-x['target_pt']) for x in rr)/len(rr)))
    def group_report(selected,erows):
        valid=[r['pearson'] for r in erows if r['pearson'] is not None]
        # Undefined correlations are reported explicitly, never silently turned into perfect/zero values.
        return dict(samples=len(selected),episodes=len(erows),
            pearson_episode_mean=math.fsum(valid)/len(valid) if valid else None,
            valid_pearson_episodes=len(valid),
            undefined_pearson_episodes=[dict(episode_id=r['episode_id'],reason=r['undefined_reason'])
                                      for r in erows if r['pearson'] is None],
            mae_pp=100*math.fsum(abs(r['predicted_pt']-r['target_pt']) for r in selected)/len(selected))
    overall=group_report(records,per)
    by_suite={s:group_report([r for r in records if r['suite']==s],[r for r in per if r['suite']==s])
              for s in sorted({r['suite'] for r in per})}
    by_task={t:group_report([r for r in records if r['task']==t],[r for r in per if r['task']==t])
             for t in sorted({r['task'] for r in per})}
    originals={(r['episode_id'],r['source_sample_index']):r['predicted_pt']
               for r in records if r['kind']=='original'}
    by_kind={}
    for kind in sorted({r['kind'] for r in records}):
        rr=[r for r in records if r['kind']==kind]
        item=dict(samples=len(rr),mae_pp=100*math.fsum(abs(r['predicted_pt']-r['target_pt']) for r in rr)/len(rr))
        if kind!='original':
            item['anchor_prediction_drift_mae_pp']=100*math.fsum(
                abs(r['predicted_pt']-originals[(r['episode_id'],r['anchor_sample_index'])]) for r in rr)/len(rr)
        by_kind[kind]=item
    return dict(overall=overall,by_suite=by_suite,by_task=by_task,by_episode=per,by_kind=by_kind)
