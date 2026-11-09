from pathlib import Path
import json, hashlib, math
import numpy as np
from safetensors.torch import load_file
import torch
S=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def main():
    c=read(S/'training.config.json');o=Path(c['output_dir']);archives=[]
    for epoch in range(1,4):
        step=epoch*c['steps_per_epoch'];p=o/f'checkpoint-{step}'
        receipt=read(p/'complete.json');state=read(p/'trainer_state.json');cfg=read(p/'run_config.json')
        assert receipt['complete'] and receipt['saved_before_validation'] and receipt['step']==step
        assert state['global_step']==step and math.isclose(state['epoch'],epoch,abs_tol=1e-8)
        assert cfg['initialization_policy']=='official_weights' and not cfg.get('initial_checkpoint')
        hashes={}
        for name in receipt['files']:
            f=p/name;assert f.is_file() and f.stat().st_size>0
            h=hashlib.sha256()
            with f.open('rb') as stream:
                for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
            hashes[name]=h.hexdigest()
        for name in ['adapter_model.safetensors','progress_head.safetensors']:
            assert all(torch.isfinite(v).all() for v in load_file(p/name).values())
        archives.append({'epoch':epoch,'step':step,'checkpoint':str(p),'files_sha256':hashes})
    for rank in [0,1]:
        a=read(S/f'initialization-rank{rank}.json')
        assert a['strict_original_weight_load'] and a['official_progress_head_exactly_preserved'] and a['fresh_lora_B_all_zero']
        assert a['initial_checkpoint'] is None and a['official_revision']==c['initial_model_revision']
    report=read(o/'report.json');assert report['status']=='completed' and report['final_checkpoint']==str(o/'checkpoint-1800')
    rows=[json.loads(x) for x in (o/'evaluation/predictions.jsonl').read_text().splitlines()]
    expected=[json.loads(x) for x in Path(c['validation_data']).read_text().splitlines()]
    assert len(rows)==len(expected)==2177
    by={}
    for r,e in zip(rows,expected):
        assert r['id']==e['id'] and r['source_frame_index']==e['source_frame_index'] and abs(r['target_pt']-e['pt'])<1e-6
        assert np.isfinite(r['predicted_pt']) and np.isfinite(r['target_pt']);by.setdefault(r['episode_id'],[]).append(r)
    assert len(by)==7
    correlations=[];undefined=0
    for rr in by.values():
        a=np.array([r['predicted_pt'] for r in rr]);b=np.array([r['target_pt'] for r in rr])
        if np.ptp(a)==0 or np.ptp(b)==0:undefined+=1
        else:correlations.append(float(np.corrcoef(a,b)[0,1]))
    mae=float(np.mean([abs(r['predicted_pt']-r['target_pt']) for r in rows]));corr=float(np.mean(correlations)) if correlations else None
    m=read(o/'evaluation/metrics.json');assert abs(m['eval_mae_pp']-mae*100)<1e-5
    if correlations:assert abs(m['eval_episode_mean_pearson']-corr)<1e-8
    life=[json.loads(x) for x in (o/'lifecycle.jsonl').read_text().splitlines()]
    validation=[r for r in life if r['event']=='validation_started'];assert len(validation)==1 and validation[0]['step']==1800
    assert any(r['event']=='checkpoint_saved' and r['step']==1800 and r['time']<=validation[0]['time'] for r in life)
    result={'checkpoint':report['final_checkpoint'],'checkpoint_epoch':3,'coverage_complete':True,'validation_augmented':False,
            'overall':{'episodes':7,'frames':2177,'mae':mae,'mae_pp':mae*100,'pearson':corr,'pearson_defined_episodes':len(correlations),'pearson_undefined_episodes':undefined},
            'protocol':{'mae':'frame equal','pearson':'episode equal; constants undefined'}}
    (o/'evaluation/verified-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (S/'checkpoint-verification.json').write_text(json.dumps({'status':'passed','archives':archives},indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
