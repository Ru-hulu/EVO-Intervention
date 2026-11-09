"""Validate paired predictions and report episode-mean Pearson plus absolute errors."""
import json
from common import *
def validate(records,p):
    expected={(r['episode_id'],r['sample_index']):r for e in p['episodes'] for r in rows(e['timeline'])}
    assert len(records)==len(expected)==p['augmented_samples']
    seen=set()
    for r in records:
        key=r['episode_id'],r['sample_index']
        assert key in expected and key not in seen
        e=expected[key];seen.add(key)
        for k in ('suite','task','source_sample_index','anchor_sample_index','frame_index','kind'):
            assert r[k]==e[k],(key,k)
        assert r['target_pt']==e['pt'] and math.isfinite(r['predicted_pt'])
    assert seen==set(expected)
    order={e['episode_id']:i for i,e in enumerate(p['episodes'])}
    records.sort(key=lambda r:(order[r['episode_id']],r['sample_index']))
    return records
def main():
    p=plan();check_inputs(p)
    result=dict(status='completed',completed_at=now(),plan_sha256=p['plan_sha256'],episodes=417,tasks=47,
                samples=p['augmented_samples'],augmentation=p['contract'],metric=p['metric'],
                undefined_policy=p['undefined_policy'],checkpoint_selection=p['checkpoint_selection'],
                models={})
    for model in ('forest','robometer'):
        records=validate([r for i in range(4) for r in rows(OUT/model/f'predictions-{i}.jsonl')],p)
        result['models'][model]=summarize(records)
        result['models'][model].update(checkpoint=str(F_CKPT if model=='forest' else R_CKPT),epoch=1)
        with (OUT/model/'predictions.jsonl').open('w') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        write(OUT/model/'metrics.json',result['models'][model])
    result['comparison']={}
    for name in ('overall','Count','Order','State'):
        a=result['models']['forest']['overall' if name=='overall' else 'by_suite']
        b=result['models']['robometer']['overall' if name=='overall' else 'by_suite']
        if name!='overall':a,b=a[name],b[name]
        result['comparison'][name]=dict(forest_pearson=a['pearson_episode_mean'],robometer_pearson=b['pearson_episode_mean'],
                                       forest_mae_pp=a['mae_pp'],robometer_mae_pp=b['mae_pp'],episodes=a['episodes'])
    write(OUT/'metrics.json',result)
    lines=['# Forest / Robometer: epoch 1, augmented validation','',
           'Pearson(predicted PT, labeled PT) is computed separately per episode, then averaged equally. MAE is frame weighted.',
           '', '| Suite | Episodes | Forest Pearson | Robometer Pearson | Forest MAE (pp) | Robometer MAE (pp) |',
           '|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'undefined' if v is None else f'{v:.6f}'
    for name,v in result['comparison'].items():
        lines.append('| '+name+' | '+str(v['episodes'])+' | '+' | '.join(fmt(v[k]) for k in
            ('forest_pearson','robometer_pearson','forest_mae_pp','robometer_mae_pp'))+' |')
    lines += ['',f'Shared augmented observations: {p["augmented_samples"]}. Seed: 42. Static and jitter PT remain fixed at their event anchor.',
              'Undefined correlations, their episode IDs, per-task/per-episode results, and augmentation drift are recorded in metrics.json.',
              'Both checkpoints are final epoch 1; the augmented validation set is not used to select checkpoints.','']
    (OUT/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(result['comparison'],ensure_ascii=False))
if __name__=='__main__':main()
