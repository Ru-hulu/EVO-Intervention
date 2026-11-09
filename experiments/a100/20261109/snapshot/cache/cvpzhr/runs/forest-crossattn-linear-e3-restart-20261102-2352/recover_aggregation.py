"""CPU-only aggregation recovery; reuse completed predictions, preserve failure evidence."""
from pathlib import Path
import json, math, struct, time, hashlib
import torch
RUN=Path(__file__).resolve().parent
settings=json.loads((RUN/'settings.json').read_text());OUT=Path(settings['output']);VAL=OUT/'validation-03'
def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write(p,obj):
    tmp=p.with_name(p.name+'.recover.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
key=lambda r:(r['episode_id'],r['frame_index'])
records=lines(VAL/'predictions.jsonl')
parts=[read(VAL/f'metrics-rank{i}.json') for i in range(3)]
shards=[lines(VAL/f'predictions-rank{i}.jsonl') for i in range(3)]
assert records==sorted([r for shard in shards for r in shard],key=lambda r:r['index'])
assert [r['index'] for r in records]==list(range(32164))
assert len({key(r) for r in records})==32164
assert all(len(shard)==part['num_samples'] for shard,part in zip(shards,parts))
expected=[]
for directory in read(Path(settings['validation_files'])):
    p=Path(directory);p=p/'samples.jsonl' if p.is_dir() else p
    expected.extend((str(p.parent.resolve()),r['frame_index'],float(r['pt'])) for r in lines(p))
assert len(expected)==len(records)
assert all((r['episode_id'],r['frame_index'],r['target_pt'])==e for r,e in zip(records,expected))
assert all(math.isfinite(r['predicted_pt']) and 0<=r['target_pt']<=1 for r in records)
prior=lines(Path('/cache/cvpzhr/outputs/forest-proprimem-e3-20261031-0009/validation-03/predictions.jsonl'))
prior_map={key(r):r['target_pt'] for r in prior}
assert len(prior_map)==len(prior)==len(records)
assert all(key(r) in prior_map and f32(r['target_pt'])==prior_map[key(r)] for r in records)
errors=[r['predicted_pt']-prior_map[key(r)] for r in records]
raw_errors=[r['predicted_pt']-r['target_pt'] for r in records]
metrics=dict(mode='stateful_cached_history',num_samples=len(records),loss=math.fsum(p['loss_sum'] for p in parts)/len(records),
    mae_pp=100*math.fsum(map(abs,errors))/len(errors),rmse_pp=100*math.sqrt(math.fsum(e*e for e in errors)/len(errors)),
    bias_pp=100*math.fsum(errors)/len(errors),seconds=max(p['seconds'] for p in parts),
    peak_allocated_gib_per_gpu=[p['peak_allocated_gib'] for p in parts],batch_video_per_gpu=4,sequence_window=8,
    coverage_verified=True,same_samples_and_pt_as_forest=True,target_precision_for_metrics='float32, exactly matching Forest reference and loss targets',
    raw_json_target_metrics=dict(mae_pp=100*math.fsum(map(abs,raw_errors))/len(errors),rmse_pp=100*math.sqrt(math.fsum(e*e for e in raw_errors)/len(errors))),
    recovered_from_completed_shards=True,note='Ordered per-video stateful batch evaluation, not online latency. All predictions reused unchanged; only CPU aggregation recovered.')
rows=lines(OUT/'train.jsonl')
assert [r['step'] for r in rows]==list(range(1,217)) and sum(r['samples'] for r in rows)==379323
assert sorted(p.name for p in OUT.glob('*.pt'))==['last.pt']
checkpoint=torch.load(OUT/'last.pt',map_location='cpu',weights_only=True)
assert checkpoint['format']=='mem-lite-streamforest-crossattn-linear-v1'
assert checkpoint['epoch']==3 and checkpoint['step']==216 and checkpoint['next_batch']==0
del checkpoint
train=dict(first_loss=rows[0]['loss'],last_group_loss=rows[-1]['loss'],weighted_mean_loss=math.fsum(r['loss']*r['samples'] for r in rows)/379323,
    samples_seen=379323,losses_and_gradients_finite=all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows),
    max_grad_norm_before_clip=max(r['grad_norm'] for r in rows),peak_allocated_gib=max(max(r['peak_allocated_gib_per_gpu']) for r in rows),
    peak_reserved_gib=max(max(r['peak_reserved_gib_per_gpu']) for r in rows),logged_group_seconds=math.fsum(r['step_seconds'] for r in rows))
assert train['losses_and_gradients_finite']
state=read(RUN/'status.json')
recovery=dict(recovered_at=time.time(),cause='Exact comparison of raw JSON float64 and Forest float32 target values caused rank0 assertion; peers subsequently timed out at final barrier.',
    target_mismatch_count=sum(r['target_pt']!=prior_map[key(r)] for r in records),
    max_target_difference=max(abs(r['target_pt']-prior_map[key(r)]) for r in records),float32_exact_match_all=True,
    prediction_shards_complete=True,raw_dataset_labels_verified=True,checkpoint_verified=True,no_training_or_inference_rerun=True,
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [VAL/'predictions.jsonl',*(VAL/f'predictions-rank{i}.jsonl' for i in range(3))]})
comparisons={}
for name,directory in [('forest','forest-proprimem-e3-20261031-0009'),('forest_cross','forest-crossattn-v2-e3-20261101-210436')]:
    baseline=read(Path('/cache/cvpzhr/outputs')/directory/'validation-03/metrics.json')
    comparisons[name]=dict(loss=baseline['loss'],mae_pp=baseline['mae_pp'],rmse_pp=baseline['rmse_pp'],
        mae_difference_pp=metrics['mae_pp']-baseline['mae_pp'],rmse_difference_pp=metrics['rmse_pp']-baseline['rmse_pp'])
report=dict(status='completed',aggregation_recovered=True,epochs=3,steps=216,checkpoint=str(OUT/'last.pt'),checkpoint_format='mem-lite-streamforest-crossattn-linear-v1',
    world_size=3,physical_gpus=[1,2,3],video_batch_per_gpu=4,sequence_window=8,gradient_checkpointing=True,
    training_seconds=(OUT/'last.pt').stat().st_mtime-state['training_started_at'],training_seconds_definition='Wall time from training process launch until last.pt write; excludes subsequent checkpoint verification.',
    validation_seconds=metrics['seconds'],validation_seconds_definition='Maximum rank inference-loop duration; excludes model loading, failed barrier wait and CPU recovery.',
    training=train,validation=metrics,source=read(RUN/'source.json'),baseline_comparison=comparisons,aggregation_recovery=recovery,
    comparison_note='Same split and PT at float32 precision. Ordered video groups/state carry change effective batch and update frequency; not a matched training ablation.')
write(RUN/'status.before-aggregation-recovery.json',state)
write(VAL/'metrics.json',metrics);write(OUT/'report.json',report);write(RUN/'aggregation-recovery.json',recovery)
validator=RUN/'validate.py';original=validator.read_text()
old="assert all(key(r) in prior_map and r['target_pt']==prior_map[key(r)] for r in records)"
new="assert all(key(r) in prior_map and float(torch.tensor(r['target_pt'], dtype=torch.float32))==prior_map[key(r)] for r in records)"
assert old in original
(RUN/'validate.before-precision-fix.py').write_text(original)
validator.write_text(original.replace(old,new))
state.update(phase='completed',updated_at=time.time(),completed_at=time.time(),aggregation_recovered=True,original_failure=state.get('error'),training_seconds=report['training_seconds'],validation_seconds=report['validation_seconds'])
state.pop('error',None);write(RUN/'status.json',state)
print(json.dumps(report,ensure_ascii=False,indent=2))
