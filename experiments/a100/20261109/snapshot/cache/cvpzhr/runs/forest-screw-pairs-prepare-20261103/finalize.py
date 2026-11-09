import hashlib,json,pathlib,time
r=pathlib.Path('/cache/cvpzhr/runs/forest-screw-pairs-prepare-20261103')
code=pathlib.Path('/cache/cvpzhr/code/forest-screw-pairs-20261103')
cache=pathlib.Path('/cache/cvpzhr/datasets/forest-screw-pairs-5fps-20261103')
prepared=pathlib.Path('/cache/EVO-Intervention/datasets/insert-screw-recover-pairs-20261103')
def digest(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
index=json.loads((prepared/'index-validation.json').read_text())
report=json.loads((cache/'validation-report.json').read_text())
smoke=json.loads((r/'preflight-report.json').read_text())
config=json.loads((r/'finetune-config.json').read_text())
assert report['status']=='verified' and report['pairs']==88 and report['clips']==264
assert {k:v['episodes'] for k,v in report['splits'].items()}=={'train':68,'validation':16,'review':4}
assert report['source_video_leakage'] is False and report['review_excluded']
assert smoke['status']=='passed' and smoke['formal_training_started'] is False and smoke['checkpoint_written'] is False
for row in report['episodes']:
    p=pathlib.Path(row['sample_file'])
    assert digest(p)==row['samples_sha256'] and digest(p.with_name('summary.json'))==row['summary_sha256']
provenance=json.loads((r/'provenance.json').read_text())
assert digest(config['init_checkpoint'])==provenance['baseline_checkpoint_sha256']
files=list(code.rglob('*.py'))+list((code/'mem_lite/configs').glob('*.json'))+[code/'mem_lite/recovery_pairs.md']
provenance['files']={str(p.relative_to(code)):digest(p) for p in sorted(files)}
provenance['run_files']={p.name:digest(p) for p in sorted(r.iterdir()) if p.suffix in ('.py','.sh') or p.name=='finetune-config.json'}
(r/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
ready={'status':'ready','completed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prepared_dir':str(prepared),'cache_dir':str(cache),'cache_report':str(cache/'validation-report.json'),'preflight_report':str(r/'preflight-report.json'),'config':str(r/'finetune-config.json'),'config_sha256':digest(r/'finetune-config.json'),'init_checkpoint':config['init_checkpoint'],'splits':report['splits'],'observations':report['observations'],'history_tensors_checked':report['history_tensors_checked'],'formal_training_started':False,'launch_command':'bash '+str(r/'train.sh'),'learning_rate':config['learning_rate'],'epochs':config['epochs'],'gpu_ids':[1,2,3],'test_result':'14 passed'}
(r/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
(r/'status.json').write_text(json.dumps({'phase':'ready','training_started':False,'ready_report':str(r/'ready.json'),'updated_at':ready['completed_at']},indent=2)+'\n')
print(json.dumps(ready,indent=2))
