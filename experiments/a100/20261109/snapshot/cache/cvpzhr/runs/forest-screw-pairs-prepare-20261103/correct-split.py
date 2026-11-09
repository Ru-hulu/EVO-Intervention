"""Correct split membership to match the initialization checkpoint's actual held-out videos."""
import json, pathlib, shutil, time
run=pathlib.Path('/cache/cvpzhr/runs/forest-screw-pairs-prepare-20261103')
prepared=pathlib.Path('/cache/EVO-Intervention/datasets/insert-screw-recover-pairs-20261103')
cache=pathlib.Path('/cache/cvpzhr/datasets/forest-screw-pairs-5fps-20261103')
assert len(list(cache.rglob('summary.json')))==88, 'Wait for all cache workers to finish'
for pidfile in run.glob('cache-scaled-*.pid'):
    pid=int(pidfile.read_text())
    proc=pathlib.Path(f'/proc/{pid}/stat')
    assert not proc.exists() or proc.read_text().split()[2]=='Z', f'Worker still running: {pid}'
backup=run/'split-correction-backup'
backup.mkdir(exist_ok=False)
for name in ('all.jsonl','train.jsonl','validation.jsonl','review.jsonl','manifest.json','index-validation.json'):
    shutil.copy2(prepared/name,backup/name)
rows=[json.loads(line) for line in (prepared/'all.jsonl').read_text().splitlines()]
changes=[]
for row in rows:
    video=row['source_video_id']
    if video not in ('file-017','file-021'):continue
    old=row['split'];new='validation' if video=='file-017' else 'train'
    assert old!=new
    source=cache/old/row['episode_id']
    target=cache/new/row['episode_id']
    saved=json.loads((source/'summary.json').read_text())
    assert saved['source']==row
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    shutil.copy2(source/'summary.json',backup/(row['episode_id'].replace('/','-')+'-summary.json'))
    source.rename(target)
    row['split']=new
    saved['source']=row
    (target/'summary.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
    changes.append({'episode_id':row['episode_id'],'from':old,'to':new})
for split in ('all','train','validation','review'):
    selected=rows if split=='all' else [r for r in rows if r['split']==split]
    (prepared/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
manifest=json.loads((prepared/'manifest.json').read_text())
for source in manifest['sources']:
    if source['video_id']=='file-017':source['split']='validation'
    elif source['video_id']=='file-021':source['split']='train'
manifest['splits']={s:[x['video_id'] for x in manifest['sources'] if x['split']==s] for s in ('train','validation','review')}
manifest['validation_split_basis']='Actual training sample list in the initialization checkpoint configuration; file-017 held out, file-021 trained'
(prepared/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
audit=json.loads((prepared/'index-validation.json').read_text())
assert len({r['source_video_id'] for r in rows if r['split']=='train'} & {r['source_video_id'] for r in rows if r['split']=='validation'})==0
assert all(r['split']=='review' for r in rows if r['source_video_id']=='file-019')
audit['validation_source_videos']=manifest['splits']['validation']
audit['baseline_split_verified']=True
(prepared/'index-validation.json').write_text(json.dumps(audit,indent=2)+'\n')
report={'reason':'file-021 was present in the initialization checkpoint training set; file-017 was held out','changes':changes,'progress_labels_changed':False,'clip_frames_changed':False,'history_tensors_changed':False,'corrected_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(run/'split-correction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
