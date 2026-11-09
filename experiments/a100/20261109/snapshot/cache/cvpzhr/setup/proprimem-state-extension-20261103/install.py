"""Publish verified State tasks and extend the manifest without changing existing splits."""
import collections, hashlib, json, pathlib, shutil, time
import av
CONTROL=pathlib.Path('/cache/cvpzhr/setup/proprimem-state-extension-20261103')
ROOT=pathlib.Path('/cache/EVO-Intervention/datasets/ProPriMem-current').resolve()
STAGE=CONTROL/'staging'
def digest(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,obj):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.state-import.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def write_text(p,text):
    p=pathlib.Path(p);tmp=p.with_name(p.name+'.state-import.tmp');tmp.write_text(text);tmp.replace(p)
report=json.loads((CONTROL/'download-verification.json').read_text())
assert report['status']=='verified' and report['local_relay'] is False
assert str(ROOT)==report['target'] and ROOT.name=='ProPriMem-hf-9afafe07'
assert (ROOT/'manifest.json').read_bytes()==(CONTROL/'before-import/manifest.json').read_bytes(), 'Target manifest changed during download'
old=json.loads((ROOT/'manifest.json').read_text())
upstream=json.loads((CONTROL/'upstream/manifest.json').read_text())
tasks=report['tasks'];new=[r for r in upstream['tasks'] if r['task_path'] in {'State/'+t for t in tasks}]
assert len(new)==9
assert not ({r['task_path'] for r in old['tasks']} & {r['task_path'] for r in new})
inventory=json.loads((CONTROL/'inventory.json').read_text())
items={r['path']:r for r in inventory['files']}
checks=[];videos_checked=0
for task in new:
    rel=task['task_path'];directory=STAGE/rel
    assert directory.is_dir() and not (ROOT/rel).exists()
    metadata=json.loads((directory/'metadata.json').read_text())
    rows=[json.loads(line) for line in (directory/'metadata.jsonl').read_text().splitlines() if line.strip()]
    assert digest(directory/'metadata.jsonl')==task['trajectory_index_sha256']
    assert len(rows)==metadata['trajectory_count']
    ids={r['sample_id'] for r in rows}
    assert len(ids)==len(rows) and ids==set(task['train'])|set(task['val']) and not set(task['train'])&set(task['val'])
    assert all(r['has_annotation'] is False and r['annotation_path'] is None for r in rows)
    for row in rows:
        assert set(row['video_views'])=={'main_view','left_view','right_view'}
        geometries=[]
        for view,info in row['video_views'].items():
            path=pathlib.PurePosixPath(info['video_path'])
            assert not path.is_absolute() and '..' not in path.parts
            f=directory/path;entry=items[str(pathlib.PurePosixPath(rel)/path)]
            assert entry['algorithm']=='sha256' and entry['hash']==info['video_sha256']
            assert f.stat().st_size==entry['size']==info['video_size_bytes']
            with av.open(str(f)) as container:
                stream=container.streams.video[0]
                count=stream.frames
                rate=float(stream.average_rate)
                start=float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.0
                assert count==row['total_frames'] and abs(rate-row['fps'])<1e-6,(str(f),count,row['total_frames'],rate)
                geometries.append((count,rate,start))
            videos_checked+=1
        assert len(set(geometries))==1,(rel,row['sample_id'],geometries)
    checks.append({'task_id':task['task_id'],'task_path':rel,'trajectories':len(rows),'train':len(task['train']),'val':len(task['val']),'video_files':3*len(rows),'stage_annotations':0})
assert sum(x['trajectories'] for x in checks)==323 and videos_checked==969
provenance=ROOT/'provenance/state-extension-20261103';provenance.mkdir(parents=True,exist_ok=False)
for name in ('inventory.json','download-verification.json','metadata-inventory.json'):
    shutil.copy2(CONTROL/name,provenance/name)
shutil.copytree(CONTROL/'before-import',provenance/'previous-root-metadata')
shutil.copytree(CONTROL/'upstream',provenance/'upstream-metadata')
for task in tasks:(STAGE/'State'/task).rename(ROOT/'State'/task)
combined={**old,'tasks':old['tasks']+new}
summary={}
for row in combined['tasks']:
    suite=summary.setdefault(row['suite'],{'tasks':0,'train':0,'val':0})
    suite['tasks']+=1;suite['train']+=len(row['train']);suite['val']+=len(row['val'])
combined['summary']=summary
combined['totals']={'tasks':sum(s['tasks'] for s in summary.values()),'trajectories':sum(s['train']+s['val'] for s in summary.values()),'train':sum(s['train'] for s in summary.values()),'val':sum(s['val'] for s in summary.values())}
assert combined['totals']=={'tasks':47,'trajectories':2336,'train':1863,'val':473}
assert combined['tasks'][:len(old['tasks'])]==old['tasks']
combined['dataset_version']=old['dataset_version']+'+state.b682e418'
combined['split_version']=old['split_version']+'+state.b682e418'
combined['updated_on']='2026-11-03'
combined['local_extensions']=old.get('local_extensions',[])+[{'repository':report['repository'],'revision':report['revision'],'task_ids':[t['task_id'] for t in new],'existing_task_splits_unchanged':True,'stage_annotations_available':False,'provenance':'provenance/state-extension-20261103/import.json'}]
write_json(ROOT/'manifest.json',combined)
text=(CONTROL/'before-import/README.md').read_text()
assert '38 tasks · 2,013 trajectories.' in text and '| [State](State/README.md) | 10 | 618 |' in text
text=text.replace('38 tasks · 2,013 trajectories.','47 tasks · 2,336 trajectories.')
text=text.replace('Each task contains videos/, annotations/, metadata.json (task details), and metadata.jsonl (trajectory index).', 'Each task contains videos/, metadata.json (task details), and metadata.jsonl (trajectory index); annotations/ is present when stage labels are available.')
text=text.replace('| [State](State/README.md) | 10 | 618 |','| [State](State/README.md) | 19 | 941 |')
text=text.replace('Dataset v1.3.0 · Split v1.3.0 · 1,606 train / 407 val.','Dataset v1.3.0+state.b682e418 · Split v1.3.0+state.b682e418 · 1,863 train / 473 val.')
text+='\nLocal extension (2026-11-03): State S11–S19 were imported directly on this server from syding9/ProPriMem-State at b682e418c6aad6d721ddef164da7c79a22d4c67a. These 323 real-robot trajectories have synchronized three-view videos and metadata, with no stage annotations. All existing tasks and split assignments are preserved; new tasks use the upstream frozen splits. See provenance/state-extension-20261103/import.json.\n'
write_text(ROOT/'README.md',text)
write_text(ROOT/'State/README.md',(CONTROL/'upstream/State/README.md').read_text())
result={**report,'status':'installed','installed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'tasks':checks,'state':{'tasks':19,'trajectories':941,'train':750,'val':191},'dataset_totals':combined['totals'],'existing_tasks_and_splits_unchanged':True,'video_headers_and_view_synchronization_verified':videos_checked,'manifest_sha256':digest(ROOT/'manifest.json'),'new_annotations_available':False}
write_json(provenance/'import.json',result);write_json(CONTROL/'installation.json',result)
write_json(CONTROL/'status.json',{'phase':'installed_cleanup_pending','local_relay':False,'installation_report':str(CONTROL/'installation.json')})
print(json.dumps(result,ensure_ascii=False,indent=2))
