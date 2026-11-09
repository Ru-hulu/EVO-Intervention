"""Retire exactly the two user-approved old raw-data roots, preserving caches and results."""
import collections, hashlib, json, os, pathlib, shutil, time
CONTROL=pathlib.Path('/cache/cvpzhr/setup/proprimem-state-extension-20261103')
CURRENT=pathlib.Path('/cache/EVO-Intervention/datasets/ProPriMem-current')
ROOT=CURRENT.resolve()
RETIRED=[pathlib.Path('/cache/EVO-Intervention/datasets/ProPriMem'),pathlib.Path('/cache/cvpzhr/datasets/ProPriMem')]
def hashfile(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
installation=json.loads((CONTROL/'installation.json').read_text())
assert installation['status']=='installed' and installation['video_headers_and_view_synchronization_verified']==969
assert CURRENT.is_symlink() and str(ROOT)==installation['target']
assert hashfile(ROOT/'manifest.json')==installation['manifest_sha256']
for task in installation['tasks']:assert (ROOT/task['task_path']/'metadata.jsonl').exists()
cache_status={}
for name in ('forest-proprimem-5fps-20261030','forest-proprimem-5fps-repeat5-seed42-20261031','forest-screw-pairs-5fps-20261103'):
    cache=pathlib.Path('/cache/cvpzhr/datasets')/name
    counts={}
    for split in ('train','validation','review'):
        listing=cache/f'{split}.sample_files.json'
        if not listing.exists():continue
        samples=json.loads(listing.read_text())
        for sample in samples:
            p=pathlib.Path(sample);p=p if p.is_dir() else p.parent
            assert (p/'summary.json').is_file() and (p/'samples.jsonl').is_file()
            assert all(not p.is_relative_to(old) for old in RETIRED)
        counts[split]=len(samples)
    cache_status[name]=counts
plan=pathlib.Path('/cache/cvpzhr/setup/screw-mixed-aug-20261103/training-plan.json')
if plan.exists():assert not any(str(old)+'/' in plan.read_text() for old in RETIRED)
open_files=[]
for process in pathlib.Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:
        for fd in (process/'fd').iterdir():
            try:path=os.readlink(fd)
            except (OSError,PermissionError):continue
            if any(path.startswith(str(old)+'/') for old in RETIRED):open_files.append({'pid':int(process.name),'path':path})
    except (OSError,PermissionError):pass
assert not open_files,open_files
archive=ROOT/'provenance/state-extension-20261103/retired-snapshot-metadata'
archive.mkdir(exist_ok=False)
groups={};snapshots=[]
for old,expected,label in zip(RETIRED,(2108,816),('old-full','old-cap30'),strict=True):
    assert old.is_dir() and not old.is_symlink() and old.resolve()==old
    assert ROOT!=old and not ROOT.is_relative_to(old)
    manifest=json.loads((old/'manifest.json').read_text())
    assert manifest['totals']['trajectories']==expected
    allowed={'Count','Order','State','README.md','manifest.json'}
    if label=='old-cap30':allowed|={'.gitattributes','insert-screw','insert-screw-recover'}
    assert {p.name for p in old.iterdir()}==allowed
    target=archive/label;target.mkdir()
    for name in ('manifest.json','README.md','.gitattributes'):
        if (old/name).exists():shutil.copy2(old/name,target/name)
    count=total=0
    with (target/'removed-files.jsonl').open('w') as audit:
        for p in sorted(old.rglob('*')):
            if not p.is_file():continue
            assert not p.is_symlink(),str(p)
            s=p.stat();count+=1;total+=s.st_size
            key=(s.st_dev,s.st_ino)
            group=groups.setdefault(key,{'removed_links':0,'links':s.st_nlink,'allocated_bytes':s.st_blocks*512})
            group['removed_links']+=1
            audit.write(json.dumps({'path':str(p.relative_to(old)),'bytes':s.st_size,'inode':s.st_ino,'links_before':s.st_nlink},ensure_ascii=False)+'\n')
    snapshots.append({'path':str(old),'files':count,'logical_bytes':total,'manifest_sha256':hashfile(old/'manifest.json')})
historical_labels=[]
for p in pathlib.Path('/cache/EVO-Intervention/outputs/proprimem/recovery-labels-20261102').rglob('*.json'):
    if p.stat().st_size>3000000:continue
    text=p.read_text()
    if any(str(old)+'/' in text for old in RETIRED):historical_labels.append(str(p))
receipt={'authorized_scope':'Remove the two old ProPriMem raw-data directories named in the conversation; retain model caches, labels, checkpoints and experiment outputs','snapshots':snapshots,'estimated_reclaimable_file_bytes':sum(g['allocated_bytes'] for g in groups.values() if g['links']==g['removed_links']),'shared_inodes_remaining_elsewhere':sum(g['links']>g['removed_links'] for g in groups.values()),'current_target':str(ROOT),'cache_lists_verified':cache_status,'historical_labels_with_retired_source_paths':len(historical_labels),'historical_label_example_paths':historical_labels[:5],'historical_label_note':'These legacy labels are retained for provenance; rerunning them requires deliberately migrating their source paths or restoring their old source revision. No active process had the retired dataset files open.','started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
write(CONTROL/'cleanup-plan.json',receipt)
# Only the two explicit, validated raw-data roots are removed.
for old in RETIRED:shutil.rmtree(old)
assert all(not old.exists() for old in RETIRED)
assert CURRENT.is_symlink() and CURRENT.resolve()==ROOT
assert hashfile(ROOT/'manifest.json')==installation['manifest_sha256']
for task in installation['tasks']:assert (ROOT/task['task_path']/'metadata.jsonl').is_file()
receipt.update(status='completed',completed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),retired_paths_absent=True,current_dataset_unchanged_by_cleanup=True)
write(CONTROL/'cleanup-report.json',receipt)
write(ROOT/'provenance/state-extension-20261103/cleanup-report.json',receipt)
write(CONTROL/'status.json',{'phase':'completed','local_relay':False,'installation_report':str(CONTROL/'installation.json'),'cleanup_report':str(CONTROL/'cleanup-report.json')})
print(json.dumps(receipt,ensure_ascii=False,indent=2))
