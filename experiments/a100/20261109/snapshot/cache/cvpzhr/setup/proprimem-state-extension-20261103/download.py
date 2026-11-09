"""Remote-only import of State S11-S19; all dataset payloads stay on this server."""
import concurrent.futures as cf
import importlib.util, json, pathlib, re, shutil, time, traceback

CONTROL=pathlib.Path('/cache/cvpzhr/setup/proprimem-state-extension-20261103')
TARGET=pathlib.Path('/cache/EVO-Intervention/datasets/ProPriMem-current').resolve()
STAGE=CONTROL/'staging'
REPO='syding9/ProPriMem-State'
REV='b682e418c6aad6d721ddef164da7c79a22d4c67a'
TASKS=['11-put_three_coins_into_piggy_bank','12-inspect_four_objects_and_return','13-inspect_three_drawers','14-weigh_four_objects_and_return','15-inspect_three_test_tubes','16-press_five_buttons_in_any_order','17-pour_water_into_three_cups','18-inspect_five_upside_down_cups','19-shake_three_sealed_bottles']
spec=importlib.util.spec_from_file_location('direct_download',CONTROL/'downloader_base.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
d.ROOT=STAGE/'State';d.CONTROL=CONTROL;d.REPO=REPO;d.TASK='State';d.REVISION=REV
d.STATE={'phase':'listing','started_at':time.time(),'repository':REPO,'revision':REV,'destination':str(TARGET),'staging':str(STAGE),'endpoint':d.ENDPOINT,'workers':12,'completed_files':0,'completed_bytes':0,'local_relay':False}
(CONTROL/'parts').mkdir(exist_ok=True)
def get_json(url):
    for attempt in range(8):
        try:
            with d.open_url(url) as response:return json.load(response),response.headers.get('Link','')
        except Exception:
            if attempt==7:raise
            time.sleep(min(2**attempt,16))
def tree(path='',recursive=False):
    url=f'{d.ENDPOINT}/api/datasets/{REPO}/tree/{REV}'+('/'+path if path else '')+f'?recursive={str(recursive).lower()}&expand=false&limit=1000'
    result=[]
    while url:
        entries,link=get_json(url);result+=entries
        next_link=re.search(r'<([^>]+)>;\s*rel="next"',link)
        url=next_link.group(1).replace('https://huggingface.co/',d.ENDPOINT+'/') if next_link else None
    return result
def normalize(f):
    return {'path':f['path'],'size':f['size'],'hash':f.get('lfs',{}).get('oid',f['oid']),'algorithm':'sha256' if f.get('lfs') else 'git-sha1'}
def main():
    d.status()
    inventory=CONTROL/'inventory.json'
    if inventory.exists():
        record=json.loads(inventory.read_text());assert record['revision']==REV
        items=record['files']
    else:
        info,_=get_json(f'{d.ENDPOINT}/api/datasets/{REPO}/revision/{REV}')
        assert info['sha']==REV and not info['private']
        entries=[]
        for task in TASKS:
            if (TARGET/'State'/task).exists():raise FileExistsError(task)
            entries.extend(f for f in tree('State/'+task,True) if f['type']=='file')
        items=[normalize(f) for f in sorted(entries,key=lambda f:f['path'])]
        assert sum('/videos/main_view/' in f['path'] and f['path'].endswith('.mp4') for f in items)==323
        assert sum(f['path'].endswith('.mp4') for f in items)==969
        assert len(items)==987
        record={'repository':REPO,'revision':REV,'files':items}
        inventory.write_text(json.dumps(record,indent=2)+'\n')
        metadata=[f for f in tree() if f['type']=='file' and f['path'] in ('manifest.json','README.md')]
        metadata += [f for f in tree('State') if f['type']=='file' and f['path']=='State/README.md']
        (CONTROL/'metadata-inventory.json').write_text(json.dumps([normalize(f) for f in metadata],indent=2)+'\n')
    for f in json.loads((CONTROL/'metadata-inventory.json').read_text()):
        dest=CONTROL/'upstream'/f['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or d.digest(dest,f)!=f['hash']:
            mode='resolve' if f['algorithm']=='sha256' else 'raw'
            with d.open_url(f"{d.ENDPOINT}/datasets/{REPO}/{mode}/{REV}/{f['path']}") as response:dest.write_bytes(response.read())
        assert dest.stat().st_size==f['size'] and d.digest(dest,f)==f['hash']
    total=sum(f['size'] for f in items)
    d.status(phase='downloading',total_files=len(items),total_bytes=total)
    d.emit('download_start',files=len(items),bytes=total,local_relay=False)
    items.sort(key=lambda f:(not f['path'].endswith(('.json','.jsonl')),f['path']))
    failed=[];last=0
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(d.download,f):f for f in items}
        for future in cf.as_completed(futures):
            f=futures[future]
            try:
                future.result();d.STATE['completed_files']+=1;d.STATE['completed_bytes']+=f['size']
            except Exception as e:failed.append({'path':f['path'],'error':str(e)})
            if time.time()-last>10:
                d.status(failed_files=len(failed));d.emit('progress',completed_files=d.STATE['completed_files'],completed_bytes=d.STATE['completed_bytes'],failed_files=len(failed));last=time.time()
    if failed:
        (CONTROL/'failures.json').write_text(json.dumps(failed,indent=2)+'\n')
        raise RuntimeError(f'{len(failed)} files failed')
    d.status(phase='verifying')
    def verify(f):
        p=STAGE/f['path']
        assert p.stat().st_size==f['size'] and d.digest(p,f)==f['hash'],f['path']
    with cf.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,items))
    report={'status':'verified','repository':REPO,'revision':REV,'tasks':TASKS,'trajectories':323,'video_files':969,'total_files':len(items),'total_bytes':total,'all_checksums_verified':True,'local_relay':False,'downloaded_on_remote':True,'staging':str(STAGE),'target':str(TARGET),'completed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (CONTROL/'download-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    d.status(phase='download_verified',verification=report);d.emit('download_verified',**report)
if __name__=='__main__':
    try:main()
    except BaseException:
        d.status(phase='failed',error=traceback.format_exc());raise
