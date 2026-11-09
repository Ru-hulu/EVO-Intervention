from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

reports=Path('/cache/cvpzhr/setup/dataset-replacement-20261030')
root=Path('/cache/cvpzhr/datasets/ProPriMem')
info=json.loads((reports/'repo-info.json').read_text())
def check(entry):
    p=root/entry['rfilename']
    assert p.is_file() and not p.is_symlink() and p.stat().st_size==entry['size'],entry['rfilename']
    h=hashlib.new('sha256' if entry.get('lfs') else 'sha1')
    if not entry.get('lfs'):h.update(f'blob {entry["size"]}\0'.encode())
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==(entry['lfs']['sha256'] if entry.get('lfs') else entry['blobId']),entry['rfilename']
start=time.time()
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,info['siblings']))
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
assert actual=={e['rfilename'] for e in info['siblings']},'Unexpected/missing files'
counts=Counter(e['rfilename'].split('/')[0] for e in info['siblings'])
report={'status':'completed','repo':'MINT-SJTU/ProPriMem','revision':info['sha'],'path':str(root),
        'files':len(info['siblings']),'bytes':sum(e['size'] for e in info['siblings']),
        'all_hashes_verified':True,'hash_method':'LFS SHA256 and Git blob SHA1 from authenticated Hugging Face API',
        'source_method':'Reused 7477 bit-identical previously downloaded files on AutoDL; every file verified against the pinned Hugging Face snapshot before and after transfer',
        'top_level_file_counts':dict(counts),'seconds':time.time()-start,
        'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'old_forest_cache_deleted':not Path('/cache/cvpzhr/datasets/forest-5fps').exists(),
        'training_cache_built':False,'training_started':False}
(reports/'download-receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
ready=reports.parent/'readiness.json';state=json.loads(ready.read_text())
state.update(status='RAW_DATASET_READY_PREPROCESSING_REQUIRED',training_data_ready=False,raw_dataset=report,
             train_samples=None,validation_samples=None,
             free_disk_gib=shutil.disk_usage(root).free/2**30,training_configuration_valid=False,
             reason='Old Forest caches deleted as requested. Full upstream ProPriMem snapshot downloaded; current training split/sample preparation and history-cache construction have not been run.')
for key in ['data_archives_sha256_verified','data_bytes_transferred_uncompressed','data_bytes_transferred_compressed']:
    state.pop(key,None)
ready.write_text(json.dumps(state,ensure_ascii=False,indent=2))
readme=reports.parent/'README.txt'
original=readme.read_text();backup=reports/'previous-README.txt'
if not backup.exists():backup.write_text(original)
readme.write_text('数据更新：原 forest-5fps 训练/验证缓存已按用户要求删除。\n'
                  'ProPriMem 原始数据位于 /cache/cvpzhr/datasets/ProPriMem，完整性校验已通过。\n'
                  '尚未将新数据接入训练配置，也未构建 5 FPS 样本和历史缓存；暂不能直接使用旧训练配置启动。\n'
                  '独立环境、代码和 Qwen3.5-4B 权重保留。\n\n原环境配置记录（旧数据路径已失效）：\n'+original)
print(json.dumps(report,ensure_ascii=False))
