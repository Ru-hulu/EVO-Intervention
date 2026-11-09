from pathlib import Path
import json, hashlib, csv, ast, os, time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    started=time.time();policy=read(ROOT/'launch-policy.json');fc=read(ROOT/'forest/training.config.json');rc=read(ROOT/'robometer/training.config.json')
    assert not fc.get('initial_checkpoint') and not rc.get('initial_checkpoint')
    assert fc['batch_size']*fc['gradient_accumulation_steps']*2==18
    assert rc['training']['per_device_train_batch_size']*rc['training']['gradient_accumulation_steps']*2==18
    assert fc['epochs']==rc['training']['num_train_epochs']==3
    assert fc['learning_rate']==rc['training']['learning_rate']==1e-4
    assert fc['warmup_ratio']==rc['training']['warmup_ratio']==.03
    assert fc['lr_schedule']==rc['training']['lr_scheduler_type']=='cosine'
    assert rc['input_contract']=='history7-main-current3-v1' and rc['images_per_sample']==10
    for t in policy['tasks'].values():assert not Path(t['output_dir']).exists()
    fd=Path('/cache/cvpzhr/datasets/forest-spray-5fps-no-augmentation-20261109')
    rd=Path(rc['train_data']).parent
    assert read(fd/'cache-ready.json')['status']=='completed'
    assert read(rd/'cache-ready.json')['status']=='ready' and read(rd/'label-audit.json')['status']=='passed'
    plan=read(fd/'plan.json')['records'];summary={};seen_splits={};files=set()
    for split,n,episodes in [('train',7416,25),('validation',1919,7)]:
        expected={};selected=[r for r in plan if r['split']==split]
        assert len(selected)==episodes
        assert read(fd/f'{split}.sample_files.json')==[r['output_dir'] for r in selected]
        for r in selected:
            assert not r['augmentation_enabled'] and r['added_samples']==0 and not r['events']
            assert digest(r['progress_csv'])==r['progress_sha256']
            assert digest(r['source_progress'])==r['source_progress_sha256']
            with Path(r['progress_csv']).open() as f:labels={int(x['frame_index']):float(x['pt']) for x in csv.DictReader(f)}
            folder=Path(r['output_dir']);rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
            frames=list(range(0,r['frames'],6))
            if frames[-1]!=r['frames']-1:frames.append(r['frames']-1)
            assert [x['frame_index'] for x in rows]==frames
            for row in rows:
                key=(r['episode_id'],row['frame_index']);assert key not in expected
                assert row['pt']==labels[row['frame_index']];expected[key]=row['pt']
                assert set(row['images'])=={'cam_high','cam_left_wrist','cam_right_wrist'}
                files.update(str(folder/x) for x in row['images'].values())
                if row['history']:files.add(str(folder/row['history']))
        rows=[json.loads(line) for line in Path(rc['train_data'] if split=='train' else rc['validation_data']).read_text().splitlines()]
        assert len(rows)==n==len(expected)
        actual={}
        for row in rows:
            key=(row['episode_id'],row['source_frame_index']);assert key not in actual
            assert row['kind']=='original' and row['pt_policy']=='none' and not row['missing_current_views']
            assert row['task_id']==rc['task'] and len(row['image'])==10
            assert row['input_contract']==rc['input_contract'] and row['source_frame_index']==row['frame_index']
            assert all(x<=row['source_frame_index'] for x in row['history_source_frame_indices'])
            assert row['pt']==expected[key];actual[key]=row['pt']
            files.update(row['image']);files.add(row['pixel_cache'])
        assert actual==expected
        seen_splits[split]={key[0] for key in actual}
        summary[split]={'episodes':episodes,'samples':n,'both_models_same_frames_and_pt':True}
    assert not seen_splits['train'] & seen_splits['validation']
    for p in files:assert Path(p).is_file() and Path(p).stat().st_size>0,p
    metadata=read('/cache/cvpzhr/setup/robometer-20261031/model-metadata.json')['repositories']
    official=next(r for r in metadata if r['id']=='robometer/Robometer-4B')
    assert official['sha']==rc['initial_model_revision'];weights={}
    for info in official['siblings']:
        p=Path(rc['model_path'])/info['rfilename'];assert p.stat().st_size==info['size']
        if 'lfs' in info:
            sha=digest(p);assert sha==info['lfs']['sha256'];weights[p.name]=sha
        else:
            raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==info['blobId']
    hashes={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and (p.suffix in ['.py','.json'] or p.name.endswith('.py.disabled')) and p.name not in ['preflight.json','run-files-sha256.json']:
            if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
            if p.suffix=='.json':read(p)
            hashes[str(p.relative_to(ROOT))]=digest(p)
    (ROOT/'run-files-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    result={'status':'passed','splits':summary,'checked_cache_files':len(files),'source_labels_preserved':True,'augmentation':'none',
            'official_robometer_repository':official['id'],'official_revision':official['sha'],'official_weight_sha256':weights,
            'forest_initialization':'fresh Qwen3.5-4B; no fine-tuned checkpoint','previous_results_preserved':True,'seconds':time.time()-started}
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
