"""Independent integer-grid, label, pixel and per-model input contract audit."""
import csv
import os
import time
from collections import Counter
from .common import *


def verify():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    data=read(DATA/'inputs.json');source=read(SOURCE)
    assert sha(SOURCE)==data['source_inputs_sha256']
    assert not data['evaluation_started'] and data['protocol']['mode']=='cumulative'
    assert data['protocol']['automatic_evaluation_enabled'] is False
    assert data['registry']==source['registry'] and data['episodes']==source['episodes']
    assert sha(ROOT/'source-checksums.json')==data['source_checksums_sha256']
    for p,h in read(ROOT/'source-checksums.json')['files'].items():assert sha(ROOT/'code'/p)==h,p
    for p,h in data['protected_source_files'].items():assert sha(p)==h,p
    for p,h in data['cache_files_sha256'].items():assert sha(p)==h,p
    rows={r['key']:r for r in data['episodes']}
    assert len(rows)==60 and Counter(r['suite'] for r in rows.values())==dict(State=16,Order=30,Count=14)
    labels={};image_count=0
    for key,row in rows.items():
        resolved=data['resolved_sources'][row['progress_path']]
        assert resolved['sha256']==row['progress_sha256'] and sha(resolved['path'])==row['progress_sha256']
        with open(resolved['path']) as f:csv_rows=list(csv.DictReader(f))
        assert [int(r['frame_index']) for r in csv_rows]==list(range(row['frames']))
        labels[key]=[float(r['pt']) for r in csv_rows]
        image=data['image_caches'][key];proof=read(image['proof'])
        assert sha(image['proof'])==image['sha256']
        assert proof['input']==dict(video_hashes=row['video_hashes'],frame_indices=list(range(row['frames'])))
        assert set(proof['png_sha256'])=={f'{f:08d}-{v}.png' for f in range(row['frames']) for v,p in enumerate(row['videos']) if p}
        for name,h in proof['png_sha256'].items():assert sha(Path(image['path'])/name)==h
        image_count+=len(proof['png_sha256'])
    old6={r['key']:r for r in read(source['streams']['6']['path'])['episodes']}
    previous={};counts={};model_observations=0
    for level in range(1,6):
        ref=data['streams'][str(level)];manifest=read(ref['path']);assert sha(ref['path'])==ref['sha256']
        entries=manifest['episodes'];assert len(entries)==len({e['key'] for e in entries})==60
        assert {e['key'] for e in entries}==set(rows)
        suite_counts=Counter(); observations=0
        for e in entries:
            row=rows[e['key']];n=row['frames'];order=e['segment_order'];assert sorted(order)==list(range(10))
            rates=[6]*10
            for k in range(1,level):
                for s in order[2*(k-1):2*k]:rates[s]=6*(k+1)
            assert e['segment_rates']==rates
            assert Counter(rates)==Counter([6]*(12-2*level)+[6*(k+1) for k in range(1,level) for _ in range(2)])
            # Reconstruct native grids independently, without the generator or old grids.
            grids={}
            for rate in set(rates):
                grid=set();k=0
                while (k*30+rate-1)//rate<n:
                    grid.add((k*30+rate-1)//rate);k+=1
                grid.add(n-1);grids[rate]=grid
            expected=[f for f in range(n) if f in grids[rates[(f*10)//n]]]
            frames=e['source_frame_indices'];assert frames==expected
            assert e['segment_ids']==[f*10//n for f in frames]
            assert e['segment_frame_bounds']==[(j*n+9)//10 for j in range(11)]
            assert e['segment_time_boundaries']==[j*n/300 for j in range(11)]
            assert e['target_pt']==[labels[e['key']][f] for f in frames]
            assert e['timestamps']==[f/30 for f in frames] and e['progress_sha256']==row['progress_sha256']
            assert [frames[i] for i in e['common_scoring_indices']]==list(range(0,n,5))
            assert e['common_scoring_timestamps']==[f/30 for f in range(0,n,5)]
            if level==1:
                for k in ('source_frame_indices','timestamps','target_pt','common_scoring_indices'):assert e[k]==old6[e['key']][k]
            else:
                prior=previous[e['key']]
                assert order==prior['segment_order'] and set(prior['source_frame_indices'])<=set(frames)
                assert {i for i in range(10) if rates[i]!=prior['segment_rates'][i]}==set(order[2*(level-2):2*(level-1)])
            previous[e['key']]=e;observations+=len(frames);suite_counts[row['suite']]+=len(frames)
            for model in MODELS:
                c=read(model_path(model,level,e['key']));info=data['registry'][f'{model}-{row["suite"]}']
                assert c['model']==model and c['suite']==row['suite'] and c['level']==level
                assert c['checkpoint']==info['checkpoint'] and c['checkpoint_sha256']==info['checkpoint_sha256']
                assert c['shared_stream_sha256']==ref['sha256'] and c['entry_sha256']==digest(e)
                inference=c['inference'];assert set(inference)=={'key','task','positions','images'}
                assert inference['key']==e['key'] and inference['task']==row['task_prompt'] and inference['positions']==frames
                assert inference['images']==[[str(Path(row['image_root'])/f'{f:08d}-{v}.png') if p else None for v,p in enumerate(row['videos'])] for f in frames]
                assert c['memory_enabled'] and c['uses_only_observed_prefix'] and not c['targets_in_inference']
                assert not c['evaluation_started'] and not c['future_endpoint_reference']
                assert c['memory_reset']=='once at trajectory start; never at segment boundaries'
                assert len(c['input_playback_slots'])==len(frames)
                for i,actual in enumerate(c['input_playback_slots']):
                    if model=='procvlm-2b':expected_slots=[[max(0,j),0] for j in range(i-7,i+1)]
                    elif model=='robometer-4b':
                        history=(list(range(i+1))+[i]*(7-i))[:7] if i<7 else [round(j*i/7) for j in range(7)]
                        expected_slots=[[j,0] for j in history]+[[i,v] for v in range(3)]
                    else:expected_slots=[[i,v] for v in range(3)]
                    assert actual==expected_slots and all(0<=j<=i for j,v in actual)
                assert c['reuse_predictions']==(level==1)
                if level==1:
                    b=c['baseline_result'];assert sha(b['path'])==b['sha256']
                    r=read(b['path']);assert r['source_frame_indices']==frames and r['target_pt']==e['target_pt']
                    assert r['identity']['checkpoint_sha256']==c['checkpoint_sha256']
                    assert len(r['predictions'])==len(r['raw_responses'])==len(frames)
                else:assert c['baseline_result'] is None
        assert observations==ref['observations']==manifest['expected_observations']
        counts[str(level)]=dict(trajectories=60,observations_per_model=observations,observations_by_suite=dict(suite_counts),nominal_average_fps=[6,7.2,9.6,13.2,18][level-1])
        model_observations+=observations*3
    jobs=read(DATA/'jobs.json');assert not jobs['automatic_launch'] and jobs['requires_user_instruction']
    assert len(jobs['jobs'])==len({(j['model'],j['level'],j['key']) for j in jobs['jobs']})==900
    assert Counter(j['status'] for j in jobs['jobs'])==dict(reuse_verified_baseline=180,prepared_waiting_for_user_instruction=720)
    assert model_observations==data['expected_model_observations']
    proof=dict(status='prepared_waiting_for_user_instruction',inputs_sha256=sha(DATA/'inputs.json'),
        trajectories=60,models=3,model_suite_bindings=9,levels=5,model_input_caches=900,
        baseline_reuse_curves=180,new_inference_curves=720,observations=model_observations,
        new_inference_observations=model_observations-3*counts['1']['observations_per_model'],
        by_level=counts,verified_pngs=image_count,all_original_data_labels_predictions_unchanged=True,
        all_model_history_maps_verified=True,common_6fps_grid_identical=True,
        original_files_resolved_from_archive=sum(p!=r['path'] for p,r in data['resolved_sources'].items()),
        original_label_csvs_resolved_from_archive=sum(data['resolved_sources'][r['progress_path']]['path']!=r['progress_path'] for r in rows.values()),
        gpu_used=False,model_forward_executed=False,evaluation_started=False,automatic_evaluation_enabled=False,
        finished=time.time())
    save(DATA/'verification.json',proof);save(ROOT/'status.json',proof)
    return proof


if __name__=='__main__':print(__import__('json').dumps(verify()))
