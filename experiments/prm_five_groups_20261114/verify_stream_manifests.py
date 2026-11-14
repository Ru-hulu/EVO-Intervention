"""Independently verify every frozen playback-map entry against source PT."""
import csv
import hashlib
import json
import time

from .eval_common import EVAL
from .prepare_cache import save,sha


def verify():
    root=EVAL/'streams';receipt=json.loads((root/'preparation.json').read_text())
    baseline=EVAL/'group1-manifest.json';source={r['key']:r for r in json.loads(baseline.read_text())['episodes']}
    native={};common_by_key={};summary={};total=0
    for path,proof in receipt['manifests'].items():
        assert sha(path)==proof['sha256']
        with open(path) as file:manifest=json.load(file)
        assert manifest['baseline_manifest_sha256']==sha(baseline)
        keys=[r['key'] for r in manifest['episodes']]
        assert len(keys)==len(set(keys))==manifest['expected_episodes']
        samples=0
        for entry in manifest['episodes']:
            original=source[entry['key']];frames=entry['source_frame_indices']
            truth=entry['target_pt'];times=entry['playback_timestamps']
            assert len(frames)==len(truth)==len(times) and all(a<b for a,b in zip(times,times[1:]))
            assert entry['input_issues']==original.get('input_issues',[])
            unhashed=dict(entry);expected=unhashed.pop('map_sha256')
            actual=hashlib.sha256(json.dumps(unhashed,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            assert actual==expected
            if manifest['group']==2:
                steps=entry['source_step_indices']
                assert frames==[original['frame_indices'][i] for i in steps]
                assert truth==[original['target_pt'][i] for i in steps]
                inserted=set()
                for event in entry['events']:
                    anchor=event['after_source_step'];duration=event['duration_steps']
                    lo=event['first_inserted_step'];hi=event['last_inserted_step']+1
                    inserted.update(range(lo,hi))
                    expected=([anchor]*duration if entry['condition']=='static_only' else
                        list(range(anchor-1,anchor-duration-1,-1))+list(range(anchor-duration+1,anchor+1)))
                    assert steps[lo:hi]==expected and steps[hi-1]==anchor
                assert [v for i,v in enumerate(steps) if i not in inserted]==list(range(len(original['frame_indices'])))
            elif manifest['group']==5:
                if entry['key'] not in native:
                    assert sha(original['progress_path'])==original['progress_sha256']
                    with open(original['progress_path']) as file:raw=list(csv.DictReader(file))
                    assert [int(r['frame_index']) for r in raw]==list(range(original['frames']))
                    native[entry['key']]=[float(r['pt']) for r in raw]
                assert truth==[native[entry['key']][i] for i in frames]
                common=[frames[i] for i in entry['common_scoring_indices']]
                assert common==list(range(0,original['frames'],30))==entry['common_scoring_native_frames']
                if entry['key'] in common_by_key:assert common==common_by_key[entry['key']]
                common_by_key[entry['key']]=common
            else:
                raise ValueError('Unexpected group in stream map')
            total+=len(frames);samples+=len(frames)
        assert samples==manifest['expected_samples']
        summary[f"group{manifest['group']}/{manifest['condition']}"]={'episodes':len(keys),'samples':samples}
    assert len(summary)==12 and len(native)==235
    proof={'status':'passed','all_maps_independently_checked':True,'total_manifests':len(summary),
        'total_versioned_observations':total,'native_label_sources_checked':len(native),
        'no_inference_started_by_preparation':True,'summary':summary,'checked_at':time.time(),
        'verifier_sha256':sha(__file__)}
    save(root/'verification.json',proof)
    return proof


if __name__=='__main__':print(json.dumps(verify()))
