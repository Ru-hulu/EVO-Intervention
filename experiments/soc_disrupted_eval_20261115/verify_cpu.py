"""Independent exhaustive map/metric checks, with zero model or GPU calls."""
from .common import *
from .protocol import timeline, model_episode, image_slots
from .metrics import cycle_triplets, score


def verify():
    sources=read(DATA/'inputs.json');rows={r['key']:r for r in sources['episodes']}
    counts={};triplets=0;overlap_versions=0
    for condition in CONDITIONS:
        manifest=read(DATA/'manifests'/f'{condition}.json');observations=0
        for entry in manifest['episodes']:
            row=rows[entry['key']];clock=timeline(row,entry);episode=model_episode(row,entry)
            n=len(entry['target_pt']);observations+=n
            assert len(clock['playback_positions'])==n
            for i in range(n):
                for model,images in (('procvlm-2b',8),('robometer-4b',10),('forest',3)):
                    slots=image_slots(episode,i,model)
                    assert len(slots)==images and all(step<=i for step,view in slots)
            oracle=score(entry['target_pt'],row,entry)
            assert abs(oracle['pearson']-1)<1e-12 and abs(oracle['spearman']-1)<1e-12
            if entry['condition']=='rewind_only':
                assert abs(oracle['circle_consistency'])<1e-15
                groups=cycle_triplets(row,entry)
                for j,g in enumerate(groups):
                    e=entry['events'][j];start=e['first_inserted_step'];length=e['duration_steps']
                    # Independent lookup by source-frame value in each disjoint
                    # playback phase, rather than the closed-form index formula.
                    original=clock['original_positions']
                    reverse=range(start,start+length);forward=range(start+length,start+2*length)
                    for point in g['points']:
                        frame=point['source_frame']
                        found=[[i for i in phase if entry['source_frame_indices'][i]==frame]
                               for phase in (original,reverse,forward)]
                        assert all(len(v)==1 for v in found)
                        assert [v[0] for v in found]==point['prediction_indices'];triplets+=1
                a,b=[{p['source_frame'] for p in g['points']} for g in groups]
                overlap_versions+=bool(a&b)
            else:assert abs(oracle['static_mae'])<1e-15
        assert observations==manifest['expected_samples']
        counts[condition]=dict(episodes=len(manifest['episodes']),observations=observations)
    result=dict(status='passed',conditions=counts,observations=sum(v['observations'] for v in counts.values()),
        versions=sum(v['episodes'] for v in counts.values()),loop_triplets_independently_paired=triplets,
        versions_with_overlapping_loop_interiors=overlap_versions,model_forward_calls=0,
        gpu_forward_tested=False,evaluation_started=False)
    assert result['versions']==504 and result['observations']==105286
    save(ROOT/'cpu-map-metric-verification.json',result)
    return result


if __name__=='__main__':print(json.dumps(verify()))
