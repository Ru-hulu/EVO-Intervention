"""Fixed new-data evaluation inputs and complete-trajectory metrics; no GPU imports."""
import csv
import json
import math
from fractions import Fraction
from pathlib import Path
from collections import Counter

from .prepare_cache import BASE, DATA, save, sha, progress_values

OUT = Path('/cache/cvpzhr/evaluations/prm-five-groups-20261114')
EVAL = BASE / 'evaluation'
IMAGE_ROOT = Path('/cache/cvpzhr/datasets/prm-five-groups-evaluation-images-20261114')
PLAN = Path(__file__).parent / 'experiment-plan.json'


def sampling_indices(frame_count, source_rate, target_rate=5):
    rate = Fraction(source_rate) / Fraction(str(target_rate))
    if frame_count < 1 or rate < 1:
        raise ValueError('Need nonempty video; upsampling is forbidden')
    indices, k = [], 0
    while True:
        f = math.ceil(k * rate)
        if f >= frame_count:
            break
        indices.append(f)
        k += 1
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def uniform_indices(current):
    import numpy as np
    if current < 0:
        raise ValueError('Negative current frame')
    return list(range(current + 1)) + [current] * (7-current) if current < 7 else np.rint(np.linspace(0,current,8)).astype(int).tolist()


def build_baseline_manifest():
    import av
    EVAL.mkdir(parents=True, exist_ok=True)
    destination = EVAL / 'group1-manifest.json'
    if destination.exists():
        raise FileExistsError('Frozen manifest exists; verify rather than replace it')
    plan = json.loads(PLAN.read_text())
    for suite in ['State','Order','Count']:
        assert sha(DATA/suite/'manifest.json') == plan['datasets']['suites'][suite]['manifest_sha256']
    source = json.loads((BASE/'cache-input-index.json').read_text())
    assert source['plan_sha256'] == sha(PLAN)
    records = []
    for row in sorted(source['records'], key=lambda x:x['key']):
        if row['split'] != 'val' or row['suite'] not in ['State','Order','Count']:
            continue
        pts = progress_values(row)
        rates = []; input_issues=[]
        for view,(path,digest) in enumerate(zip(row['videos'],row['video_hashes'])):
            if path:
                assert sha(path) == digest, path
                with av.open(path) as v:
                    stream = v.streams.video[0]
                    rates.append(stream.average_rate)
                    if stream.frames and stream.frames != row['frames']:
                        input_issues.append({'reason':'view_frame_count_mismatch','view':view,'path':path,
                                             'expected':row['frames'],'header_frames':stream.frames})
        assert len(set(rates)) == 1
        indices = sampling_indices(row['frames'],rates[0])
        records.append(dict(row, source_rate=str(rates[0]), input_fps=5,input_issues=input_issues,
            frame_indices=indices, timestamps=[i/float(rates[0]) for i in indices],
            target_pt=[pts[i] for i in indices]))
    assert Counter(r['suite'] for r in records) == {'State':89,'Order':119,'Count':177}
    manifest={'schema':'prm-five-groups-group1-v1','group':1,'condition':'baseline',
        'plan_sha256':sha(PLAN),'input_index_sha256':sha(BASE/'cache-input-index.json'),
        'sampling':'first source frame at or after each exact 5 Hz timestamp; include final source frame',
        'episodes':records,'expected_episodes':385,'expected_samples':sum(len(r['frame_indices']) for r in records),
        'by_suite':{s:{'episodes':sum(r['suite']==s for r in records),
                       'samples':sum(len(r['frame_indices']) for r in records if r['suite']==s)}
                    for s in ['State','Order','Count']}}
    save(destination,manifest)
    return manifest


def ensure_images(episode):
    """Decode synchronized raw views once, store lossless PNG, and verify all output hashes."""
    import av
    from contextlib import ExitStack
    root=IMAGE_ROOT/episode['key']/'fps5'
    receipt=root/'verification.json'
    signature={'video_hashes':episode['video_hashes'],'frame_indices':episode['frame_indices']}
    if receipt.exists():
        proof=json.loads(receipt.read_text())
        assert proof['input']==signature
        for name,digest in proof['png_sha256'].items():
            assert sha(root/name)==digest, name
        return root
    root.mkdir(parents=True,exist_ok=True)
    for path,digest in zip(episode['videos'],episode['video_hashes']):
        if path:
            assert sha(path)==digest
    wanted=set(episode['frame_indices']); observed=[]; hashes={}
    active=[i for i,p in enumerate(episode['videos']) if p]
    with ExitStack() as stack:
        videos=[stack.enter_context(av.open(episode['videos'][i])) for i in active]
        for video in videos:
            video.streams.video[0].codec_context.thread_count=1
        streams=[v.decode(video=0) for v in videos]
        decoded=0
        for index,frames in enumerate(zip(*streams,strict=True)):
            decoded=index+1
            if index not in wanted:
                continue
            observed.append(index)
            for view,frame in zip(active,frames):
                path=root/f'{index:08d}-{view}.png'
                temporary=path.with_suffix('.png.tmp')
                frame.to_image().save(temporary,format='PNG')
                temporary.replace(path)
                hashes[path.name]=sha(path)
    assert decoded==episode['frames'] and observed==episode['frame_indices']
    save(receipt,{'input':signature,'png_sha256':hashes,'decoded_native_frames':decoded,
                  'lossless':True,'missing_view_policy':'RGB black 64x64 placeholder'})
    return root


def correlations(pred,target):
    assert len(pred)==len(target)
    if any(p is None or not math.isfinite(p) for p in pred):
        return None,None,'invalid_predictions'
    x,y=list(map(float,pred)),list(map(float,target))
    if len(x)<2:return None,None,'fewer_than_two_samples'
    if max(x)==min(x):return None,None,'constant_prediction'
    if max(y)==min(y):return None,None,'constant_target'
    def pearson(a,b):
        ma,mb=math.fsum(a)/len(a),math.fsum(b)/len(b)
        a,b=[v-ma for v in a],[v-mb for v in b]
        r=math.fsum(i*j for i,j in zip(a,b))/math.sqrt(math.fsum(v*v for v in a)*math.fsum(v*v for v in b))
        return max(-1.,min(1.,r))
    def average_ranks(a):
        order=sorted(range(len(a)),key=a.__getitem__);ranks=[0.]*len(a);i=0
        while i<len(order):
            j=i+1
            while j<len(order) and a[order[j]]==a[order[i]]:j+=1
            for k in range(i,j):ranks[order[k]]=(i+j+1)/2
            i=j
        return ranks
    return pearson(x,y),pearson(average_ranks(x),average_ranks(y)),None


def trajectory_metrics(pred,target,timestamps):
    from .metric_reference.vs_scoring import score_curve
    p,s,reason=correlations(pred,target)
    invalid=sum(x is None or not math.isfinite(x) for x in pred)
    result={'pearson':p,'spearman':s,'correlation_undefined_reason':reason,
            'invalid_predictions':invalid,'samples':len(pred),'VS':None,'ER':None,'C':None}
    if invalid:
        result['vs_undefined_reason']='invalid_predictions'
    elif len(pred)<7:
        result['vs_undefined_reason']='fewer_than_seven_samples'
    else:
        vs=score_curve(pred,timestamps)
        result.update(VS=vs.vs,ER=vs.er,C=vs.scale,vs_details=vs.to_dict(),vs_undefined_reason=None)
    return result


def result_path(model,key,condition='baseline'):
    return OUT/'group1'/condition/model/'episodes'/(key.replace('/','__')+'.json')


def aggregate(model):
    import fcntl
    directory=OUT/'group1/baseline'/model
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'aggregation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _aggregate_unlocked(model)


def _aggregate_unlocked(model):
    manifest=json.loads((EVAL/'group1-manifest.json').read_text());digest=sha(EVAL/'group1-manifest.json')
    rows=[];missing=[]
    for e in manifest['episodes']:
        path=result_path(model,e['key'])
        if not path.exists():
            missing.append(e['key']);continue
        r=json.loads(path.read_text())
        assert r['manifest_sha256']==digest and r['frame_indices']==e['frame_indices']
        assert r['target_pt']==e['target_pt'] and r['timestamps']==e['timestamps']
        assert len(r['predictions'])==len(e['frame_indices'])
        rows.append(dict(key=e['key'],suite=e['suite'],task_id=e['task_id'],
                         **trajectory_metrics(r['predictions'],r['target_pt'],r['timestamps'])))
    def summary(rr,expected):
        out={'completed_episodes':len(rr),'expected_episodes':expected,'samples':sum(r['samples'] for r in rr),
             'invalid_frames':sum(r['invalid_predictions'] for r in rr)}
        for metric in ['pearson','spearman','VS','ER','C']:
            valid=[r[metric] for r in rr if r[metric] is not None]
            out[metric]=sum(valid)/len(valid) if valid else None
            out[metric+'_valid_episodes']=len(valid)
        out['correlation_undefined_reasons']=dict(Counter(r['correlation_undefined_reason'] for r in rr if r['correlation_undefined_reason']))
        return out
    result={'model':model,'group':1,'complete':not missing,'missing_episodes':missing,
        'aggregation':'equal weight per trajectory; valid denominators explicit; never pool frames',
        'overall':summary(rows,385),'by_suite':{s:summary([r for r in rows if r['suite']==s],n) for s,n in [('State',89),('Order',119),('Count',177)]},
        'by_episode':rows,'manifest_sha256':digest,'metric_source':json.loads((Path(__file__).parent/'metric_reference/provenance.json').read_text())}
    save(OUT/'group1/baseline'/model/'metrics.json',result)
    return result


if __name__=='__main__':
    import sys
    if sys.argv[1]=='manifest':
        m=build_baseline_manifest();print(json.dumps({k:m[k] for k in ['expected_episodes','expected_samples','by_suite']}))
    elif sys.argv[1]=='aggregate':
        print(json.dumps(aggregate(sys.argv[2])['overall']))
