"""Independent checks of the saved CSVs, source hashes, and agreed rollback amounts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def validate(root, progress_dir, names):
    reports=[]
    for name in names:
        annotation_path=root/'videos/annotation'/f'{name}.json'
        a=json.loads(annotation_path.read_text())
        metadata=json.loads((progress_dir/f'{name}.metadata.json').read_text())
        csv_path=progress_dir/f'{name}.progress.csv'
        with csv_path.open() as file:
            rows=list(csv.DictReader(file))
        p=[float(r['pt']) for r in rows]
        d=[float(r['delta']) for r in rows]
        stages={s['stage_id']:s for s in a['stages']}
        totals={sid:math.fsum(d[s['start_frame']:s['end_frame']+1]) for sid,s in stages.items()}
        split_recovery = len(stages) == 13
        reference_stages = [8,2,3,4,10,11,12,13] if split_recovery else [8,2,3,4,9,10,11,12]
        denom=math.fsum(totals[sid] for sid in reference_stages)
        assert math.isclose(denom,metadata['normalization_denominator'],abs_tol=1e-12)
        assert len(rows)==a['total_frames']
        assert [int(r['frame_index']) for r in rows]==list(range(len(rows)))
        assert all(abs(float(r['time_seconds'])-i/a['fps'])<1e-12 for i,r in enumerate(rows))
        assert all(math.isfinite(v) and 0<=v<=1+1e-12 for v in p)
        assert p[0]==0 and abs(p[-1]-1)<1e-12
        assert d[0]==0 and all(math.isfinite(v) and v>=0 for v in d)
        assert hashlib.sha256(annotation_path.read_bytes()).hexdigest()==metadata['annotation_sha256']
        assert hashlib.sha256((root/'videos/main_view'/f'{name}.mp4').read_bytes()).hexdigest()==metadata['video_sha256']
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest()==metadata['progress_csv_sha256']
        delta_path=root/'delta'/f'{name}-delta.txt'
        receipt=json.loads((root/'delta'/f'{name}.receipt.json').read_text())
        delta_hash=hashlib.sha256(delta_path.read_bytes()).hexdigest()
        assert delta_hash==metadata['delta_sha256']==receipt['delta_sha256']
        assert receipt['source_sha256']==a['video_sha256'] and receipt['frames']==len(rows)
        drops=[i for i in range(1,len(p)) if p[i]<p[i-1]-1e-12]
        assert drops==[stages[sid]['start_frame'] for sid in [6,7]]
        recover_cup2 = 10 if split_recovery else 9
        complete_returns = [(1,8),(5,recover_cup2),(recover_cup2,recover_cup2)]
        if not split_recovery: complete_returns.append((8,8))
        for sid,target in complete_returns:
            s=stages[sid]
            before=p[s['start_frame']-1] if s['start_frame'] else 0
            assert math.isclose(p[s['end_frame']]-before,totals[target]/denom,abs_tol=1e-12)
        events=[]
        if split_recovery:
            a8,b9=stages[8]['start_frame'],stages[9]['end_frame']
            assert math.isclose(p[b9]-p[a8-1],totals[8]/denom,abs_tol=1e-12)
            for sid in (8,9):
                s=stages[sid]
                expected=totals[8]/denom*totals[sid]/(totals[8]+totals[9])
                assert math.isclose(p[s['end_frame']]-p[s['start_frame']-1],expected,abs_tol=1e-12)
            assert len(metadata['milestones'])==8
        for sid,target in [(6,8),(7,recover_cup2)]:
            s=stages[sid]; i=s['start_frame']
            assert math.isclose(p[i-1]-p[i],totals[target]/denom,abs_tol=1e-12)
            assert all(v==p[i] for v in p[i:s['end_frame']+1])
            events.append({'stage_id':sid,'frame':i,'seconds':i/a['fps'],'pt_before':p[i-1],'pt_after':p[i]})
        for i,r in enumerate(rows):
            if int(r['stage_id'])==0:
                assert p[i]==(p[i-1] if i else 0)
        uncertain=set()
        for boundary in a['boundary_reviews']:
            lo,hi=boundary['review_source_frame_range_inclusive'];uncertain.update(range(lo,hi+1))
        assert {i for i,r in enumerate(rows) if r['boundary_needs_review']=='1'}==uncertain
        reports.append({'episode':name,'frames':len(rows),'stage_count':len(stages),'final_pt':p[-1],
                        'stage_1_end_pt':p[stages[1]['end_frame']],
                        'stage_2_end_pt':p[stages[2]['end_frame']],
                        'rollback_events':events,'uncertain_boundary_frames':len(uncertain),
                        'checks_passed':True})
    result={'episode_count':len(reports),'total_frames':sum(x['frames'] for x in reports),
            'all_checks_passed':True,'episodes':reports}
    (progress_dir/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--progress-dir',type=Path,required=True)
    parser.add_argument('--episodes',nargs='+',required=True)
    args=parser.parse_args()
    validate(args.dataset_root,args.progress_dir,args.episodes)
