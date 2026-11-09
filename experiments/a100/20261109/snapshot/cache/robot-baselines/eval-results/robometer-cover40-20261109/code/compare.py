"""Compare saved predictions to unchanged labels and assemble a viewer dataset."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'datasets/RMBench-CoverBlocks-40-20261105'
DEST = ROOT.parent/'datasets/RMBench-CoverBlocks-RoboMeter-20261109'


def main():
    (DEST/'progress').mkdir(parents=True, exist_ok=True)
    (ROOT/'predictions').mkdir(exist_ok=True)
    rows = [json.loads(s) for s in (SOURCE/'metadata.jsonl').read_text().splitlines()]
    summaries, metadata, events = [], [], []
    errors, regular_errors = [], []
    for row in rows:
        sid = row['sample_id']
        path = ROOT/'predictions'/(sid+'.json')
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        pp = result['predictions']
        gt_path = SOURCE/'progress'/(sid+'.json')
        gt = json.loads(gt_path.read_text())
        dense = gt['dense_progress']
        assert len(dense) == row['total_frames']
        assert all(p['frame'] == max(p['history_frames']) and min(p['history_frames']) == 0 for p in pp)
        assert all(p['annotated_progress'] == dense[p['frame']] for p in pp)
        ee = [abs(p['predicted_progress']-p['annotated_progress']) for p in pp]
        regular = {round(t*row['fps']) for t in range(int((row['total_frames']-1)/row['fps'])+1)}
        re = [e for p,e in zip(pp,ee) if p['frame'] in regular]
        errors.extend(ee)
        regular_errors.extend(re)
        summary = dict(sample_id=sid, success=row['task_success'], points=len(pp),
            mae_all=statistics.mean(ee), mae_1hz=statistics.mean(re),
            final_annotation=dense[-1], final_prediction=pp[-1]['predicted_progress'],
            annotation_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest())
        summaries.append(summary)
        predictions = {p['frame']:p['predicted_progress'] for p in pp}
        ann = json.loads((SOURCE/row['annotation_path']).read_text())
        for stage in ann['stages']:
            if stage.get('activity') in ('wrong_approach','memory_retrieval','wrong_open','irreversible_failure'):
                a,b = stage['start_frame'], min(row['total_frames']-1, stage['end_frame']-1)
                events.append(dict(sample_id=sid, activity=stage['activity'], start_frame=a, end_frame=b,
                    annotation_delta=dense[b]-dense[a], predicted_delta=predictions[b]-predictions[a]))
        for f in range(1, len(dense)):
            if dense[f]-dense[f-1] < -.05:
                events.append(dict(sample_id=sid, activity='annotation_drop', frame=f,
                    annotation_before=dense[f-1], annotation_after=dense[f],
                    prediction_before=predictions[f-1], prediction_after=predictions[f]))
        gt['prediction'] = dict(model='RoboMeter-4B 零样本', points=[[p['frame'],p['predicted_progress']] for p in pp],
            mae=summary['mae_all'], protocol='causal prefix, 8 main-view images, 1 Hz + event boundaries')
        gt['presentation'] = 'simple_progress'
        (DEST/'progress'/(sid+'.json')).write_text(json.dumps(gt,ensure_ascii=False))
        if (DEST/sid).is_symlink():
            (DEST/sid).unlink()
        if not (DEST/sid).exists():
            shutil.copytree(SOURCE/sid, DEST/sid, copy_function=os.link)
        metadata.append(row)
    if not summaries:
        return
    if (DEST/'boundaries').is_symlink():
        (DEST/'boundaries').unlink()
    if not (DEST/'boundaries').exists():
        shutil.copytree(SOURCE/'boundaries', DEST/'boundaries', copy_function=os.link)
    (DEST/'metadata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in metadata))
    report = dict(trajectories=len(summaries), expected_trajectories=40, points=sum(s['points'] for s in summaries),
        mae_all_points=statistics.mean(errors), mae_regular_1hz=statistics.mean(regular_errors),
        episode_macro_mae_1hz=statistics.mean(s['mae_1hz'] for s in summaries),
        terminal_groups={str(success):dict(count=len(group), mean_prediction=statistics.mean(group),
             min_prediction=min(group), max_prediction=max(group))
             for success in (True,False) if (group:=[s['final_prediction'] for s in summaries if s['success']==success])},
        episodes=summaries, events=events)
    (ROOT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (ROOT/'per-trajectory.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (DEST/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','events')},ensure_ascii=False))


if __name__ == '__main__':
    main()
