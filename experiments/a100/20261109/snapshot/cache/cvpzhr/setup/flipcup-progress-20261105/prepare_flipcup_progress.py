"""Flipcup progress: last-attempt delta weights, explicit rollback events, and idle holds."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


MILESTONES = {
    1: 'cup1_pick_initial', 2: 'cup1_flip_first', 3: 'cup1_pick_second',
    4: 'cup1_flip_second', 5: 'cup2_pick_first', 8: 'cup1_pick_initial',
    9: 'cup2_pick_first', 10: 'cup2_flip_first', 11: 'cup2_pick_second',
    12: 'cup2_flip_second',
}
MILESTONES_13 = {
    1: 'cup1_pick_initial', 2: 'cup1_flip_first', 3: 'cup1_pick_second',
    4: 'cup1_flip_second', 5: 'cup2_pick_first', 8: 'cup1_pick_initial',
    9: 'cup1_pick_initial', 10: 'cup2_pick_first', 11: 'cup2_flip_first',
    12: 'cup2_pick_second', 13: 'cup2_flip_second',
}
ROLLBACKS = {6: 'cup1_pick_initial', 7: 'cup2_pick_first'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calculate_progress(deltas, annotation):
    """Original-frame inclusive intervals; pure failure events earn no positive credit.

    As in the old episode_to_progress, stage delta includes its first frame, and
    each semantic ID uses its last positive attempt's delta sum as target credit.
    Stage 6 revokes stage 1, stage 7 revokes stage 5, at each event's first frame.
    With 13 stages, only stage 8 defines the cup1 pickup target. Stages 8/9
    jointly restore that one target, allocated by their combined delta curve.
    Stage 10 restores stage 5. The legacy 12-stage mapping remains supported.
    All gaps hold progress; recovery handoff earns no independent milestone.
    """
    count = annotation['total_frames']
    stages = annotation['stages']
    if len(deltas) != count or not all(math.isfinite(x) and x >= 0 for x in deltas):
        raise ValueError('Delta must be finite, nonnegative, and match every original frame')
    if deltas[0] != 0:
        raise ValueError('The first frame must have delta zero')
    if annotation.get('frame_index_base') != 0 or annotation.get('frame_interval') != 'inclusive':
        raise ValueError('Expected 0-based inclusive annotations')
    stage_count = len(stages)
    if stage_count not in (12, 13) or [s['stage_id'] for s in stages] != list(range(1, stage_count+1)):
        raise ValueError('Expected the agreed 12 or 13 stages')
    milestones = MILESTONES_13 if stage_count == 13 else MILESTONES
    previous_end = -1
    totals, targets, last_stages = {}, {}, {}
    for stage in stages:
        sid, start, end = stage['stage_id'], stage['start_frame'], stage['end_frame']
        if not previous_end < start <= end < count:
            raise ValueError('Overlapping or invalid stage bounds')
        previous_end = end
        if sid in milestones:
            total = math.fsum(deltas[start:end+1])
            if not math.isfinite(total) or total <= 0:
                raise ValueError(f'Stage {sid} has zero/nonfinite delta; review instead of inventing a time label')
            totals[sid] = total
            key = milestones[sid]
            if not (stage_count == 13 and sid == 9):
                targets[key], last_stages[key] = total, sid
    denominator = math.fsum(targets.values())
    credit = dict.fromkeys(targets, 0.0)
    progress = [0.0] * count
    stage_ids = [0] * count
    events, stage_reports = [], []
    cursor, current = 0, 0.0
    for stage in stages:
        sid, start, end = stage['stage_id'], stage['start_frame'], stage['end_frame']
        progress[cursor:start] = [current/denominator] * (start-cursor)
        before = current/denominator
        if sid in ROLLBACKS:
            key = ROLLBACKS[sid]
            removed = credit[key]
            if removed <= 0:
                raise ValueError(f'Stage {sid} cannot revoke absent credit')
            credit[key] = 0.0
            current = math.fsum(credit.values())
            progress[start:end+1] = [current/denominator] * (end-start+1)
            events.append({'stage_id':sid, 'frame_index':start, 'milestone':key,
                           'pt_before':before, 'pt_after':current/denominator,
                           'removed_credit':removed, 'removed_normalized_credit':removed/denominator})
        else:
            key = milestones[sid]
            continuing_recovery = stage_count == 13 and sid == 9
            if credit[key] != 0 and not continuing_recovery:
                raise ValueError(f'Repeated milestone {key} has not been revoked')
            target, total = targets[key], totals[sid]
            if stage_count == 13 and sid in (8, 9):
                target *= totals[sid] / math.fsum([totals[8], totals[9]])
            baseline = math.fsum(credit.values())
            cumulative = 0.0
            for frame in range(start, end+1):
                cumulative += deltas[frame]
                earned = target * min(cumulative/total, 1.0)
                progress[frame] = (baseline+earned)/denominator
            credit[key] += target
            if continuing_recovery:
                credit[key] = targets[key]  # exact restoration at placement completion
            current = math.fsum(credit.values())
            progress[end] = current/denominator
        stage_ids[start:end+1] = [sid] * (end-start+1)
        stage_reports.append({'stage_id':sid, 'stage_name':stage['stage_name'],
                              'frame_range_inclusive':[start,end], 'milestone':key,
                              'kind':'rollback' if sid in ROLLBACKS else ('restore' if stage_count == 13 and sid in (8, 9, 10) else 'advance'),
                              'raw_delta_sum':math.fsum(deltas[start:end+1]),
                              'positive_target_credit':0.0 if sid in ROLLBACKS else target,
                              'pt_before':before, 'pt_end':current/denominator})
        cursor = end+1
    progress[cursor:] = [current/denominator] * (count-cursor)
    if not math.isclose(progress[-1], 1.0, abs_tol=1e-12):
        raise ValueError('Final progress is not one')
    if not all(math.isfinite(p) and -1e-12 <= p <= 1+1e-12 for p in progress):
        raise ValueError('Progress is outside [0,1]')
    drops = [i for i in range(1,count) if progress[i] < progress[i-1]-1e-12]
    if drops != [event['frame_index'] for event in events]:
        raise ValueError('Unexpected rollback locations')
    return progress, stage_ids, {
        'stage_count': stage_count,
        'recovery_rule': ('Stage 8 pickup alone defines cup1 pickup credit; stages 8 and 9 jointly restore that credit in proportion to their visual delta sums; stage 9 adds no independent milestone'
                          if stage_count == 13 else 'Stage 8 jointly defines and restores cup1 pickup credit'),
        'normalization_denominator':denominator,
        'milestones':{key:{'last_positive_stage_id':last_stages[key], 'target_credit':value,
                           'normalized_weight':value/denominator} for key,value in targets.items()},
        'rollback_events':events, 'stages':stage_reports,
    }


def prepare_batch(dataset_root, output_dir=None, episodes=None):
    root = Path(dataset_root).resolve()
    output = Path(output_dir).resolve() if output_dir else root/'progress'
    paths = sorted((root/'videos/annotation').glob('episode_*.json'))
    expected = [f'episode_{i:06d}' for i in range(35)]
    if episodes is not None:
        if not episodes or len(set(episodes)) != len(episodes) or not set(episodes) <= set(expected):
            raise ValueError('Expected unique valid episode IDs')
        expected = sorted(episodes)
        paths = [p for p in paths if p.stem in expected]
    if [p.stem for p in paths] != expected:
        raise ValueError(f'Missing requested annotations: {expected}')
    prepared = []
    for path in paths:
        annotation = json.loads(path.read_text())
        delta_path = root/'delta'/f'{path.stem}-delta.txt'
        deltas = [float(line.split()[0]) for line in delta_path.read_text().splitlines() if line.strip()]
        video_path = root/'videos/main_view'/f'{path.stem}.mp4'
        if sha256(video_path) != annotation['video_sha256']:
            raise ValueError(f'Video changed: {video_path}')
        progress, stage_ids, details = calculate_progress(deltas, annotation)
        prepared.append((path,annotation,delta_path,deltas,progress,stage_ids,details))
    output.mkdir(parents=True,exist_ok=True)
    manifest_rows = []
    for path,annotation,delta_path,deltas,progress,stage_ids,details in prepared:
        name = path.stem
        csv_path = output/f'{name}.progress.csv'
        uncertainty = set()
        for boundary in annotation['boundary_reviews']:
            a,b = boundary['review_source_frame_range_inclusive']
            uncertainty.update(range(a,b+1))
        names = {s['stage_id']:s['stage_name'] for s in annotation['stages']}
        names[0] = '空闲或退手复位'
        temporary = csv_path.with_suffix('.csv.tmp')
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame_index','time_seconds','stage_id','stage_name','delta','pt','boundary_needs_review'])
            writer.writerows((i,i/annotation['fps'],stage_ids[i],names[stage_ids[i]],deltas[i],progress[i],int(i in uncertainty))
                             for i in range(len(progress)))
        temporary.replace(csv_path)
        metadata = {
            'format_version':'flipcup-last-attempt-rollback-progress-v2' if details['stage_count'] == 13 else 'flipcup-last-attempt-rollback-progress-v1',
            'episode':name, 'fps':annotation['fps'], 'total_frames':len(progress),
            'annotation_path':str(path), 'annotation_sha256':sha256(path),
            'video_sha256':annotation['video_sha256'], 'delta_path':str(delta_path),
            'delta_sha256':sha256(delta_path), 'progress_csv_sha256':sha256(csv_path),
            'delta_rule':'Adjacent original-frame frozen Qwen3-VL-4B post-merger mean-feature L2; first delta=0',
            'progress_rule':'Each semantic milestone uses its last positive attempt delta sum; same inclusive-frame delta convention as the old episode_to_progress; divide by final retained credit',
            'rollback_timing':'Instantaneous at stage 6 and 7 first original frame; stages 6/7 earn no positive credit',
            'idle_rule':'Hold previous PT; initial idle=0; final idle=1',
            'boundary_reviews':annotation['boundary_reviews'], **details,
        }
        (output/f'{name}.metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        manifest_rows.append({'episode':name,'frames':len(progress),'csv':csv_path.name,
                              'metadata':f'{name}.metadata.json','rollback_events':details['rollback_events']})
    manifest = {'format_version':'flipcup-last-attempt-rollback-progress-v2' if all(x[-1]['stage_count'] == 13 for x in prepared) else 'flipcup-last-attempt-rollback-progress-v1',
                'episode_count':len(manifest_rows),'total_frames':sum(x['frames'] for x in manifest_rows),
                'rollback_count':sum(len(x['rollback_events']) for x in manifest_rows),
                'all_checks_passed':True,'episodes':manifest_rows}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--episodes',nargs='+',help='Explicit subset for review; defaults to all 35')
    args = parser.parse_args()
    result = prepare_batch(args.dataset_root,args.output_dir,args.episodes)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False))
