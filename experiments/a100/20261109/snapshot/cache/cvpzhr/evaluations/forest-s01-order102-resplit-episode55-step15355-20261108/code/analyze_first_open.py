"""Compare first versus second uncover failures without changing labels or training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE = Path('/cache/cvpzhr/evaluations')
OUT = BASE / 'forest-s01-first-open-diagnostic-20261106'
PREVIOUS = BASE / 'forest-s01-critical-val12-step13224-20261106'
SOURCE = Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline/State/01-cover_uncover_blocks')


def covered(frame):
    return ''.join('1' if abs(c[0]-b[0]) < .03 and abs(c[1]-b[1]) < .03 and c[2] < .742 else '0'
                   for c, b in zip(frame['covers'], frame['blocks']))


def longest_run(points, threshold, fps, last_frame):
    longest = current = 0.0
    for i, point in enumerate(points):
        end = points[i+1]['frame_index'] if i+1 < len(points) else last_frame+1
        current = current + (end-point['frame_index'])/fps if point['predicted_pt'] > threshold else 0.0
        longest = max(longest, current)
    return longest


def main():
    split = json.loads(Path('/cache/cvpzhr/setup/forest-s01-decline-20261106/split.json').read_text())['records']
    predictions = defaultdict(list)
    for folder in [PREVIOUS, *[OUT/f'gpu{i}' for i in range(4)]]:
        assert (folder/'metrics.json').exists(), folder
        assert json.loads((folder/'metrics.json').read_text())['step'] == 13224
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            point = json.loads(line)
            predictions[point['episode_index']].append(point)
    results, counts = [], Counter()
    for row in split:
        if row['trajectory_source'] != 'vla' or row['task_success']:
            continue
        boundaries = json.loads((SOURCE/row['subtask_boundary_path']).read_text())
        geometry = json.loads((SOURCE/row['evidence_path']/'geometry.json').read_text())
        frames = geometry['frames']
        first_failure = next((f for f in frames if f['fail']), None)
        if first_failure is None:
            continue
        expected = geometry['open_order']
        states = [covered(frame) for frame in frames]
        all_covered = next(i for i, state in enumerate(states) if state == '111')
        lifts = []
        for lid in range(3):
            frame = next((i for i in range(all_covered+1, len(states)-4)
                          if all(states[t][lid] == '0' for t in range(i, i+5))), None)
            if frame is not None:
                lifts.append((frame, lid))
        lifts.sort()
        actual = [lid for _, lid in lifts]
        rank = next((i+1 for i, (a,b) in enumerate(zip(actual, expected)) if a != b), None)
        native_rank = first_failure['phase'] - 2
        assert rank == native_rank and rank in (1, 2), (row['episode_index'], rank, native_rank)
        counts[(row['split'], tuple(expected), tuple(actual))] += 1
        if row['split'] not in ('val', 'validation'):
            continue
        points = sorted(predictions[row['episode_index']], key=lambda r: r['frame_index'])
        assert points and len({p['frame_index'] for p in points}) == len(points)
        fail_frame = boundaries['failure_frame']
        assert fail_frame == first_failure['frame']
        failed = [point for point in points if point['frame_index'] >= fail_frame]
        assert all(point['target_pt'] == 0 for point in failed)
        high20 = [p for p in failed if p['predicted_pt'] > .2]
        high80 = [p for p in failed if p['predicted_pt'] > .8]
        state, expected_states = '111', []
        for lid in expected:
            state = state[:lid]+'0'+state[lid+1:]
            expected_states.append(state)
        joined = [p for p in failed if states[p['frame_index']] in expected_states[:-1]]
        results.append(dict(episode=row['episode_index'], category=row['category'], sample_id=row['sample_id'],
            renderer_spp=row.get('renderer_spp'), first_wrong_open_rank=rank,
            expected_open_order=expected, actual_open_order=actual, lift_frames=lifts,
            failure_frame=fail_frame, fps=row['fps'], post_failure_samples=len(failed),
            max_predicted_progress=max(p['predicted_pt'] for p in failed),
            post_failure_mae_pp=sum(p['predicted_pt'] for p in failed)/len(failed)*100,
            above_20_count=len(high20), above_80_count=len(high80),
            longest_above20_seconds=longest_run(failed,.2,row['fps'],row['total_frames']-1),
            longest_above80_seconds=longest_run(failed,.8,row['fps'],row['total_frames']-1),
            above20_frame_range=[high20[0]['frame_index'],high20[-1]['frame_index']] if high20 else None,
            revisits_valid_nonterminal_state=len(joined),
            above20_in_valid_nonterminal_state=sum(p['predicted_pt']>.2 for p in joined)))
    assert len(results) == 12
    groups = {}
    for rank in (1, 2):
        rows = [r for r in results if r['first_wrong_open_rank'] == rank]
        n = sum(r['post_failure_samples'] for r in rows)
        groups[str(rank)] = dict(trajectories=len(rows),
            sustained_over20_episodes=sum(r['longest_above20_seconds'] >= 1 for r in rows),
            sustained_over80_episodes=sum(r['longest_above80_seconds'] >= 1 for r in rows),
            above20_frames=sum(r['above_20_count'] for r in rows), post_failure_frames=n,
            frame_rate_above20=sum(r['above_20_count'] for r in rows)/n)
    checkpoint = PREVIOUS/'checkpoint.pt'
    with checkpoint.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(checkpoint_sha256=sha, groups=groups, episodes=results,
        order_counts=[dict(split=k[0], expected=k[1], actual=k[2], count=v) for k,v in sorted(counts.items())],
        thresholds=dict(high_progress=.2, very_high_progress=.8, sustained_seconds=1),
        notes=['All 12 natural VLA failures in validation, not random independent repetitions.',
               'Episode 137 is the historical 32-spp pilot; report separately when needed.',
               'Geometry uses native cover criteria; lift detection requires five consecutive frames.',
               'A revisit matches cover/uncover configuration, not necessarily every pixel or robot pose.',
               'Predictions come from the same weighted FOREST step13224; no training or label changes.'])
    report['same_renderer_groups'] = {}
    for rank in (1, 2):
        same = [r for r in results if r['renderer_spp'] == 8 and r['first_wrong_open_rank'] == rank]
        report['same_renderer_groups'][str(rank)] = dict(trajectories=len(same),
            sustained_over20_episodes=sum(r['longest_above20_seconds'] >= 1 for r in same),
            sustained_over80_episodes=sum(r['longest_above80_seconds'] >= 1 for r in same))
    scene_counts, missing, script_details = Counter(), [], []
    for row in split:
        evidence = SOURCE/row['evidence_path']
        geometry = evidence/'geometry.json'
        result = evidence/'result.json'
        if geometry.exists():
            order = json.loads(geometry.read_text()).get('open_order')
        elif result.exists():
            order = json.loads(result.read_text()).get('memory', {}).get('color_order_positions')
        else:
            order = None
        if order is None:
            missing.append(row['episode_index'])
        if order != [1, 2, 0]:
            continue
        scene_counts[(row['split'], row['category'])] += 1
        if row['split'] == 'train' and row['trajectory_source'] != 'vla':
            saved = json.loads(result.read_text())
            states = [json.loads(line) for line in (SOURCE/row['states_path']).read_text().splitlines()]
            script_details.append(dict(episode=row['episode_index'], category=row['category'],
                memory=saved['memory'], final_covered=states[-1]['covered'],
                fail_frame=next((s['frame'] for s in states if s['fail_flag']), None)))
    report['matching_scene_order_counts'] = [dict(split=k[0], category=k[1], count=v)
                                            for k,v in sorted(scene_counts.items())]
    report['scene_order_unresolved_episodes'] = missing
    report['matching_scene_script_training'] = script_details
    (OUT/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
