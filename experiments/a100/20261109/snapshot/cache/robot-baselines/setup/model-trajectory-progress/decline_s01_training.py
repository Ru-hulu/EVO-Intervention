"""Copy the frozen S01 v2 dataset and add visual-integral wrong-action penalties."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

BASE = Path('/cache/robot-baselines/data')
SOURCE = BASE / 'ProPriMem-S01-IMPRINT-Mixed-20261106-v2'
OUTPUT = BASE / 'ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline'
TASK = Path('State/01-cover_uncover_blocks')
POLICY = 's01_wrong_approach_decline_visual_integral_v1'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def integral(delta, start, end):
    values = delta[start:end]
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    total = math.fsum(values)
    assert total > 0, 'No visual evidence: do not substitute elapsed time'
    accum, result = 0., []
    for value in values:
        accum += value
        result.append(min(1., accum / total))
    result[-1] = 1.
    return result


def decline(curve, delta, start, end, failure=None, restored=None):
    """Error is [start,end); recovery ends at the inclusive restored frame."""
    n = len(curve)
    assert len(delta) == n and 0 < start < end < n
    assert (failure is not None) != (restored is not None)
    assert max(curve[start:end]) - min(curve[start:end]) < 1e-9
    anchor = curve[start - 1]
    amount = min(.10, anchor / 2)
    assert amount > 0
    result = list(curve)
    for f, ratio in zip(range(start, end), integral(delta, start, end)):
        result[f] = anchor - amount * ratio
    if failure is not None:
        assert end == failure
        result[failure:] = [0.] * (n - failure)
    else:
        assert end <= restored < n
        for f, ratio in zip(range(end, restored + 1), integral(delta, end, restored + 1)):
            result[f] = curve[f] - amount * (1 - ratio)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in result)
    assert all(b <= a + 1e-10 for a, b in zip(result[start-1:end-1], result[start:end]))
    if restored is not None:
        assert all(b >= a - 1e-10 for a, b in zip(result[end-1:restored], result[end:restored+1]))
        assert result[restored:] == curve[restored:]
    return result, amount


def build(out, baseline_manifest):
    shutil.copytree(SOURCE, out)  # Independent files, including videos; no shared writable inodes.
    verification = out / 'verification'
    verification.rename(out / 'source-v2-verification')
    verification.mkdir()
    save(verification / 'source-v2-manifest.json', baseline_manifest)
    task = out / TASK
    rule = dict(policy=POLICY, parent_dataset=str(SOURCE), max_penalty=.10,
        penalty_cap='min(0.10, progress_before_error / 2)',
        decline='anchor - penalty * cumulative visual delta / error interval visual delta',
        failure='first recorded native fail_flag=True: zero permanently',
        recovery='v2 progress minus remaining penalty; restore penalty by visual delta through correct-target-reached frame',
        intervals='error [start,end); recovery [end,restored+1)',
        time_based_progress=False, native_reward=False,
        abnormal_only=True, unmodified_groups=['IMPRINT success', 'timeout without wrong-target evidence'])
    rule_path = task / 'code/rules/wrong_approach_decline.json'
    save(rule_path, rule)
    index = [json.loads(line) for line in (task / 'metadata.jsonl').read_text().splitlines()]
    report = []
    for row in index:
        ev = task / row['evidence_path']
        progress = read(ev / 'progress.json')
        old = progress['dense_progress']
        if row['mixture_group'] != 'abnormal':
            continue
        boundary, intervention = read(ev / 'boundaries.json'), read(ev / 'intervention.json')
        if row['trajectory_source'] == 'vla':
            if not boundary['wrong_frames']:
                assert boundary['failure_frame'] is None
                continue
            start = boundary['wrong_frames'][0]
        else:
            start = intervention['pre_lift_window'][0]
        states = [json.loads(line) for line in (ev / 'states.jsonl').read_text().splitlines()]
        failure = next((i for i, s in enumerate(states) if s['fail_flag']), None)
        assert failure == intervention['native_failure_frame']
        end = failure if failure is not None else intervention['correction_start_frame']
        restored = intervention.get('correct_target_reached_frame') if failure is None else None
        delta = read(task / 'deltas' / f"episode_{row['episode_index']:06d}.json")['deltas']
        curve, amount = decline(old, delta, start, end, failure, restored)
        entry = dict(sample_id=row['sample_id'], episode=row['episode_index'],
            trajectory_source=row['trajectory_source'], error_range=[start, end],
            recovery_range=[end, restored + 1] if restored is not None else None,
            native_failure_frame=failure, penalty=amount, before=old[start-1], lowest_before_outcome=curve[end-1])
        report.append(entry)
        # Keep original geometric boundaries as evidence; this explicit override
        # governs the alternative p curve without changing native task events.
        boundary['progress_policy'] = POLICY
        boundary['progress_override'] = entry
        save(ev / 'boundaries.json', boundary)
        progress.update(parent_policy=progress['policy'], policy=POLICY, dense_progress=curve,
            label_rule_path='code/rules/wrong_approach_decline.json', rule_sha256=sha(rule_path),
            boundary_json_sha256=sha(ev / 'boundaries.json'), penalty_intervals=[entry],
            freeze_frames=[], negative_intervals=[[start, end]],
            real_progress_regression=True, regression_semantics='training penalty for wrong target; not proof of physical damage',
            source='错误动作视觉积分扣减；原生失败归零；纠正动作视觉积分恢复',
            first_issue=dict(frame=start, kind='wrong_approach_decline'))
        for event in progress.get('events', []):
            if 'progress' in event:
                event['progress'] = curve[event['frame']]
        progress.setdefault('events', []).append(dict(frame=start, kind='wrong_approach_decline', progress=curve[start]))
        progress['events'].sort(key=lambda event: event['frame'])
        save(ev / 'progress.json', progress)
        ann = read(task / row['annotation_path'])
        descriptions = [''] * len(curve)
        for stage in ann['stages']:
            descriptions[stage['start_frame']:stage['end_frame']] = [stage['state_description']] * (stage['end_frame'] - stage['start_frame'])
        descriptions[start:end] = ['错误目标动作：progress 随视觉变化逐步下降'] * (end - start)
        if restored is not None:
            descriptions[end:restored+1] = ['纠正并接近正确目标：progress 逐步恢复并继续增长'] * (restored + 1 - end)
        starts = [0] + [f for f in range(1, len(curve)) if descriptions[f] != descriptions[f-1]]
        ann['stages'] = [dict(stage_id=i, start_frame=a, end_frame=starts[i+1] if i+1<len(starts) else len(curve), state_description=descriptions[a]) for i,a in enumerate(starts)]
        ann.update(progress=progress, num_stages=len(starts))
        save(task / row['annotation_path'], ann)
        row.update(progress_policy=POLICY, annotation_sha256=sha(task / row['annotation_path']), num_stages=len(starts))
        lp = out / 'labels' / TASK / f"episode{row['episode_index']}.json"
        label = read(lp)
        label.update(dense_progress=curve, progress_label=POLICY)
        for frame in label['frames']:
            frame['progress'] = curve[frame['index']]
        save(lp, label)
    (task / 'metadata.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in index))
    summary = read(out / 'summary.json')
    for key in ['existing_progress_curves_changed', 'new_progress_curves', 'label_alignment_cases_passed', 'vla_progress_recomputed', 'scripted_progress_preserved_and_verified']:
        summary.pop(key, None)
    summary.update(parent_dataset=str(SOURCE), dataset_variant=POLICY,
        changed_progress_trajectories=len(report), unchanged_progress_trajectories=len(index)-len(report),
        decline_then_native_failure=sum(r['native_failure_frame'] is not None for r in report),
        decline_then_scripted_recovery=sum(r['recovery_range'] is not None for r in report))
    save(out / 'summary.json', summary)
    save(verification / 'decline-relabel.json', report)
    config = read(out / 'training-data.json')
    config.update(data=str(OUTPUT), labels=str(OUTPUT/'labels'))
    save(out / 'training-data.json', config)
    (out / 'train-data.yaml').write_text(f'data: {OUTPUT}\nlabels: {OUTPUT}/labels\nsuites: State\npause_prob: 0.2\nrepeat_boost: 1.0\n')
    shutil.copy2(Path(__file__), out/'code'/Path(__file__).name)
    shutil.copy2(SOURCE/'README.md', out/'README-v2-source.md')
    (out/'README.md').write_text('''# S01 错误接近下降标签实验 v3

这是 v2 的独立文件副本，原版未修改。138 条轨迹、44 成功+94 异常，每轮47成功+94异常、暂停增强20%保持不变。没有启动训练或加入可视化工具。

只改异常组中有错误目标证据的93条曲线：63条错误接近→原生失败、30条脚本纠正。44条正常成功与1条未发现错误接近的超时轨迹保持原标签。

错误开始点沿用核验的几何/脚本边界。失败轨迹从首次错误开始连续扣到首次记录的 fail_flag=True 前一帧（原错误检测掩码的短暂中断不恢复 p）。每段扣 min(0.10, 错误前p/2)，按原视频视觉delta累计量分配，不按时间加减分；原生失败帧起永久为0。

脚本纠正：从错误开始扣到纠正开始前一帧；随后按纠正动作的视觉delta逐步返还扣分，到正确目标到达帧与v2曲线重新汇合，之后相同。成功末帧仍为1。下降是训练标签约定，不声称物理成果已受损，不能当作RMBench原生reward。脚本恢复不能称为VLA自发恢复。

静止复制的暂停增强仍复制同一个p，不人为按等待时间降低；本规则的视觉积分需要离线完整区间。实际停滞/下降触发介入阈值尚未评估。

RMBench原生判断：仅在全部盖子回到初始位置或覆盖位置附近（XY距离<=0.035、Z<=0.742）时判断覆盖状态；空中/偏离时记tmp暂不推进。覆盖要求XY各差<0.03且Z<0.742。稳定状态既不等于当前合法状态、也不等于下一合法状态时置fail_flag=True；不会仅因接近错误盖子判失败。代码快照在code/rmbench_cover_blocks.py。

规则见 State/01-cover_uncover_blocks/code/rules/wrong_approach_decline.json；逐轨迹区间、扣分和验证见verification/。旧边界作为几何证据保留，progress_override指定新p规则，原基线规则不应直接套用于本版。source-v2-verification仅是父版本审计，当前标签验收以verification为准。边界沿用AI关键帧+几何复核，非人工逐帧验收。
''')
    assert manifest(SOURCE) == baseline_manifest, 'Original v2 changed'
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    before = manifest(SOURCE)
    with tempfile.TemporaryDirectory(prefix='.s01-decline-', dir=BASE) as temp:
        stage = Path(temp)/'dataset'
        build(stage, before)
        stage.rename(OUTPUT)
