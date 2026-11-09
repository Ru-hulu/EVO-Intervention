# ruff: noqa: RUF001, RUF003
"""将 MemLong 的累计事件状态转换为与 RMBench 一致的逐轨迹监督 JSON。"""
import argparse
import json
from pathlib import Path

from qwen_training.datasets.memlong.tasks import TASKS, state_counts
from qwen_training.datasets.rmbench.labels import MAX_COUNT, sample_frames


def episode_labels(annotation, task, sample_fps=5):
    total, fps = annotation['total_frames'], annotation['fps']
    stages = annotation['stages']
    if (annotation['frame_index_base'] != 0 or annotation['frame_range_convention'] != '[start_frame, end_frame)'
            or total < 1 or fps <= 0 or not 0 < sample_fps <= fps):
        raise ValueError('需要从 0 开始的左闭右开帧区间和有效帧率')
    if annotation['num_stages'] != len(stages) or len(stages) < 2:
        raise ValueError('阶段数量无效')
    counts, trace, previous_end = [0] * len(TASKS[task][2]), [], 0
    for order, stage in enumerate(stages):
        start, end = stage['start_frame'], stage['end_frame']
        if stage['stage_id'] != order or start != previous_end or not start < end <= total:
            raise ValueError('阶段必须连续、非空并覆盖整条视频')
        previous_end = end
        current = state_counts(task, stage['state_description'], order)
        if order == 0:
            if any(current):
                raise ValueError('首段必须是无完成事件状态')
            continue
        changes = [j for j, (old, new) in enumerate(zip(counts, current, strict=True)) if old != new]
        if len(changes) != 1 or current[changes[0]] != counts[changes[0]] + 1:
            raise ValueError(f'每次状态切换应恰好新增一个完成事件：{stage}')
        step = changes[0]
        trace.append({'step': step, 'instance': current[step],
                      'start': trace[-1]['end'] + 1 if trace else 0, 'end': start,
                      'annotation': stage['state_description']})
        counts = current
    if previous_end != total or max(counts) > MAX_COUNT:
        raise ValueError('视频覆盖不完整或累计次数超出模型容量')
    frames = []
    for index in sample_frames(total, fps, sample_fps):
        completed = [0] * len(counts)
        for event in trace:
            completed[event['step']] += index >= event['end']
        active = next((event for event in trace if index <= event['end']), trace[-1])
        elapsed = ((index - active['start']) / (active['end'] - active['start'] + 1)
                   if index < active['end'] else 0)
        frames.append({'index': index, 'stage': active['step'], 'counts': completed,
                       'progress': round((sum(completed) + elapsed) / len(trace), 6)})
    # 固定任务的重复动作也展示每个实例；场景次数任务只展示每种动作第一次。
    demo = [{'step': event['step'], 'frame': event['end']} for event in trace
            if TASKS[task][0] == 'fixed' or event['instance'] == 1]
    return {'steps': list(TASKS[task][2]), 'step_counts': counts, 'trace': trace, 'demo': demo, 'frames': frames}


def video_info(path):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total = stream.frames or sum(1 for _ in container.decode(stream))
    return total, fps


def build(root, out, tasks=None, sample_fps=5, inspect_video=video_info):
    root, out = Path(root), Path(out)
    split, written, videos = {}, 0, 0
    for task in sorted(tasks or TASKS):
        directory = root / task
        annotations = sorted(directory.glob('annotations/*.annotations.json'))
        if not annotations:
            raise ValueError(f'{directory}: 没有阶段标注')
        episodes = []
        for path in annotations:
            annotation = json.loads(path.read_text())
            source = directory / annotation['video_file']
            if source.parent != directory or not source.is_file():
                raise ValueError(f'{path}: 对应视频不存在或文件名无效')
            total, fps = inspect_video(source)
            if total != annotation['total_frames'] or abs(fps - annotation['fps']) > 0.01:
                raise ValueError(f'{source}: 视频帧数或帧率与标注不一致')
            episode = int(source.stem.removeprefix('episode_'))
            labels = episode_labels(annotation, task, sample_fps)
            kind, instruction, _ = TASKS[task]
            record = {'dataset': 'memlong', 'task': task, 'episode': episode, 'kind': kind,
                      'source': str(source.relative_to(root)), 'source_fps': fps, 'total_frames': total,
                      'cameras': ['video'], 'instruction': instruction, **labels}
            target = out / task / f'episode{episode}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, ensure_ascii=False))
            episodes.append(episode)
            written += 1
        n = len(episodes)
        if n < 5:
            raise ValueError(f'{task}: 至少需要 5 条标注轨迹以隔离示范、训练和验证')
        pool, val = min(4, max(2, n // 4)), max(1, n // 5)
        split[task] = {'pool': episodes[:pool], 'eval_demo': episodes[pool],
                       'train': episodes[pool + 1:-val], 'val': episodes[-val:]}
        videos += len(list(directory.glob('*.mp4')))
    (out / '_splits.json').write_text(json.dumps(split, ensure_ascii=False, indent=2))
    print(f'wrote {written} annotated episodes; skipped {videos - written} videos without annotations')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tasks', nargs='+', choices=sorted(TASKS))
    parser.add_argument('--sample-fps', type=float, default=5)
    args = parser.parse_args()
    build(args.root, args.out, args.tasks, args.sample_fps)
