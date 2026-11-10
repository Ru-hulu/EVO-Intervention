"""ProPriMem：按源视频划分、全视频视觉变化 PT、保留插螺栓 PT、5 FPS 因果缓存。"""

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from mem_lite.datasets.prepare_progress import extract_deltas, file_sha256
from mem_lite.datasets.prepare_recovery import TASK_PROMPT, probe_video
from mem_lite.datasets.prepare_stream import DEFAULT_MEMORY_CONFIG, build_episode, iter_observations
from mem_lite.datasets.views import VIEW_POLICY
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder
from qwen_training.delta_to_progress import episode_to_progress


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def whole_video_progress(deltas):
    """首帧0、末帧1；零变化视频报错，绝不默默换成时间插值。"""
    delta = np.asarray(deltas, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < 2 or delta[0] != 0 or not np.isfinite(delta).all() or (delta < 0).any():
        raise ValueError('无效的逐帧视觉变化量，需要复核')
    cumulative = np.cumsum(delta, dtype=np.float64)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 1e-8:
        raise ValueError('整条视频视觉变化为零或过小，需要用户确认，不生成替代 PT')
    result = cumulative / total
    result[-1] = 1.0
    return result


def split_videos(video_ids, task, seed=42):
    video_ids = sorted(video_ids)
    count = max(1, min(len(video_ids)-1, round(len(video_ids)*0.2)))
    validation = set(random.Random(f'{seed}:{task}').sample(video_ids, count))
    return {video: 'validation' if video in validation else 'train' for video in video_ids}


def check_views(paths, expected_frames):
    info = [probe_video(path) for path in paths if path is not None]
    if any(x['frames'] != expected_frames or abs(x['fps']-info[0]['fps']) > 1e-3
           or abs(x['start_time']-info[0]['start_time']) > 1e-6 for x in info):
        raise ValueError(f'视角不同步或帧数与索引不符，需要复核：{paths[0]}')
    return info[0]['fps']


def write_progress(path, frames, deltas, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'delta', 'pt'])
        writer.writerows(zip(frames, map(float, deltas), map(float, values), strict=True))
    temp.replace(path)


def prepare_index(dataset_root, output_dir, legacy_dir, seed=42):
    root, output, legacy = Path(dataset_root).resolve(), Path(output_dir).resolve(), Path(legacy_dir).resolve()
    if (output/'index.json').exists():
        raise FileExistsError('索引已存在，不覆盖已有划分')
    sources = []
    for index in sorted(root.glob('*/*/metadata.jsonl')):
        task = str(index.parent.relative_to(root))
        for line in index.read_text().splitlines():
            row = json.loads(line)
            views = row['video_views']
            ordered = ['main_view', 'wrist_view', None] if 'wrist_view' in views else ['main_view', 'left_view', 'right_view']
            paths = [str(index.parent/views[v]['video_path']) if v in views else None for v in ordered]
            fps = check_views(paths, row['total_frames'])
            prompt = row.get('source_instruction') or row.get('task_prompt') or row['task_name']
            sources.append(dict(task=task, video_id=f"episode_{int(row['episode_index']):06d}",
                                frames=row['total_frames'], fps=fps, videos=paths,
                                task_prompt=prompt, pt_method='whole-video-cumulative-visual-delta-v1',
                                video_sha256=[views[v]['video_sha256'] if v in views else None for v in ordered]))
    old_manifest = json.loads((legacy/'insert-screw-recover/manifest.json').read_text())
    old_sources = {r['video_id']:r for r in old_manifest['sources']}
    for task in ('insert-screw', 'insert-screw-recover'):
        base = root/task
        multiview = json.loads((base/'multiview_manifest.json').read_text())
        for record in multiview['episodes']:
            video_id = record['episode_id']
            if task == 'insert-screw-recover' and video_id == 'file-019':
                continue  # 用户确认沿用旧排除；原始数据保留。
            paths = [str(base/record['views'][v]['path']) for v in ('main', 'left_wrist', 'right_wrist')]
            fps = check_views(paths, record['num_frames'])
            annotation = base/'splitlable'/f'{video_id}.JSON'
            delta = base/'delta'/f'{video_id}-delta.txt'
            item = dict(task=task, video_id=video_id, frames=record['num_frames'], fps=fps, videos=paths,
                        task_prompt=TASK_PROMPT, pt_method='legacy-rollback-unchanged',
                        video_sha256=[record['views'][v]['sha256'] for v in ('main', 'left_wrist', 'right_wrist')],
                        annotation=str(annotation), annotation_sha256=file_sha256(annotation),
                        delta=str(delta), delta_sha256=file_sha256(delta))
            if task == 'insert-screw-recover':
                original = old_sources[video_id]
                for key in ('annotation_sha256', 'delta_sha256', 'video_sha256'):
                    if item[key] != original[key]:
                        raise ValueError(f'旧 PT 的源数据发生变化：{task}/{video_id}/{key}')
            sources.append(item)
    tasks = sorted({r['task'] for r in sources})
    for task in tasks:
        group = [r for r in sources if r['task'] == task]
        splits = split_videos([r['video_id'] for r in group], task, seed)
        for row in group:
            row['split'] = splits[row['video_id']]
            row['source_id'] = f"{task}/{row['video_id']}"
    sources.sort(key=lambda r:r['source_id'])
    # 按视频负载分配四个独立 worker；单个原视频的所有轮次留在同一个 worker。
    loads = [0.0]*4
    for row in sorted(sources, key=lambda r:r['frames']/r['fps'], reverse=True):
        worker = min(range(4), key=loads.__getitem__)
        row['worker'] = worker
        loads[worker] += row['frames']/row['fps']
    result = dict(format_version='proprimem-forest-preparation-v1', dataset_root=str(root), seed=seed,
                  created_at=datetime.now(timezone.utc).isoformat(), legacy_dir=str(legacy),
                  split_policy='per task, original video, nearest integer 20% validation; no segment leakage',
                  exclusions=[dict(source_id='insert-screw-recover/file-019', reason='user-approved legacy exclusion')],
                  view_policy=VIEW_POLICY, source_video_count=len(sources),
                  splits=dict(Counter(r['split'] for r in sources)),
                  tasks={t:dict(Counter(r['split'] for r in sources if r['task']==t)) for t in tasks}, sources=sources)
    write_json(output/'index.json', result)
    return result


def label_source(row, output, legacy, encoder, batch_size, pt_max_pixels):
    labels = output/'labels'/row['source_id']
    metadata = labels/'metadata.json'
    if metadata.exists():
        existing = json.loads(metadata.read_text())
        for unit in existing['units']:
            if file_sha256(Path(unit['progress_csv'])) != unit['progress_sha256']:
                raise ValueError(f'已生成 PT 被修改：{unit["unit_id"]}')
        return existing['units']
    units = []
    if row['pt_method'] == 'whole-video-cumulative-visual-delta-v1':
        model, processor, vision_hash = encoder
        deltas, info = extract_deltas(row['videos'][0], model, processor, batch_size, pt_max_pixels)
        if len(deltas) != row['frames']:
            raise ValueError(f'解码帧数与源索引不符：{row["source_id"]}')
        values = whole_video_progress(deltas)
        path = labels/'progress.csv'
        write_progress(path, range(len(values)), deltas, values)
        units.append(dict(unit_id=row['source_id'], frame_range=[0,len(values)], progress_csv=str(path),
                          progress_sha256=file_sha256(path)))
        details = dict(vision_sha256=vision_hash, feature='post-merger tokens mean, FP32',
                       delta='adjacent-frame L2, FP32; all original frames', pt_max_pixels=pt_max_pixels,
                       delta_sum=float(np.sum(deltas)), first_pt=float(values[0]), last_pt=float(values[-1]),
                       original_fps=info['fps'], annotations_used=False)
    else:
        annotation = json.loads(Path(row['annotation']).read_text())
        deltas = [float(line.split()[0]) for line in Path(row['delta']).read_text().splitlines() if line.strip()]
        if len(deltas) != row['frames'] or not np.isfinite(deltas).all() or min(deltas)<0:
            raise ValueError(f'原 delta 无效：{row["source_id"]}')
        episodes = [e for t in annotation['tasks'] for e in t['episodes']]
        for number, episode in enumerate(episodes, 1):
            start = episode['subtasks'][0]['30FPSstart_frame']
            end = episode['subtasks'][-1]['30FPSend_frame']+1
            path = labels/f'episode-{number:02d}'/'progress.csv'
            old = legacy/row['task']/'episodes'/row['video_id']/f'episode-{number:02d}'/'progress.csv'
            if old.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, path)
                entries = list(csv.DictReader(path.open()))
                if [int(x['frame_index']) for x in entries] != list(range(start,end)):
                    raise ValueError(f'旧 PT 帧范围不符：{old}')
                values = [float(x['pt']) for x in entries]
            else:
                values, _ = episode_to_progress(deltas, episode['subtasks'], [t['id'] for t in annotation['taskid']])
                write_progress(path, range(start,end), deltas[start:end], values)
            if not np.isfinite(values).all() or min(values)<0 or max(values)>1 or abs(values[-1]-1)>1e-8:
                raise ValueError(f'插螺栓原 PT 超出范围：{row["source_id"]}/{number}')
            units.append(dict(unit_id=f'{row["source_id"]}/episode-{number:02d}', frame_range=[start,end],
                              progress_csv=str(path), progress_sha256=file_sha256(path),
                              copied_legacy_csv=str(old) if old.exists() else None,
                              rollback_transitions=sum(b<a-1e-8 for a,b in zip(values,values[1:]))))
        details = dict(original_delta_preserved=True, original_algorithm='qwen_training.delta_to_progress.episode_to_progress',
                       copied_legacy_csv_count=sum(bool(u['copied_legacy_csv']) for u in units))
    write_json(metadata, dict(source=row, **details, units=units))
    return units


def build_worker(output_dir, model_path, worker, device='cuda', batch_size=16, pt_max_pixels=262144,
                 memory_config=None, event_split_method=None, clip_model_path=None):
    import torch
    from mem_lite.memory.splitting import override_memory_config

    torch.set_num_threads(4)
    output = Path(output_dir).resolve()
    index = json.loads((output/'index.json').read_text())
    sources = [r for r in index['sources'] if r['worker']==worker]
    status_path = output/'workers'/f'{worker}.json'
    started = perf_counter()
    state = dict(worker=worker, phase='loading_vision', total=len(sources), completed=0, samples=0)
    write_json(status_path, state)
    try:
        encoder = load_vision_encoder(model_path, device)
        visual, processor, vision_hash = encoder
        config = override_memory_config(json.loads(Path(memory_config or DEFAULT_MEMORY_CONFIG).read_text()),
                                        event_split_method, clip_model_path)
        origin = dict(method='streamforest-v1', vision_sha256=vision_hash, memory_config=config,
                      sample_fps=5.0, include_episode_last_frame=True, max_pixels=65536, min_pixels=4096,
                      view_policy=VIEW_POLICY)
        stream = StreamMemory(QwenFrameEncoder(visual), processor, config, 65536, 4096)
        origin = stream.bind_history_source(origin)
        for row in sources:
            state.update(phase='labels', current=row['source_id'], elapsed_seconds=perf_counter()-started)
            write_json(status_path, state)
            units = label_source(row, output, Path(index['legacy_dir']), encoder, batch_size, pt_max_pixels)
            state['phase'] = 'history_cache'
            write_json(status_path, state)
            for unit in units:
                directory = output/'cache'/row['split']/unit['unit_id']
                summary = directory/'summary.json'
                if not summary.exists():
                    if directory.exists():
                        shutil.rmtree(directory)  # 只清理由本 worker 创建但尚未完成的单个输出。
                    with Path(unit['progress_csv']).open() as f:
                        progress = {int(r['frame_index']):float(r['pt']) for r in csv.DictReader(f)}
                    observations = iter_observations(row['videos'], progress, sample_fps=5.0,
                                                    start_frame=unit['frame_range'][0], end_frame=unit['frame_range'][1])
                    result = build_episode(observations, stream, directory, row['task_prompt'], unit['unit_id'],
                                           origin, source={**row, **unit})
                else:
                    result = json.loads(summary.read_text())
                    if result['history_source'] != origin:
                        raise ValueError(f'缓存配置不一致：{directory}')
                state['samples'] += result['num_samples']
            state.update(completed=state['completed']+1, elapsed_seconds=perf_counter()-started)
            write_json(status_path, state)
            print(json.dumps(dict(worker=worker, source=row['source_id'], completed=state['completed'],
                                  total=len(sources), samples=state['samples'], elapsed_seconds=state['elapsed_seconds'])), flush=True)
        state.update(phase='completed', current=None, peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except Exception as exc:
        state.update(phase='needs_review', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['elapsed_seconds'] = perf_counter()-started
        write_json(status_path, state)


def finalize(output_dir):
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset

    output = Path(output_dir).resolve()
    index = json.loads((output/'index.json').read_text())
    workers = [json.loads((output/'workers'/f'{i}.json').read_text()) for i in range(4)]
    if any(s['phase'] != 'completed' for s in workers):
        raise ValueError('所有 worker 完成后才能生成训练入口')
    files = {'train':[], 'validation':[]}
    copied, reconstructed, labels = 0, 0, 0
    for row in index['sources']:
        record = json.loads((output/'labels'/row['source_id']/'metadata.json').read_text())
        for unit in record['units']:
            if file_sha256(Path(unit['progress_csv'])) != unit['progress_sha256']:
                raise ValueError('PT 文件摘要不符')
            if unit.get('copied_legacy_csv'):
                if file_sha256(Path(unit['copied_legacy_csv'])) != unit['progress_sha256']:
                    raise ValueError('旧 PT 未保持原样')
                copied += 1
            elif row['pt_method']=='legacy-rollback-unchanged':
                reconstructed += 1
            else:
                labels += 1
            files[row['split']].append(str(output/'cache'/row['split']/unit['unit_id']))
    samples = {}
    for split, paths in files.items():
        dataset = M3Dataset(paths)
        cursor = 0
        for directory, summary in dataset.episodes:
            source = summary['source']
            with Path(source['progress_csv']).open() as f:
                target = {int(r['frame_index']):float(r['pt']) for r in csv.DictReader(f)}
            previous = None
            for line in (directory/'samples.jsonl').read_text().splitlines():
                r = json.loads(line)
                if r['pt'] != target[r['frame_index']]:
                    raise ValueError('采样 PT 与逐帧标签不一致')
                if (r['history'] is None) != (previous is None):
                    raise ValueError('首帧/历史引用错误')
                if r['history'] is not None and not (directory/r['history']).is_file():
                    raise FileNotFoundError(directory/r['history'])
                for image in r['images'].values():
                    if image is not None and not (directory/image).is_file():
                        raise FileNotFoundError(directory/image)
                previous = r['frame_index']
            # 每个单轮的首/末样本实际加载：检查重置、时间戳、预算及数值。
            first = dataset[cursor]
            last = dataset[cursor+summary['num_samples']-1]
            if first['history_tokens'] is not None:
                raise ValueError('新单轮未重置历史')
            history = last['history_tokens']
            if history is not None and (len(history)>dataset.history_capacity or
                                        history.shape[1]!=dataset.history_dim or not torch.isfinite(history).all()):
                raise ValueError('历史快照维度、预算或数值无效')
            cursor += summary['num_samples']
        samples[split] = len(dataset)
        write_json(output/f'{split}.sample_files.json', paths)
    report = dict(status='completed', source_videos=index['source_video_count'], source_splits=index['splits'],
                  units={k:len(v) for k,v in files.items()}, samples=samples,
                  ordinary_whole_video_pt=labels, legacy_pt_csv_copied=copied, legacy_pt_reconstructed=reconstructed,
                  all_pt_hashes_verified=True, view_policy=VIEW_POLICY, exclusions=index['exclusions'], workers=workers,
                  training_started=False, completed_at=datetime.now(timezone.utc).isoformat())
    write_json(output/'report.json', report)
    # 留下可审阅的训练配置，但训练轮数尚未确认，保持启动门禁关闭。
    setup = Path('/cache/cvpzhr/setup')
    config = json.loads((setup/'train-4gpu.json').read_text())
    config.update(sample_files=files['train'],output_dir='/cache/cvpzhr/outputs/forest-proprimem-4gpu',
                  history_source=dataset.history_source,history_capacity=dataset.history_capacity)
    config['model']['history_dim'] = dataset.history_dim
    config_path = setup/'proprimem-5fps-20261030/train-4gpu.template.json'
    write_json(config_path,config)
    readiness = json.loads((setup/'readiness.json').read_text())
    readiness.update(status='DATASET_READY_TRAINING_SCHEDULE_PENDING',training_data_ready=True,
                     training_configuration_valid=True,training_schedule_confirmed=False,
                     train_samples=samples['train'],validation_samples=samples['validation'],
                     training_started=False,configuration=str(config_path),
                     prepared_dataset=str(output),prepared_code=str(Path(__file__).parents[2]),
                     reason='新数据与缓存已完成；训练轮数/存档安排待确认，未启动训练。')
    write_json(setup/'readiness.json',readiness)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index')
    index.add_argument('--dataset-root', required=True)
    index.add_argument('--output-dir', required=True)
    index.add_argument('--legacy-dir', required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('--output-dir', required=True)
    worker.add_argument('--model-path', required=True)
    worker.add_argument('--worker', type=int, required=True)
    worker.add_argument('--batch-size', type=int, default=16)
    worker.add_argument('--memory-config')
    from mem_lite.memory.splitting import add_split_arguments
    add_split_arguments(worker)
    final = sub.add_parser('finalize')
    final.add_argument('--output-dir', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = {'index':prepare_index, 'worker':build_worker, 'finalize':finalize}[command](**args)
    if result:
        print(json.dumps({k:v for k,v in result.items() if k!='sources'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
