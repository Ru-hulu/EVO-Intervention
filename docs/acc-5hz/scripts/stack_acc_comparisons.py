"""Export two vertically stacked ACC comparisons, enforcing one scale across groups."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time


def probe(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
    return next(stream for stream in data['streams'] if stream['codec_type'] == 'video')


def run(root):
    groups = [('成功组', 'hf_success', [10, 13]), ('失败组', 'local_failure', [2, 3])]
    limits = json.loads((root / 'axis_limits.json').read_text())
    out = root / 'stacked_5hz'
    out.mkdir(exist_ok=True)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    prepared = []
    # Validate all four before exporting either group: within-pair equality alone
    # would not establish a shared success/failure Y axis.
    for label, source, indices in groups:
        prefix = '成功组示范' if source == 'hf_success' else '失败组'
        paths = [root / 'videos' / f'{prefix}_{index:06d}_ACC.mp4' for index in indices]
        summaries = []
        for path, index in zip(paths, indices):
            metadata = json.loads(path.with_suffix('.json').read_text())
            summary = json.loads((root / 'reports' / source / f'episode_{index:06d}' / 'summary.json').read_text())
            if metadata['axis_limits'] != limits:
                raise ValueError('All four sources must share identical Y-axis limits')
            if (summary['query_interval_s'], summary['ema_alpha'], summary['action_horizon'], summary['executed_steps']) != (.2, .9, 50, 6):
                raise ValueError('Expected the new 5 Hz protocol')
            summaries.append(summary)
        prepared.append((label, indices, paths, summaries))

    def render(group):
        label, indices, paths, summaries = group
        streams = [probe(path) for path in paths]
        if any((s['width'], s['height'], Fraction(s['avg_frame_rate'])) != (1920, 1080, 30) for s in streams):
            raise ValueError('Expected 1920x1080 at 30 FPS')
        counts = [int(s['nb_frames']) for s in streams]
        frames = max(counts)
        output = out / f'ACC_{label}_5Hz_上下对比_统一Y轴.mp4'
        if output.exists():
            raise FileExistsError(output)
        filters = []
        for i, (index, summary) in enumerate(zip(indices, summaries)):
            location = '上' if i == 0 else '下'
            title = out / f'{label}_{i}_title.txt'
            title.write_text(f'{location}：{label} #{index:06d} · 5 Hz 重算 · 本次 P95={summary["acc_p95"]:.3f} · 成功/失败共用 Y 轴', encoding='utf-8')
            color = '0x54d5ae' if i == 0 else '0xffb76b'
            chain = f'[{i}:v]setpts=PTS-STARTPTS,setsar=1,tpad=stop_mode=clone:stop={frames-counts[i]},drawbox=x=0:y=0:w=1920:h=62:color=0x101827:t=fill,drawtext=fontfile={font}:textfile={title}:fontsize=32:fontcolor={color}:x=24:y=15'
            if frames > counts[i]:
                ended = out / f'{label}_{i}_ended.txt'
                ended.write_text('本条已结束 · 画面与数值保持最后状态', encoding='utf-8')
                chain += f",drawtext=fontfile={font}:textfile={ended}:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=10:x=140:y=622:enable='gte(t,{counts[i]/30:.8f})'"
            filters.append(chain + f'[p{i}]')
        filters.append('[p0][p1]vstack=inputs=2[out]')
        graph = out / f'{label}_filter.txt'
        graph.write_text(';\n'.join(filters), encoding='utf-8')
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', '-filter_complex_threads', '2']
        for path in paths:
            command += ['-threads', '2', '-i', str(path)]
        command += ['-filter_complex_script', str(graph), '-map', '[out]', '-an', '-frames:v', str(frames), '-r', '30',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        started = time.monotonic()
        with output.with_suffix('.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        stream = probe(output)
        if (int(stream['nb_frames']), stream['width'], stream['height'], Fraction(stream['avg_frame_rate'])) != (frames, 1920, 2160, 30):
            raise ValueError('Export frame count/resolution/FPS mismatch')
        if abs(float(stream['duration']) - frames / 30) > 1/30:
            raise ValueError('Export duration mismatch')
        preview = output.with_suffix('.png')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', '15', '-i', str(output),
                        '-frames:v', '1', '-vf', 'scale=960:1080', str(preview)], check=True)
        result = {'group': label, 'top_episode': indices[0], 'bottom_episode': indices[1], 'inputs': [str(p) for p in paths],
                  'output': str(output), 'frames': frames, 'fps': 30, 'duration_s': frames/30, 'width': 1920, 'height': 2160,
                  'source_frames': counts, 'axis_limits': limits, 'axis_lower_bound': 0, 'score_hz': 5, 'ema_alpha': .9,
                  'shorter_clip': 'Freeze final frame and scores with a visible end label; no speed change',
                  'render_seconds': time.monotonic()-started, 'ffprobe': stream}
        output.with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print('EXPORTED', output.name, frames, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(render, prepared))
    (out / 'COMPLETE.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
