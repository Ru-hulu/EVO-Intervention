"""Overlay the same original validation episodes from all three epoch-3 models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

MODELS = {'streamforest': ('StreamForest / Qwen3.5-4B', '#1764C0'),
          'robobrain': ('RoboBrain2.5-4B', '#D97706'),
          'robometer': ('Robometer-4B', '#16836A')}


def load_aligned(sources, selection):
    data = {}
    for name, path in sources.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        data[name] = {(r['episode_id'], r['frame_index']): r for r in rows}
        if len(data[name]) != len(rows):
            raise ValueError(f'Duplicate prediction keys: {name}')
    keys = data['streamforest'].keys()
    if not all(rows.keys() == keys for rows in data.values()):
        raise ValueError('Validation frame keys differ between models')
    for key in keys:
        expected = data['streamforest'][key]['target_pt']
        if not all(abs(rows[key]['target_pt'] - expected) < 1e-6 for rows in data.values()):
            raise ValueError(f'Target mismatch: {key}')
    items, paired = [], []
    for episode in selection:
        selected = sorted(k for k in keys if k[0] == episode['episode_id'])
        if len(selected) != episode['samples']:
            raise ValueError('Episode selection differs from the previous plots')
        target = np.array([data['streamforest'][k]['target_pt'] for k in selected]) * 100
        item = {'title': episode['title'], 'episode_id': episode['episode_id'],
                'frames': np.array([k[1] for k in selected]), 'target': target, 'metrics': {}}
        for name in MODELS:
            prediction = np.array([data[name][k]['predicted_pt'] for k in selected], dtype=float) * 100
            if not np.isfinite(prediction).all():
                raise ValueError(f'Invalid predictions: {name}/{episode["episode_id"]}')
            item[name] = prediction
            error = prediction - target
            item['metrics'][name] = {'mae_pp': float(abs(error).mean()), 'rmse_pp': float(np.sqrt((error**2).mean()))}
        if abs(item['metrics']['streamforest']['mae_pp'] - episode['mae_pp']) > 1e-7:
            raise ValueError('StreamForest predictions differ from the original figure')
        items.append(item)
        for i, key in enumerate(selected):
            paired.append({'episode_id': key[0], 'frame_index': key[1], 'target_pct': target[i],
                           **{name + '_pct': item[name][i] for name in MODELS}})
    return items, paired, len(keys)


def draw(ax, item):
    ax.plot(item['frames'], item['target'], color='#475569', lw=1.8,
            linestyle=(0, (5, 3)), label='真实 PT 标签', zorder=5)
    for name, (label, color) in MODELS.items():
        ax.plot(item['frames'], item[name], color=color, lw=1.6, marker='o', markersize=1.8,
                alpha=0.95, label=label)
    ax.set_title(item['title'], loc='left', fontsize=17, y=1.29, pad=0)
    ax.text(0, 1.225, item['episode_id'].split('/')[-1], transform=ax.transAxes, color='#64748B', fontsize=10)
    for i, (name, (_, color)) in enumerate(MODELS.items()):
        metric = item['metrics'][name]
        ax.text(0, 1.16 - i * 0.065,
                f"{name.title()}：MAE {metric['mae_pp']:.2f} / RMSE {metric['rmse_pp']:.2f} 个百分点",
                transform=ax.transAxes, color=color, fontsize=10)
    ax.set(xlim=(0, item['frames'][-1]), ylim=(-3, 103), yticks=[0, 25, 50, 75, 100],
           xlabel='原始视频帧号', ylabel='任务进度（%）')
    ax.grid(color='#CBD5E1', alpha=0.4, lw=0.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODELS:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--font', default='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    args = parser.parse_args()
    sources = {name: getattr(args, name) for name in MODELS}
    selection = json.loads(args.selection.read_text())
    items, paired, total = load_aligned(sources, selection)
    if len(items) != 8:
        raise ValueError('Expected the same eight tasks as the previous figure')
    font_manager.fontManager.addfont(args.font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=args.font).get_name(),
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.86, bottom=0.065, wspace=0.22, hspace=0.8)
    fig.text(0.065, 0.98, '每个任务一条验证视频 · 三模型进度预测对比', fontsize=25, va='top')
    fig.text(0.065, 0.957, '沿用此前 8 条原始验证视频 · 第 3 轮检查点 · 5 FPS · 未平滑 · 未插入重复帧',
             color='#64748B', fontsize=12, va='top')
    for ax, item in zip(axes.flat, items, strict=True):
        draw(ax, item)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper left',
               bbox_to_anchor=(0.06, 0.941), ncol=4, frameon=False, fontsize=11, handlelength=3)
    fig.text(0.065, 0.032, '输入条件：StreamForest 为三视角当前观测与历史记忆；Robometer 为主视角最近 8 帧。',
             fontsize=10.5, color='#64748B')
    fig.text(0.065, 0.018, 'RoboBrain 额外提供同视频结束目标帧，属于已知目标条件；三者并非完全相同的输入设定。',
             fontsize=10.5, color='#64748B')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.output_dir / f'八个任务-三模型进度对比.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    summary = {'validation_samples_aligned': total, 'selected_samples': len(paired), 'sample_fps': 5,
               'epoch': 3, 'smoothing': False, 'repeated_frames_inserted': False,
               'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
               'episodes': [{k: item[k] for k in ('title', 'episode_id', 'metrics')} for item in items]}
    (args.output_dir / 'comparison-metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (args.output_dir / 'aligned-predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
