#!/usr/bin/env python3
"""Validate a completed run and build compact, self-contained HTML plus scientific plots."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
    assert summary['status'] == manifest['status'] == 'completed'
    assert [r['frame_index'] for r in rows] == list(range(690))
    assert all(abs(r['timestamp_s'] - i / 30) < 1e-9 for i, r in enumerate(rows))
    assert all(r['input_tokens'] > 0 and len(r['image_sizes']) == 6 for r in rows)
    assert manifest['batch_size'] == 1 and manifest['query_history'] is False
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    x = np.array([r['timestamp_s'] for r in rows])
    y = np.array([r['progress_percent'] if r['progress_percent'] is not None else np.nan for r in rows])
    latency = np.array([r['latency_ms'] / 1000 for r in rows])
    assert abs(latency.mean() * 1000 - summary['mean_latency_ms']) < 1e-6
    diffs = np.diff(y)
    finite_diffs = diffs[np.isfinite(diffs)]
    overview = {
        'progress_min_percent': float(np.nanmin(y)) if np.isfinite(y).any() else None,
        'progress_max_percent': float(np.nanmax(y)) if np.isfinite(y).any() else None,
        'downward_transitions': int((finite_diffs < 0).sum()),
        'upward_transitions': int((finite_diffs > 0).sum()),
        'largest_downward_jump_pp': max(0, float(-finite_diffs.min())) if len(finite_diffs) else None,
        'actual_image_sizes': sorted({tuple(size) for r in rows for size in r['image_sizes']}),
        'input_tokens': sorted({r['input_tokens'] for r in rows}),
        'meets_30fps_sequential': bool(latency.mean() <= 1 / 30),
        'realtime_slowdown_vs_30fps': float(latency.mean() * 30),
        'no_ground_truth': 'Anchors are reference labels; query trajectory has no independently annotated progress GT.',
        'structural_format_note': 'format_valid checks existence of four output tags, not numeric-only ref formatting.',
    }
    (root / 'analysis.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(11.5, 3.65), layout='constrained')
    ax.plot(x, y, color='#126b78', linewidth=1.3, label='Raw model prediction')
    if np.isfinite(y[0]):
        ax.scatter([x[0]], [y[0]], color='#126b78', s=28, zorder=3)
    if np.isfinite(y[-1]):
        ax.scatter([x[-1]], [y[-1]], color='#126b78', s=28, zorder=3)
    ax.set(xlim=(0, 23), ylim=(-2, 102), xlabel='First-round video time (s)', ylabel='Predicted progress (%)')
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(alpha=.17)
    ax.legend(loc='upper left', frameon=False)
    fig.savefig(root / 'progress.png', dpi=220)
    fig.savefig(root / 'progress.svg')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 2.6), layout='constrained')
    ax.plot(x, latency, color='#7064a8', linewidth=1)
    ax.axhline(latency.mean(), color='#413675', linewidth=1.2, linestyle='--',
               label=f'Mean {latency.mean():.3f} s')
    ax.axhline(1 / 30, color='#a26022', linewidth=1, linestyle=':', label='30 FPS budget: 0.033 s')
    ax.set(xlim=(0, 23), ylim=(0, max(latency) * 1.07), xlabel='First-round video time (s)', ylabel='Request latency (s)')
    ax.set_xticks(np.arange(0, 24, 2))
    ax.grid(alpha=.17)
    ax.legend(loc='best', frameon=False, ncol=2)
    fig.savefig(root / 'latency.png', dpi=220)
    fig.savefig(root / 'latency.svg')
    plt.close(fig)

    def svg(name):
        text = (root / name).read_text()
        return text[text.index('<svg'):]

    def pct(value):
        return '无有效输出' if value is None else f'{value:g}%'

    states = ['尚未抓取本轮零件', '已拿起套筒', '套筒与螺栓均已拿起', '插入完成，尚未入盒', '装配件放入盒中并释放']
    table = ''.join(f'<tr><td>{ref["source_time_seconds"]:g} s / {ref["source_frame_index"]}</td>'
                    f'<td>{ref["progress"]}%</td><td>{state}</td></tr>'
                    for ref, state in zip(manifest['reference_frames'], states))
    reference_b64 = base64.b64encode((root / 'reference_top.jpg').read_bytes()).decode()
    samples = [rows[i] for i in [0, 150, 300, 450, 600, 689]]
    sample_rows = ''.join(f'<tr><td>{r["timestamp_s"]:.2f} s</td><td>{pct(r["progress_percent"])}</td>'
                          f'<td>{r["latency_ms"] / 1000:.3f} s</td><td>{r["generated_tokens"]}</td></tr>' for r in samples)
    conclusion = ('可达到' if overview['meets_30fps_sequential'] else '无法达到') + ' 30 FPS 逐帧实时预测'
    raw_last = html.escape(rows[-1]['raw_response'])
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProgressLM · 第二轮视觉示范 → 第一轮进度</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f7;color:#1e2933;font:15px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1100px;margin:auto;padding:20px 24px}}h1{{font-size:23px;line-height:1.3;margin:0 0 7px}}p{{margin:6px 0}}a{{color:#126b78}}.muted{{color:#586572;font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}}.card{{background:white;border:1px solid #dce3e7;border-radius:8px;padding:10px 13px}}.card strong{{display:block;font-size:24px;line-height:1.3;color:#126b78}}.card span{{font-size:13px;color:#586572}}.figure{{background:white;border:1px solid #dce3e7;border-radius:8px;padding:7px 9px;margin:10px 0}}.figure svg{{display:block;width:100%;height:auto}}.label{{font-weight:650;margin:0 0 4px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:5px 7px;text-align:left;border-bottom:1px solid #dce3e7}}th{{background:#eaf0f2}}.refs{{width:100%;height:auto;display:block;margin:10px 0}}details{{margin:8px 0}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#fff;padding:10px}}.note{{border-left:3px solid #b5813b;padding-left:10px}}@media(max-width:650px){{main{{padding:14px}}h1{{font-size:20px}}.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}.card strong{{font-size:22px}}}}
</style><main>
<h1>ProgressLM：第二轮视觉示范 → 第一轮 23 秒进度</h1>
<p><b>{conclusion}。</b>本机 batch=1，完整推理输出平均 {latency.mean():.3f} 秒/帧，约 {summary['effective_fps']:.3f} FPS；690 帧纯请求累计 {summary['summed_inference_s']/60:.2f} 分钟。</p>
<div class="cards"><div class="card"><strong>{latency.mean():.3f} s</strong><span>平均单帧耗时</span></div><div class="card"><strong>{summary['p95_latency_ms']/1000:.3f} s</strong><span>P95 · P50 {summary['p50_latency_ms']/1000:.3f} s</span></div><div class="card"><strong>{summary['effective_fps']:.3f} FPS</strong><span>串行处理能力</span></div><div class="card"><strong>{summary['valid_frames']} / 690</strong><span>有效进度输出</span></div></div>
<p>首帧 <b>{pct(rows[0]['progress_percent'])}</b>，末帧 <b>{pct(rows[-1]['progress_percent'])}</b>。相邻帧下降 {overview['downward_transitions']} 次；曲线保留全部原始预测，未平滑、未强制单调、未固定首尾值。</p>
<div class="figure"><div class="label">逐帧进度</div>{svg('progress.svg')}</div>
<div class="grid"><div><div class="label">已确认的参考锚点</div><table><tr><th>原长视频时间 / 帧号</th><th>锚点</th><th>人工核对的状态</th></tr>{table}</table></div>
<div><div class="label">实际输入与输出</div><p>每次：英文任务描述 + <b>5 张固定参考顶视图</b>及 0/25/50/75/100% 标签 + <b>1 张当前顶视图</b>。不输入第一轮历史；参考编码不跨请求缓存。</p>
<p class="muted">任务：插入一个铜螺栓到黑色套筒，再将装配件放入盒中。查询为第一轮帧 0–689（0–22.9667 秒）。参考起点取 24 秒，避开第一轮末帧与第二轮原起点重合的问题。</p>
<p class="muted">输出保留 ref_think、ref、score_think、score。n/a、解析失败或截断保留为无效值，不补成进度。无效 {690-summary['valid_frames']} 帧，触及 token 上限 {summary['token_cap_count']} 帧。</p></div></div>
<img class="refs" alt="五张参考顶视图，按0%、25%、50%、75%、100%排列" src="data:image/jpeg;base64,{reference_b64}">
<div class="figure"><div class="label">每帧请求耗时</div>{svg('latency.svg')}</div>
<p>平均预处理 {summary['mean_preprocess_ms']/1000:.3f} 秒，模型生成 {summary['mean_generation_ms']/1000:.3f} 秒，解码与解析 {summary['mean_decode_parse_ms']/1000:.4f} 秒。平均输出 {summary['mean_generated_tokens']:.1f} tokens；模型初始化 {summary['model_initialization_s']:.2f} 秒、3 次预热 {summary['warmup_s']:.2f} 秒另计。</p>
<p class="muted">计时包含读图、预处理、全部图像编码与完整推理文本生成，CUDA 同步计时；不含离线视频解码、结果写盘及网络。GPU：{html.escape(manifest['gpu'])}。BF16 / SDPA；贪心解码，最大 2048 tokens；图像原始 640×480、实际 {overview['actual_image_sizes'][0][0]}×{overview['actual_image_sizes'][0][1]}，每次输入 {overview['input_tokens']} tokens。此为已确认的部署配置，像素与解码参数不同于官方 benchmark 启动脚本，结果不能冒充其官方速度。</p>
<p class="note"><b>解释范围：</b>五个进度锚点由人工按任务阶段等权定义，不是时间比例。第一轮没有独立进度真值，因此本次能展示模型输出与实时性，不能据此给出预测准确率。两轮来自同一长视频，且盒中已有装配件数量不同。</p>
<details><summary>抽查数值与末帧原始回答</summary><table><tr><th>查询时间</th><th>进度</th><th>耗时</th><th>输出 tokens</th></tr>{sample_rows}</table><pre>{raw_last}</pre></details>
<p class="muted">模型：<a href="https://huggingface.co/Raymond-Qiancx/ProgressLM-3B-RL">ProgressLM-3B-RL</a> · <a href="https://github.com/ProgressLM/ProgressLM/blob/main/eval/qwen25vl/codes/prompts/visual_demo_prompt.py">官方视觉示范 prompt</a> · <a href="data/predictions.csv">逐帧 CSV</a> · <a href="data/predictions.jsonl">原始回答 JSONL</a> · <a href="data/manifest.json">运行配置</a> · <a href="results.zip">完整结果包</a></p>
</main></html>'''
    (root / 'report.html').write_text(report)
    shutil.copy2(__file__, root / 'build_results.py')
    print(json.dumps({'validated_frames': len(rows), 'analysis': overview, 'report': str(root / 'report.html')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
