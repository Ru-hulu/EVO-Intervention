#!/usr/bin/env python3
"""Validate all frame outputs and render the batched experiment."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    root = args.run
    s = json.loads((root / 'summary.json').read_text())
    m = json.loads((root / 'manifest.json').read_text())
    c = json.loads((root / 'baseline_comparison.json').read_text())
    rows = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
    assert s['status'] == m['status'] == 'completed'
    assert [r['frame_index'] for r in rows] == list(range(690))
    assert all(abs(r['timestamp_s'] - i / 30) < 1e-9 for i, r in enumerate(rows))
    assert all(r['input_tokens'] == 2865 and r['image_sizes'] == [[644, 476]] * 6 for r in rows)
    assert m['query_history'] is False and m['original_prompt_verified_identical']
    with (root / 'predictions.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 690
    assert all((v['progress_percent'] == '' if r['progress_percent'] is None else float(v['progress_percent']) == r['progress_percent']) for r, v in zip(rows, csv_rows))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    x = np.array([r['timestamp_s'] for r in rows])
    y = np.array([r['progress_percent'] if r['progress_percent'] is not None else np.nan for r in rows])
    diff = np.diff(y)
    diff = diff[np.isfinite(diff)]
    analysis = {'min_progress_percent': float(np.nanmin(y)), 'max_progress_percent': float(np.nanmax(y)),
                'downward_transitions': int((diff < 0).sum()),
                'largest_downward_jump_pp': max(0., float(-diff.min())),
                'cached_token_counts': sorted({r['num_cached_tokens'] for r in rows if r['num_cached_tokens'] is not None}),
                'invalid_frame_indices': [r['frame_index'] for r in rows if r['status'] != 'ok'],
                'unannotated_query': True, 'result_type': 'model prediction, not ground truth'}
    (root / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    fig, ax = plt.subplots(figsize=(11.5, 3.9), layout='constrained')
    ax.plot(x, y, color='#136a78', linewidth=1.25, label='ProgressLM visual demonstration (raw)')
    if np.isfinite(y[0]):
        ax.scatter([x[0]], [y[0]], color='#136a78', s=28, zorder=3)
    if np.isfinite(y[-1]):
        ax.scatter([x[-1]], [y[-1]], color='#136a78', s=28, zorder=3)
    if np.isnan(y).any():
        ax.scatter(x[np.isnan(y)], np.full(np.isnan(y).sum(), -3), marker='x', s=16, color='#b5523e', label='n/a or invalid')
    ax.set(xlim=(0, 23), ylim=(-5, 102), xlabel='First-round video time (s)', ylabel='Predicted progress (%)')
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(alpha=.17)
    ax.legend(loc='upper left', frameon=False)
    fig.savefig(root / 'progress.png', dpi=220)
    fig.savefig(root / 'progress.svg')
    plt.close(fig)
    svg = (root / 'progress.svg').read_text()
    svg = svg[svg.index('<svg'):]
    pic = base64.b64encode((root / 'reference_top.jpg').read_bytes()).decode()
    states = ['尚未抓取本轮零件', '已拿起套筒', '套筒、螺栓均已拿起', '插入完成，尚未入盒', '装配件入盒并释放']
    table = ''.join(f'<tr><td>{ref["source_time_seconds"]:g} s</td><td>{ref["progress"]}%</td><td>{state}</td></tr>' for ref,state in zip(m['reference_frames'],states))
    def pct(value):
        return '无有效值' if value is None else f'{value:g}%'
    baseline_diff = c['mean_absolute_difference_pp']
    html_text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ProgressLM · 23 秒插螺栓进度预测</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f8;color:#202d38;font:15px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1100px;margin:auto;padding:18px 22px}}h1{{font-size:23px;line-height:1.3;margin:0 0 7px}}p{{margin:6px 0}}a{{color:#136a78}}.muted{{font-size:13px;color:#5b6873}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}}.card{{background:white;border:1px solid #dae2e7;border-radius:8px;padding:10px 14px}}.card b{{display:block;color:#136a78;font-size:25px}}.card span{{font-size:13px;color:#5b6873}}.plot{{background:white;border:1px solid #dae2e7;border-radius:8px;padding:7px;margin:9px 0}}svg{{display:block;width:100%;height:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;padding:4px 7px;border-bottom:1px solid #dae2e7}}th{{background:#eaf0f3}}img{{width:100%;height:auto;display:block;margin:9px 0}}.note{{border-left:3px solid #b48443;padding-left:10px}}summary{{cursor:pointer}}pre{{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:10px}}@media(max-width:650px){{main{{padding:13px}}h1{{font-size:20px}}.cards{{gap:6px}}.card{{padding:8px}}.card b{{font-size:20px}}.grid{{grid-template-columns:1fr}}}}
</style><main><h1>ProgressLM：第二轮视觉示范 → 第一轮 23 秒进度</h1>
<p><b>已完成全部 690 帧。</b>第二轮的 5 张关键帧作为固定专家示范；采用 vLLM 批处理，保留完整推理文本和原始进度值。</p>
<div class="cards"><div class="card"><b>{s['inference_wall_s']:.1f} 秒</b><span>整段预测时间（含批输入准备）</span></div><div class="card"><b>{s['throughput_fps']:.2f} 帧/秒</b><span>离线批处理吞吐</span></div><div class="card"><b>{s['valid_frames']} / 690</b><span>有效进度输出</span></div></div>
<p>首帧 <b>{pct(y[0])}</b>，末帧 <b>{pct(y[-1])}</b>；相邻帧下降 {analysis['downward_transitions']} 次。<b>末帧实际已完成入盒，但模型仍匹配到第 2 张参考图附近，完成判断出现明显错误。</b>未平滑、未强制单调、未固定首尾值；无效输出显示为断点。</p>
<div class="plot">{svg}</div>
<div class="grid"><div><b>参考轨迹抽帧与锚点</b><table><tr><th>原长视频时间</th><th>进度</th><th>人工核对状态</th></tr>{table}</table></div><div><b>每个样本的实际输入</b><p>英文任务描述 + <b>5 张参考顶视图及标签 + 1 张当前顶视图</b>。第一轮帧 0–689，时间 0–22.9667 秒；不加入历史帧。原图 640×480，处理后 644×476，共 2865 输入 tokens。</p><p class="muted">任务终点：铜螺栓插入黑色套筒，并将装配件放入盒中。参考起点采用已确认的 24 秒，避开与第一轮末帧相同的 22.967 秒画面。</p></div></div>
<img src="data:image/jpeg;base64,{pic}" alt="0%、25%、50%、75%、100%五张专家参考图">
<p><b>加速方式：</b>每批提交 64 个独立样本，最多 32 个并发序列；缓存固定参考图及可复用的前缀。模型仍为 ProgressLM-3B-RL，BF16，贪心解码，输出上限 2048 tokens，未删减推理文本。平均输出 {s['mean_generated_tokens']:.1f} tokens，截断 {s['token_cap_count']} 帧。模型初始化另需 {s['model_initialization_s']:.1f} 秒。</p>
<p class="muted">GPU：{html.escape(m['gpu'])}。平均摊销 {s['amortized_ms_per_frame_not_latency']:.1f} ms/帧是吞吐指标，<b>不是单帧请求延迟</b>。此前串行完成 {c['frames_in_serial_baseline']} 帧，平均 {c['serial_mean_latency_ms']/1000:.3f} 秒/帧；这是第一轮前段的部分测量。最终批处理统计不用于宣称在线低延迟。</p>
<p class="note"><b>后端核对：</b>提示文本与串行版逐字一致。共同有效的 {c['jointly_valid_frames']} 帧中，{c['exact_score_matches']} 帧进度完全一致，平均绝对差 {baseline_diff:.2f} 个百分点，最大差 {c['max_absolute_difference_pp']:g} 个百分点；有效/无效状态差异 {c['status_disagreements']} 帧。更换推理后端与批处理可能通过浮点差异改变贪心输出，未人为对齐结果。</p>
<p class="muted">锚点是人工定义的等权任务阶段，不是耗时比例。第一轮没有独立进度真值，因此不能据此报告准确率。两轮来自同一长视频，盒中已有装配件数量不同。</p>
<details><summary>末帧原始回答</summary><pre>{html.escape(rows[-1]['raw_response'])}</pre></details>
<p class="muted"><a href="data/predictions.csv">逐帧 CSV</a> · <a href="data/predictions.jsonl">完整原始回答</a> · <a href="data/manifest.json">运行配置</a> · <a href="data/baseline_comparison.json">串行对照</a> · <a href="data/progress.png">曲线 PNG</a> · <a href="data/progress.svg">曲线 SVG</a> · <a href="results.zip">完整结果包</a></p>
</main></html>'''
    (root / 'report.html').write_text(html_text)
    shutil.copy2(__file__,root/'build_batch_results.py')
    print(json.dumps({'verified':len(rows),'summary':s,'comparison':c,'analysis':analysis},ensure_ascii=False))


if __name__ == '__main__':
    main()
