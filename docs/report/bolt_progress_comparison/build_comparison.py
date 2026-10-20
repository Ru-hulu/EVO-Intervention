#!/usr/bin/env python3
"""Collect existing first-round predictions and draw one combined plot. No inference."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path('/root/autodl-tmp/EvalProgress/outputs/piperx_insert_copper_screw_ep000000_0_23s')
TASK = 'Insert one copper screw into a black sleeve, then place the assembled component into the box.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        dict(id='A', label='Robometer', legend='Robometer',
             run='Robometer/single_insertions_20261015_152850', file='episode_000000.jsonl', count=69,
             input='从第 0 帧到当前帧，均匀抽取 4 张顶视图。',
             sampling='69 点 · 3 FPS', semantics='absolute_progress', color='#202c3b', style='-', marker='o'),
        dict(id='B', label='ProgressLM · 文字示范', legend='ProgressLM / text demo',
             run='ProgressLM-3B-RL/single_insertions_20261015_152851', file='episode_000000.jsonl', count=23,
             input='当前顶视图＋取套筒／取螺栓／插入／入盒四步文字示范（各占 25%）。',
             sampling='23 点 · 1 FPS', semantics='absolute_progress', color='#a84a9a', style='--', marker='s'),
        dict(id='C', label='ProgressLM · 视觉示范', legend='ProgressLM / round-2 visual demo',
             run='ProgressLM-3B-RL/visual_round2_batch_20261015_125505', file='predictions.jsonl', count=690,
             input='当前顶视图＋第二轮 24/29/52/56/58 秒顶视图，标为 0/25/50/75/100%。',
             sampling='690 点 · 30 FPS', semantics='absolute_progress', color='#5c59cb', style='-', marker=None),
        dict(id='D', label='RoboBrain 2.5 · 空白目标', legend='RoboBrain 2.5 / blank goal',
             run='RoboBrain2.5/single_insertions_3fps_20261015_213840_blank', file='episode_000000.jsonl', count=69,
             input='本段首帧顶视图＋空白目标图＋第 0 帧和当前帧三视图，共 8 图。',
             sampling='69 点 · 3 FPS', semantics='signed_start_to_current_grm', color='#dd8a00', style='--', marker='^'),
        dict(id='E', label='RoboBrain 2.5 · 本段首尾', legend='RoboBrain 2.5 / own endpoints',
             run='RoboBrain2.5/single_insertions_3fps_20261015_213840_lastframe', file='episode_000000.jsonl', count=69,
             input='本段首尾顶视参考图＋第 0 帧和当前帧三视图，共 8 图。',
             sampling='69 点 · 3 FPS', semantics='signed_start_to_current_grm', color='#c54832', style='-', marker='^'),
        dict(id='F', label='Dopamine Preview · 本段首尾', legend='Dopamine Preview / own endpoints',
             run='Robo-Dopamine/preview_static8_rerun_20261015_latency', file='own_endpoints/predictions.jsonl', count=690,
             input='本段首尾顶视参考图＋第 0 帧和当前帧三视图，共 8 图。',
             sampling='690 点 · 30 FPS', semantics='signed_start_to_current_grm', color='#00886b', style='-', marker=None),
        dict(id='G', label='Dopamine Preview · 第二轮首尾', legend='Dopamine Preview / round-2 endpoints',
             run='Robo-Dopamine/preview_static8_rerun_20261015_latency', file='round2_endpoints/predictions.jsonl', count=690,
             input='第二轮首尾顶视参考图（22.967/58 秒）＋第 0 帧和当前帧三视图，共 8 图。',
             sampling='690 点 · 30 FPS', semantics='signed_start_to_current_grm', color='#238fc0', style='--', marker=None),
    ]
    records = []
    provenance = []
    curves = {}
    for item in sources:
        run = ROOT / item['run']
        source_file = run / item['file']
        manifest_file = run / 'manifest.json'
        manifest = json.loads(manifest_file.read_text())
        actual_task = manifest.get('instruction_override') or manifest.get('task') or manifest.get('inputs', {}).get('task')
        assert actual_task == TASK, (item['id'], actual_task)
        rows = [json.loads(line) for line in source_file.read_text().splitlines()]
        assert len(rows) == item['count'], item['id']
        indices = [r['frame_index'] for r in rows]
        assert indices == sorted(set(indices)) and min(indices) >= 0 and max(indices) <= 689
        expected = {'A':list(range(0,690,10)), 'B':list(range(0,690,30)),
                    'C':list(range(690)), 'D':list(range(10,690,10))+[689],
                    'E':list(range(10,690,10))+[689], 'F':list(range(690)), 'G':list(range(690))}[item['id']]
        assert indices == expected
        local_source = out / 'data' / 'sources' / item['id']
        local_source.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, local_source / 'predictions.jsonl')
        shutil.copy2(manifest_file, local_source / 'manifest.json')
        for optional in ['summary.json', 'completion.json', 'confirmed_plan.json', 'rerun_robobrain_3fps.py', 'annotate_screw.py']:
            if (run / optional).exists(): shutil.copy2(run / optional, local_source / optional)
        points = []
        for row in rows:
            t = float(row['timestamp_s'])
            assert abs(t - row['frame_index'] / 30) < 1e-8
            valid = row.get('status', 'ok') == 'ok' and row.get('valid', True)
            value = row.get('progress_percent') if 'progress_percent' in row else (row['progress'] * 100 if row['progress'] is not None else None)
            if not valid: value = None
            assert value is None or math.isfinite(value)
            # Independently verify text-generated numeric output where present.
            raw = row.get('raw_response', row.get('raw_output'))
            if value is not None and raw:
                scores = re.findall(r'<score>\s*([+-]?[\d.]+)\s*%\s*</score>', raw, re.I)
                if scores: assert len(scores) == 1 and abs(float(scores[0]) - value) < 1e-6
            point = dict(method_id=item['id'], method=item['label'], frame_index=row['frame_index'],
                         timestamp_s=t, score_percent=value, valid=value is not None,
                         score_semantics=item['semantics'])
            points.append(point)
            records.append(point)
        curves[item['id']] = points
        provenance.append({**item, 'remote_prediction_file':str(source_file),
                           'prediction_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
                           'manifest_sha256':hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                           'first_time_s':points[0]['timestamp_s'], 'last_time_s':points[-1]['timestamp_s'],
                           'first_score_percent':points[0]['score_percent'], 'last_score_percent':points[-1]['score_percent'],
                           'valid_points':sum(p['valid'] for p in points)})
    assert len(records) == 2300
    with (out / 'data' / 'all_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    manifest = {'dataset':'piperx_insert_copper_screw_ep000000_0_23s','task':TASK,
                'models':4,'configurations':7,'points':len(records),'series':provenance,
                'processing':'Only scale0..1 to percent; retain native timestamps, signs, gaps; no smoothing, resampling, extrapolation or forced endpoints.',
                'excluded':'Incomplete serial ProgressLM baseline, invalid-decoder archives, superseded prompts, and full-video/five-assembly experiments.',
                'notes':['GRM score semantics differ from absolute progress.','Future completion images supplied to own-endpoint GRM configurations.','No independent query-progress ground truth.'],
                'new_model_inference':True,
                'rerun':'RoboBrain2.5 blank and own-endpoint conditions re-evaluated at 3 FPS (frame_interval=10); other five series retain existing results.'}
    (out/'data'/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig, ax=plt.subplots(figsize=(14,7.35))
    fig.subplots_adjust(left=.075,right=.985,bottom=.12,top=.77)
    for item in sources:
        pts=curves[item['id']]
        line, = ax.plot([p['timestamp_s'] for p in pts], [p['score_percent'] if p['valid'] else np.nan for p in pts],
                label=item['id']+'  '+item['legend'],color=item['color'],linestyle=item['style'],
                marker=item['marker'],markersize=3 if item['id']!='A' else 2.5,
                linewidth=1.35 if item['count']==690 else 1.7,alpha=.92)
        line.set_gid('curve-' + item['id'])
    ax.set(xlim=(0,23),ylim=(-105,105),xlabel='Time in first-round video (s)',ylabel='Prediction / signed GRM score (%)')
    ax.set_xticks(list(range(0,23,2))+[23]);ax.set_yticks([-100,-75,-50,-25,0,25,50,75,100])
    ax.axhline(0,color='#6b7280',linewidth=.8,zorder=0)
    ax.grid(alpha=.15)
    ax.axhspan(-105,0,color='#f3f5f7',zorder=-2)
    fig.suptitle('23 s bolt insertion: all existing predictions',x=.075,y=.98,ha='left',fontsize=17,fontweight='bold')
    legend = fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.068,.925),
               ncol=3,frameon=False,fontsize=10.5,handlelength=2.6,columnspacing=1.7,labelspacing=.55)
    for item, line, label in zip(sources, legend.get_lines(), legend.get_texts()):
        line.set_gid('legend-line-' + item['id'])
        label.set_gid('legend-label-' + item['id'])
    fig.text(.075,.025,'Native sample times; no smoothing or forced endpoints. Negative values are retained for signed GRM scores.',fontsize=10,color='#576271')
    fig.savefig(out/'comparison.png',dpi=220)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)
    svg=(out/'comparison.svg').read_text();svg=svg[svg.index('<svg'):]
    table=''.join(f'<tr><td><span class="dot" style="background:{p["color"]}"></span><b>{p["id"]}</b> {html.escape(p["label"])}</td><td>{html.escape(p["input"])}</td><td>{p["sampling"]}</td></tr>' for p in sources)
    controls=''.join(f'<label class="curve-option" style="--series-color:{p["color"]}"><input type="checkbox" data-curve="{p["id"]}" aria-controls="curve-{p["id"]}" checked><span><b>{p["id"]}</b> {html.escape(p["label"])}</span></label>' for p in sources)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>23 秒插螺栓 · 进度预测汇总</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f8;color:#24313a;font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1240px;margin:auto;padding:16px 22px}}h1{{font-size:22px;margin:0 0 5px}}p{{margin:5px 0}}.muted{{font-size:12px;color:#5d6974}}.plot{{background:white;border:1px solid #dce3e7;border-radius:7px;margin:10px 0 7px;padding:4px}}svg{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:9px}}th,td{{padding:7px 8px;text-align:left;border-bottom:1px solid #dbe2e6;vertical-align:top}}th{{background:#e9eff2}}td:first-child{{width:29%}}td:last-child{{width:15%;white-space:nowrap}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}a{{color:#096b82}}@media(max-width:680px){{main{{padding:12px}}h1{{font-size:19px}}.tablewrap{{overflow-x:auto}}table{{min-width:760px}}}}</style>
<main><h1>23 秒插螺栓：4 个模型，7 组预测结果</h1>
<p>任务：将铜螺栓插入黑色套筒，再将装配件放入盒中。RoboBrain 两组已按 3 FPS 重算。</p>
<style>.curve-controls{{margin-top:9px}}.control-actions{{display:flex;align-items:center;gap:9px;font-size:12px;margin-bottom:5px}}.control-actions button{{font:inherit;color:#096b82;background:#fff;border:1px solid #cbd6dd;border-radius:4px;padding:2px 8px;cursor:pointer}}.curve-options{{display:flex;flex-wrap:wrap;gap:4px 7px}}.curve-option{{display:inline-flex;align-items:center;gap:5px;font-size:12px;background:#fff;border:1px solid #dce3e7;border-radius:4px;padding:3px 7px;cursor:pointer}}.curve-option input{{margin:0;accent-color:var(--series-color);cursor:pointer}}.curve-option input:not(:checked)+span{{color:#8a939b}}.curve-option:focus-within{{outline:2px solid #096b82;outline-offset:1px}}.control-actions button:focus-visible{{outline:2px solid #096b82;outline-offset:1px}}</style>
<div class="curve-controls" role="group" aria-label="选择显示的曲线"><div class="control-actions"><b>勾选显示曲线</b><button type="button" data-select="all">全选</button><button type="button" data-select="none">全不选</button><output id="visible-count" aria-live="polite">显示 7 / 7</output></div><div class="curve-options">{controls}</div></div>
<div class="plot">{svg}</div>
<p class="muted">按原采样点连线，不补点、不平滑。Robometer / ProgressLM 输出绝对进度；RoboBrain / Dopamine 输出相对首帧的有符号 GRM 分数，统一以百分数显示，保留负值。</p>
<p><b>评测输入</b>：各组均含上述任务描述；三视图＝顶视＋左右腕视。</p>
<div class="tablewrap"><table><thead><tr><th>方法 / 设置</th><th>每次评测实际输入</th><th>预测点 / 采样率</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="muted">Dopamine 权重为 GRM-2.0-8B-Preview，采用静态 8 图接口，无历史面板。各曲线末次采样时间不同；“本段首尾”设置使用了未来完成画面，因此不据此作公平排名。</p>
<p class="muted"><a href="comparison.png">图片 PNG</a> · <a href="comparison.svg">矢量图 SVG</a> · <a href="data/all_predictions.csv">合并数据 CSV</a> · <a href="data/provenance.json">原始来源</a> · <a href="results.zip">全部结果包</a></p></main>
<script>
const toggles = [...document.querySelectorAll('input[data-curve]')];
function updateCurves() {{
  for (const toggle of toggles) {{
    const id = toggle.dataset.curve;
    document.getElementById('curve-' + id).style.display = toggle.checked ? '' : 'none';
    for (const prefix of ['legend-line-', 'legend-label-']) {{
      document.getElementById(prefix + id).style.opacity = toggle.checked ? '1' : '.25';
    }}
  }}
  document.getElementById('visible-count').textContent = '显示 ' + toggles.filter(t => t.checked).length + ' / ' + toggles.length;
}}
for (const toggle of toggles) toggle.addEventListener('change', updateCurves);
for (const button of document.querySelectorAll('[data-select]')) {{
  button.addEventListener('click', () => {{
    for (const toggle of toggles) toggle.checked = button.dataset.select === 'all';
    updateCurves();
  }});
}}
updateCurves();
</script></html>'''
    (out/'report.html').write_text(page)
    shutil.copy2(__file__,out/'build_comparison.py')
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'checksums.json', 'results.zip'} and not {'qa', '__pycache__'}.intersection(p.relative_to(out).parts)}
    (out/'checksums.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'output':str(out),'series':[{k:v for k,v in p.items() if k in ['id','label','count','valid_points','first_time_s','last_time_s','first_score_percent','last_score_percent']} for p in provenance],'files':len(hashes)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
