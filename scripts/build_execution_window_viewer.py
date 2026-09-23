"""Build the execution-window ACC viewer from rescored CSVs and prior camera videos."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil


def build(reference, output):
    payload=json.loads((reference/'data.js').read_text(encoding='utf-8').split('=',1)[1].strip().rstrip(';'))
    result=json.loads((output/'RESULTS.json').read_text(encoding='utf-8'))
    by_id={r['episode_id']:r for r in result['episodes']}
    for ep in payload['episodes']:
        with (output/'reports'/ep['id']/'scores.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        def series(k):return [float(r[k]) if r[k] else None for r in rows]
        ep.update(times=series('time_s'),left=series('acc_left_mm'),right=series('acc_right_mm'),rawLeft=series('acc_raw_left_mm'),rawRight=series('acc_raw_right_mm'),max=series('acc_max_mm'),p95=by_id[ep['id']]['ema_p95_mm'])
        ep['src']=os.path.relpath(reference/ep['src'],output).replace('\\','/')
    payload['method'].update(metric='execution_window_xyz_mae',units='mm',normalized=False,comparisonSteps=6,previousSlice=[6,12],currentSlice=[0,6],vMinMeters=None,source='Rescored identical cached predictions, no new model inference')
    (output/'data.js').write_text('window.ACC_DATA = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    html=(reference/'index.html').read_text(encoding='utf-8')
    html=html.replace('<title>ACC 片段查看器</title>','<title>ACC · 执行6步 · 无归一化</title>')
    html=html.replace('<h1>ACC 片段查看器</h1>','<h1>ACC · 执行6步</h1>')
    html=html.replace('<h2>片段 ACC</h2>','<h2>动作差异（mm）</h2>')
    html=html.replace('平滑 ACC（EMA）','平滑误差 EMA（mm）').replace('原始 ACC</option>','原始误差（mm）</option>')
    html=html.replace('id="zero" type="checkbox"','id="zero" type="checkbox" checked')
    html=html.replace('对比时两条轨迹使用相同纵轴','同一区间下四条轨迹共用纵轴（mm）')
    start=html.index('  <details class="method-note downloads"')
    end=html.index('</details>',start)+len('</details>')
    html=html[:start]+'''  <details class="method-note downloads" open>
    <summary>本次结果、曲线与计算说明</summary>
    <p>仅比较 old[6:12] 与 new[0:6]，不除以速度或位置跨度；单位为毫米。相机画面复用原录像，右侧分数为本次重算结果。</p>
    <div class="download-links"><a href="REPORT.md">公式与结果分析</a><a href="metrics.csv" download>汇总 CSV</a><a href="curves.png">四条完整曲线</a><a href="window_comparison.png">6 步与 44 步对比</a><a href="VERIFICATION.json">计算核验记录</a></div>
  </details>'''+html[end:]
    (output/'index.html').write_text(html,encoding='utf-8')
    js=(reference/'app.js').read_text(encoding='utf-8')
    js=js.replace("const previous = method.fixedPreviousSelection;", "const previous = method.fixedPreviousSelection;")
    start=js.index("  $('parameter-note').textContent")
    end=js.index('  const clamp',start)
    js=js[:start]+'''  $('parameter-note').textContent = `EMA α = ${method.emaAlpha} · 无归一化 · 仅比较接下来 6 步 · 单位 mm`;
  $('method-explanation').textContent = '本次复用相同的 15k 模型动作预测缓存，仅重新计算指标：上次 [6:12] 对本次 [0:6]，对 6×3 个 XYZ 坐标差取平均绝对值，再从米换算为毫米；不做速度/跨度归一化，不是欧氏距离。EMA α=0.9、初始为0，首个观测无评分。相机画面复用原视频中的相机区域，右侧曲线和数值为本次重算。执行窗口指计划执行的前6步，不是与真实执行轨迹的误差。HF 示范无原生成功标签；四条样本延用旧选样，本次不重新排名。此为自定义ACC变体，不是论文原始指标。';
'''+js[end:]
    js=js.replace("· ACC ${hz} Hz${previous ? ' 重算结果' : ''}","· ${hz} Hz · 无归一化（mm）")
    js=js.replace('P95 ${e.p95.toFixed(3)} ·','P95 ${e.p95.toFixed(3)} mm ·')
    js=js.replace('    state.episodes.forEach(e => columns(e).forEach(series => series.forEach((v, i) => {','    DATA.episodes.forEach(e => columns(e).forEach(series => series.forEach((v, i) => {')
    js=js.replace("`${state.episodes.length === 2 ? '两图共用纵轴' : '片段内自动缩放'} ${lo.toFixed(2)}–${hi.toFixed(2)}`", "`四条共用纵轴 ${lo.toFixed(2)}–${hi.toFixed(2)} mm`")
    js=js.replace('`左 ${number(values[0])}`','`左 ${number(values[0])} mm`').replace('`右 ${number(values[1])}`','`右 ${number(values[1])} mm`')
    (output/'app.js').write_text(js,encoding='utf-8')
    for name in ['styles.css','serve.py']:
        shutil.copy2(reference/name,output/name)
    print(json.dumps({'viewer':str(output/'index.html'),'episodes':4,'units':'mm'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.reference.resolve(),a.output.resolve())
