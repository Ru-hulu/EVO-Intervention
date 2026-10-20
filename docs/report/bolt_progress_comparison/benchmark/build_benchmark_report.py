"""Render benchmark artifacts remotely and extend the existing local report."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import textwrap
import zipfile

REMOTE=Path('/root/autodl-tmp/EvalProgress/outputs/benchmark_15tasks_3fps_20261016')
METHODS=[
 ('A','Robometer','Robometer','#202c3b','-', '从起点到当前时刻均匀抽 4 张主视图。'),
 ('B','ProgressLM · 文字示范','ProgressLM / text demo','#a84a9a','--','当前主视图＋该任务的 4 步文字示范（每步 25%）。'),
 ('C','ProgressLM · 视觉示范','ProgressLM / expert visual demo','#5c59cb','-','当前主视图＋轨迹 2 的 5 张阶段参考图（0/25/50/75/100%）。'),
 ('D','RoboBrain 2.5 · 空白目标','RoboBrain 2.5 / blank goal','#dd8a00','--','轨迹 1 首帧主视图＋空白目标＋首帧与当前帧三视图。'),
 ('E','RoboBrain 2.5 · 本段首尾','RoboBrain 2.5 / own endpoints','#c54832','-','轨迹 1 首尾主视图＋首帧与当前帧三视图。'),
 ('F','Dopamine Preview · 本段首尾','Dopamine Preview / own endpoints','#00886b','-','轨迹 1 首尾主视图＋首帧与当前帧三视图。'),
 ('G','Dopamine Preview · 专家首尾','Dopamine Preview / expert endpoints','#238fc0','--','轨迹 2 首尾主视图＋轨迹 1 首帧与当前帧三视图。')]

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def render():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw,ImageOps
    root=REMOTE;out=root/'publish';out.mkdir(exist_ok=True)
    tasks=json.loads((root/'prepared.json').read_text())['tasks']
    annotations=json.loads((root/'annotations.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    all_rows=[];entries=[]
    for t in tasks:
        name=t['id'];a=annotations[name];query=t['query'];reference=t['reference']
        folder=out/'data'/name;folder.mkdir(parents=True,exist_ok=True)
        curves={};counts={}
        for letter,*_ in METHODS:
            source=root/'results'/name/f'{letter}.jsonl'
            rows=list(map(json.loads,source.read_text().splitlines()))
            assert [r['frame_index'] for r in rows]==query['sample_indices'],(name,letter)
            assert all(r['task_id']==name and r['method_id']==letter for r in rows)
            assert all(r['instruction']==a['goal'] for r in rows)
            curves[letter]=rows;counts[letter]={'total':len(rows),'valid':sum(r['status']=='ok' for r in rows)}
            shutil.copy2(source,folder/f'{letter}.jsonl')
            for r in rows:
                all_rows.append({k:r[k] for k in ['task_id','method_id','frame_index','timestamp_s','score_percent','status','score_semantics']})
        fig,ax=plt.subplots(figsize=(13,5.6));fig.subplots_adjust(left=.075,right=.985,bottom=.12,top=.72)
        for letter,label,legend,color,style,desc in METHODS:
            rows=curves[letter]
            line,=ax.plot([r['timestamp_s'] for r in rows],[r['score_percent'] if r['status']=='ok' else np.nan for r in rows],
                         label=letter+'  '+legend,color=color,linestyle=style,linewidth=1.65,marker='o',markersize=2.4,alpha=.92)
            line.set_gid('curve-'+letter)
        ax.set(xlim=(0,query['last_time_s']),ylim=(-105,105),xlabel='Time in evaluated trajectory (s)',ylabel='Prediction / signed GRM score (%)')
        ax.set_yticks([-100,-50,0,25,50,75,100]);ax.axhline(0,color='#6b7280',lw=.7,zorder=0)
        ax.axhspan(-105,0,color='#f3f5f7',zorder=-2);ax.grid(alpha=.16)
        title=t['benchmark']+' / '+t['task'].replace('KITCHEN_SCENE2_','').replace('KITCHEN_SCENE8_','').replace('KITCHEN_SCENE10_','').replace('_',' ')
        fig.suptitle('\n'.join(textwrap.wrap(title,95)),x=.075,y=.985,ha='left',fontsize=14,fontweight='bold')
        fig.text(.075,.87,'Evaluated: trajectory 1  |  Expert: trajectory 2  |  3 FPS + final observation',fontsize=9,color='#576271')
        legend=fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.068,.855),ncol=3,frameon=False,fontsize=9,handlelength=2.4,columnspacing=1.6)
        for (letter,*_),line,label in zip(METHODS,legend.get_lines(),legend.get_texts()):
            line.set_gid('legend-line-'+letter);label.set_gid('legend-label-'+letter)
        fig.text(.075,.025,'Raw predictions; no smoothing or forced endpoints. Missing / n/a outputs remain gaps.',fontsize=9,color='#576271')
        charts=out/'charts';charts.mkdir(exist_ok=True)
        fig.savefig(charts/f'{name}.png',dpi=180);fig.savefig(charts/f'{name}.svg');plt.close(fig)
        media=out/'media'/name;media.mkdir(parents=True,exist_ok=True)
        for role in ['query','reference']:
            shutil.copy2(t[role]['preview_path'],media/f'{role}.mp4')
            with Image.open(t[role]['frames']['0'][0]) as im:im.convert('RGB').save(media/f'{role}_poster.jpg',quality=90)
        strip=Image.new('RGB',(1500,250),'#f5f7f8');draw=ImageDraw.Draw(strip)
        refs=[]
        for j,anchor in enumerate(a['reference_anchors']):
            p=Path(reference['frames'][str(anchor['frame_index'])][0]);dest=media/f'anchor_{j}.png';shutil.copy2(p,dest)
            with Image.open(p) as im:
                fitted=ImageOps.contain(im,(292,218));strip.paste(fitted,(300*j+(300-fitted.width)//2,0))
            draw.text((300*j+8,225),f'{anchor["progress_percent"]}% | {anchor["timestamp_s"]:.2f}s | frame {anchor["frame_index"]}',fill='#24313a')
            refs.append({**anchor,'sha256':sha(p),'image':f'media/{name}/anchor_{j}.png'})
        strip.save(media/'anchors.jpg',quality=92)
        instruction={'task':t['task'],'goal':a['goal'],'text_steps':a['text_steps'],'text_step_percent':a['text_step_percent'],
                     'visual_anchors':refs,'anchor_basis':a['anchor_basis'],'view_note':t['missing_views'],
                     'query_record':t['query_record'],'reference_record':t['reference_record'],
                     'instance_difference':a.get('instance_difference'),
                     'query_fps':query['fps'],'query_frame_count':query['frame_count'],'query_sample_indices':query['sample_indices'],
                     'reference_fps':reference['fps'],'reference_frame_count':reference['frame_count'],
                     'query_hdf5_video_note':'RoboTwin uses 15 Hz HDF5 observations; preview has one extra rendered frame and is for viewing only.' if t['benchmark']=='RoboTwin' else None}
        dump(folder/'inputs.json',instruction)
        shutil.copytree(root/'prompts'/name,folder/'prompts',dirs_exist_ok=True)
        entries.append({'id':name,'benchmark':t['benchmark'],'task':t['task'],'title_zh':a['title_zh'],'goal':a['goal'],
                        'sample_count':len(query['sample_indices']),'duration_s':query['last_time_s'],'counts':counts,'text_steps':a['text_steps'],
                        'reference_anchors':refs,'instance_note':'两段托盘颜色分配不同；按共同的“按类别分入不同托盘”目标评测。' if a.get('instance_difference') else '',
                        'views':{'RoboDojo':'主视图＋双腕空白','LIBERO':'主视图＋原始单腕＋右腕空白','RoboTwin':'真实主视图＋左右腕视图'}[t['benchmark']]})
        print('RENDERED',name,flush=True)
    with (out/'all_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    shutil.copytree(root/'runtime',out/'runtime',dirs_exist_ok=True)
    for name in ['annotations.json','prepare_benchmark.py','run_benchmark.py']:
        shutil.copy2(root/name,out/name)
    runtime={name:json.loads((root/'runtime'/f'{name}.json').read_text()) for name in ['robometer','brain','dopamine','progresslm']}
    stats={'tasks':len(entries),'series':len(entries)*7,'predictions':len(all_rows),'valid_predictions':sum(r['status']=='ok' for r in all_rows),
           'status_counts':dict(__import__('collections').Counter(r['status'] for r in all_rows)),
           'model_stage_wall_s':sum(x['total_wall_s'] for x in runtime.values()),'paused_task':'RoboTwin / place_mouse_pad',
           'reason':'Expert and query target pad colors differ; user requested pause.'}
    gpu_rows=[]
    for p in (root/'runtime').glob('gpu_usage*.csv'):
        with p.open() as f:
            for row in csv.DictReader(f):
                row={k.strip():v.strip() for k,v in row.items()}
                gpu_rows.append((float(row['utilization.gpu [%]'].split()[0]),float(row['memory.used [MiB]'].split()[0])))
    stats['gpu_peak_utilization_percent']=max(x[0] for x in gpu_rows)
    stats['gpu_peak_memory_gib']=max(x[1] for x in gpu_rows)/1024
    stats['excluded_results']='Superseded classification run that imposed the expert tray-color mapping; all seven classification series rerun with a shared category-sorting goal.'
    dump(out/'manifest.json',{'summary':stats,'tasks':entries,'methods':METHODS,'remote_source':str(root)})
    dump(out/'checksums.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='checksums.json'})
    print('COMPLETE',json.dumps(stats),flush=True)

def prefixed_svg(text,prefix):
    text=text[text.index('<svg'):]
    ids=re.findall(r'\bid="([^"]+)"',text)
    for id in sorted(ids,key=len,reverse=True):
        text=text.replace(f'id="{id}"',f'id="{prefix}-{id}"').replace(f'url(#{id})',f'url(#{prefix}-{id})').replace(f'href="#{id}"',f'href="#{prefix}-{id}"')
    return text

def integrate(root):
    bench=root/'benchmark';bundle=bench/'results';manifest=json.loads((bundle/'manifest.json').read_text())
    original=root/'bolt_original.html'
    if not original.exists():shutil.copy2(root/'report.html',original)
    page=original.read_text()
    sections=[]
    for number,t in enumerate(manifest['tasks'],1):
        id=t['id'];prefix=f'bench-{number}';asset=f'benchmark/results/media/{id}'
        checks=''.join(f'<label class="curve-option" style="--series-color:{color}"><input type="checkbox" data-benchmark-curve="{letter}" checked><span><b>{letter}</b> {html.escape(label)}</span></label>' for letter,label,legend,color,style,desc in METHODS)
        svg=prefixed_svg((bundle/'charts'/f'{id}.svg').read_text(),prefix)
        counts=t['counts'];missing=sum(v['total']-v['valid'] for v in counts.values())
        missing_note=f' · {missing} 个 n/a／无效输出保留为空缺' if missing else ''
        videos=''.join(f'<figure><figcaption>{caption}</figcaption><video controls preload="none" playsinline poster="{asset}/{role}_poster.jpg" data-src="{asset}/{role}.mp4"></video></figure>' for role,caption in [('query','待评测轨迹 1'),('reference','专家参考轨迹 2')])
        steps='；'.join(f'{i+1}. {html.escape(s)}' for i,s in enumerate(t['text_steps']))
        sections.append(f'''<section class="benchmark-task" id="{prefix}" data-task="{id}">
<h2>{number:02d} · {html.escape(t['title_zh'])} <small>{t['benchmark']}</small></h2>
<p class="muted">{html.escape(t['task'])} · 轨迹 1 评测 / 轨迹 2 参考 · {t['sample_count']} 点 · {t['views']}{missing_note} {html.escape(t['instance_note'])}</p>
<div class="curve-controls"><div class="control-actions"><b>显示曲线</b><button data-benchmark-select="all" type="button">全选</button><button data-benchmark-select="none" type="button">全不选</button><output class="benchmark-count" aria-live="polite">显示 7 / 7</output></div><div class="curve-options">{checks}</div></div>
<div class="plot">{svg}</div>
<details class="task-media"><summary>展开示范视频与参考帧</summary><div class="video-grid">{videos}</div><img class="anchor-strip" src="{asset}/anchors.jpg" loading="lazy" alt="专家轨迹五个阶段参考帧及其进度锚点"><p class="muted">文字示范：{steps}</p></details>
<p class="muted"><a href="benchmark/results/charts/{id}.png">PNG</a> · <a href="benchmark/results/charts/{id}.svg">SVG</a> · <a href="benchmark/results/data/{id}/inputs.json">实际输入配置</a></p></section>''')
    table=''.join(f'<tr><td><span class="dot" style="background:{color}"></span><b>{letter}</b> {html.escape(label)}</td><td>{html.escape(desc)}</td></tr>' for letter,label,legend,color,style,desc in METHODS)
    options=''.join(f'<option value="{t["id"]}">{i+1:02d} · {html.escape(t["title_zh"])} ({t["benchmark"]})</option>' for i,t in enumerate(manifest['tasks']))
    s=manifest['summary']
    intro=f'''<hr><div id="benchmark-results"><h1>新增评测：15 个任务 · 每任务 1 条轨迹</h1>
<p>第 1 条用于评测，第 2 条仅作参考；统一 3 FPS 并保留末帧。共 {s['series']} 条曲线、{s['predictions']} 个输出。鼠标任务因目标颜色不同，按要求暂停。</p>
<p class="muted">缺失腕视图用白色空白图补齐；LIBERO 的单腕图放入左腕槽，右腕为空白。阶段进度为人工参考锚点，无逐帧进度真值；n/a 保留为空缺。“本段首尾”包含未来画面。RoboTwin 按 HDF5 的 15 Hz 观测计算，播放视频比观测多 1 帧。</p>
<details class="benchmark-inputs"><summary>各方法的实际输入（展开）</summary><table><thead><tr><th>方法 / 设置</th><th>每次评测实际输入（均附任务文字）</th></tr></thead><tbody>{table}</tbody></table></details>
<p class="muted">模型批量评测合计 {s['model_stage_wall_s']/60:.1f} 分钟（含初始化与分类任务修正，非单帧延迟）；GPU 峰值 {s['gpu_peak_utilization_percent']:.0f}%、显存 {s['gpu_peak_memory_gib']:.1f} GiB。<a href="benchmark/results/all_predictions.csv">合并数据 CSV</a> · <a href="benchmark/results/manifest.json">结果与来源</a> · <a href="results.zip">完整离线结果包</a></p>
<label class="task-filter">查看任务 <select id="task-filter"><option value="all">全部 15 个任务</option>{options}</select></label></div>'''
    css='''<style>hr{border:0;border-top:1px solid #cbd6dd;margin:22px 0 16px}.benchmark-task{margin:18px 0 24px;scroll-margin-top:12px}.benchmark-task h2{font-size:19px;margin:0 0 3px}.benchmark-task h2 small{font-size:12px;color:#62707b;font-weight:400;margin-left:7px}.benchmark-task .muted{overflow-wrap:anywhere}.task-media,.benchmark-inputs{border:1px solid #dce3e7;border-radius:5px;background:white;padding:7px 10px;margin:6px 0}summary{cursor:pointer;color:#096b82;font-weight:600}.video-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}.video-grid figure{margin:0;min-width:0}.video-grid figcaption{font-size:12px;margin-bottom:4px}.video-grid video{display:block;width:100%;max-height:360px;background:#111}.anchor-strip{display:block;width:100%;height:auto;margin-top:8px}.task-filter{display:block;margin:12px 0;font-size:13px}.task-filter select{max-width:100%;font:inherit;padding:5px;border:1px solid #bccbd4;border-radius:4px}.benchmark-inputs table{table-layout:auto}.benchmark-inputs td:last-child{white-space:normal;width:auto}@media(max-width:680px){.video-grid{grid-template-columns:1fr}.benchmark-inputs{overflow-x:auto}.benchmark-task h2{font-size:17px}}</style>'''
    script='''<script>
document.querySelectorAll('.benchmark-task').forEach(section=>{
 const checks=[...section.querySelectorAll('[data-benchmark-curve]')];
 const update=()=>{
  checks.forEach(check=>{
   const suffix=check.dataset.benchmarkCurve;
   section.querySelector('#'+section.id+'-curve-'+suffix).style.display=check.checked?'':'none';
   ['legend-line-','legend-label-'].forEach(part=>section.querySelector('#'+section.id+'-'+part+suffix).style.opacity=check.checked?'1':'.25');
  });
  section.querySelector('.benchmark-count').textContent='显示 '+checks.filter(c=>c.checked).length+' / 7';
 };
 checks.forEach(check=>check.addEventListener('change',update));
 section.querySelectorAll('[data-benchmark-select]').forEach(button=>button.addEventListener('click',()=>{checks.forEach(check=>check.checked=button.dataset.benchmarkSelect==='all');update()}));
 section.querySelector('.task-media').addEventListener('toggle',event=>{
  section.querySelectorAll('video').forEach(video=>{
   if(event.target.open && !video.getAttribute('src')){video.src=video.dataset.src;video.load()}
   if(!event.target.open)video.pause();
  });
 });
 update();
});
document.getElementById('task-filter').addEventListener('change',event=>{
 document.querySelectorAll('.benchmark-task').forEach(section=>{
  section.hidden=event.target.value!=='all' && section.dataset.task!==event.target.value;
  if(section.hidden)section.querySelectorAll('video').forEach(video=>video.pause());
 });
});
</script>'''
    page=page.replace('<title>23 秒插螺栓 · 进度预测汇总</title>','<title>进度预测对比 · 插螺栓与 15 个任务</title>')
    page=page.replace('<main>','<main><p class="muted"><a href="#benchmark-results">跳转到新增 15 个任务评测 ↓</a></p>',1)
    page=page.replace('</main>',css+intro+''.join(sections)+'</main>')
    page=page.replace('</html>',script+'</html>')
    (root/'report.html').write_text(page)
    # Preserve the original report, raw outputs, new videos, plots and reproducible scripts together.
    hashes={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {'results.zip','checksums.json'} and not {'qa','__pycache__'}.intersection(p.relative_to(root).parts)}
    dump(root/'checksums.json',hashes)
    with zipfile.ZipFile(root/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*hashes,'checksums.json']:z.write(root/name,name)
    print('INTEGRATED',len(sections),'new plots;',len(hashes),'files')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');p.add_argument('--integrate',type=Path);a=p.parse_args()
    if a.render:render()
    if a.integrate:integrate(a.integrate)
