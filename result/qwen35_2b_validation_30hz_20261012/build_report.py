"""Rebuild the technical report and figures using only the eight retained JSON files.

Run with Python + numpy + matplotlib. No model, GPU, network, or project imports.
"""
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
import hashlib
import json
import math
import re
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, RED = '#234a70', '#d77726', '#19845b', '#c9414b'
FIELDS = ('nowid', 'finishid', 'restid')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def state(x):
    return x['nowid'], tuple(sorted(x['finishid'])), tuple(sorted(x['restid']))


def agreement(row):
    return state(row['prediction']) == state(row['ground_truth'])


def metrics(rows):
    errors = np.array([100*(r['prediction']['progress']-r['ground_truth']['progress']) for r in rows])
    absolute = np.abs(errors)
    confusion = np.zeros((4, 4), dtype=int)
    for r in rows:
        confusion[r['ground_truth']['nowid']-1, r['prediction']['nowid']-1] += 1
    per_stage = []
    for i in range(4):
        tp, truth, predicted = int(confusion[i,i]), int(confusion[i].sum()), int(confusion[:,i].sum())
        per_stage.append(dict(stage=i+1, support=truth,
            precision=tp/predicted if predicted else 0, recall=tp/truth if truth else 0,
            f1=2*tp/(truth+predicted) if truth+predicted else 0))
    result = dict(n=len(rows), mae_pp=float(absolute.mean()), rmse_pp=float(np.sqrt(np.mean(errors**2))),
        bias_pp=float(errors.mean()), p95_pp=float(np.sort(absolute)[math.ceil(.95*len(rows))-1]),
        within_5pp=float(np.mean(absolute<=5)), within_10pp=float(np.mean(absolute<=10)),
        current_stage_accuracy=float(np.trace(confusion)/len(rows)),
        full_state_accuracy=sum(map(agreement,rows))/len(rows), confusion=confusion.tolist(),
        macro_f1=float(np.mean([v['f1'] for v in per_stage])), per_stage=per_stage)
    for key in ('finishid','restid'):
        tp=fp=fn=exact=0
        for r in rows:
            pred, gt=set(r['prediction'][key]),set(r['ground_truth'][key])
            tp+=len(pred & gt);fp+=len(pred-gt);fn+=len(gt-pred);exact+=pred==gt
        result[key+'_accuracy']=exact/len(rows)
        result[key+'_micro_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1
    return result


def error_runs(rows):
    runs=[]
    for i,r in enumerate(rows):
        if agreement(r):continue
        if runs and runs[-1]['end_index']==i-1:
            runs[-1].update(end_index=i,end_seconds=r['episode_time_seconds'],frames=runs[-1]['frames']+1)
        else:
            runs.append(dict(start_index=i,end_index=i,start_seconds=r['episode_time_seconds'],
                             end_seconds=r['episode_time_seconds'],frames=1))
    return runs


def rollback_events(ep):
    rows=ep['rows'];events=[]
    for i in range(1,len(rows)):
        before,after=rows[i-1],rows[i]
        drop=100*(before['ground_truth']['progress']-after['ground_truth']['progress'])
        if drop<=3:continue
        t=after['episode_time_seconds']
        # Do not count a later match after the true stage state has changed again.
        stop=min(len(rows),i+61)
        for j in range(i+1,stop):
            if state(rows[j]['ground_truth'])!=state(after['ground_truth']):stop=j;break
        matches=[j for j in range(i,stop) if state(rows[j]['prediction'])==state(after['ground_truth'])]
        nearby=[j for j in range(1,len(rows)) if abs(rows[j]['episode_time_seconds']-t)<=2]
        peak=max(nearby,key=lambda j:rows[j-1]['prediction']['progress']-rows[j]['prediction']['progress'])
        events.append(dict(dataset=ep['dataset'],video_id=ep['video_id'],episode=ep['episode'],
            episode_id=ep['id'],time_seconds=t,frame_index=after['frame_index'],gt_drop_pp=drop,
            adjacent_predicted_drop_pp=100*(before['prediction']['progress']-after['prediction']['progress']),
            nearby_largest_drop_pp=100*(rows[peak-1]['prediction']['progress']-rows[peak]['prediction']['progress']),
            nearby_largest_drop_time_seconds=rows[peak]['episode_time_seconds'],
            first_post_event_state_match_seconds=(matches[0]-i)/30 if matches else None))
    return events


def read_data():
    videos=[];episodes=[];inputs=[];seen=set();stage_names=None
    for path in sorted(ROOT.glob('*/*.json')):
        document=json.loads(path.read_text())
        if not {'dataset','video_id','frames','stage_names','fps'}<=document.keys():continue
        assert document['fps']==30 and document['progress_range']==[0,1]
        if stage_names is None:stage_names=document['stage_names']
        assert document['stage_names']==stage_names
        grouped=defaultdict(list)
        group='recovery' if 'intentional' in document['dataset'] else 'simple'
        for r in document['frames']:
            identity=(document['dataset'],document['video_id'],r['task_index'],r['episode'],r['frame_index'])
            assert identity not in seen,identity
            seen.add(identity)
            assert abs(r['source_time_seconds']-r['frame_index']/30)<1e-8
            for key in ('prediction','ground_truth'):
                v=r[key]
                assert math.isfinite(v['progress']) and 0<=v['progress']<=1
                assert v['nowid'] in (1,2,3,4)
                for k in ('finishid','restid'):
                    assert len(v[k])==len(set(v[k])) and set(v[k])<={1,2,3,4}
            grouped[(r['task_index'],r['episode'])].append(r)
        video=dict(dataset=document['dataset'],video_id=document['video_id'],group=group,
                   id=group+'-'+document['video_id'],path=path.relative_to(ROOT).as_posix(),rows=document['frames'],episodes=[])
        for (task,number),rows in sorted(grouped.items()):
            assert [r['frame_index'] for r in rows]==list(range(rows[0]['frame_index'],rows[-1]['frame_index']+1))
            for i,r in enumerate(rows):assert abs(r['episode_time_seconds']-i/30)<1e-8
            ep=dict(dataset=document['dataset'],video_id=document['video_id'],group=group,task_index=task,
                episode=number,id=f'{group}-{document["video_id"]}-ep{number:02d}',rows=rows,
                figure_dir=f'figures/{group}/{document["video_id"]}/episode-{number:02d}')
            ep['metrics']=metrics(rows);ep['error_runs']=error_runs(rows)
            ep['events']=rollback_events(ep)
            changed=[i for i in range(1,len(rows)) if state(rows[i]['ground_truth'])!=state(rows[i-1]['ground_truth'])]
            near=set()
            for i in changed:near.update(range(max(0,i-15),min(len(rows),i+16)))
            ep['near_transition_indices']=near;ep['transitions']=len(changed)
            episodes.append(ep);video['episodes'].append(ep)
        video['metrics']=metrics(video['rows']);videos.append(video)
        inputs.append(dict(path=video['path'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           frames=len(document['frames']),episodes=len(grouped)))
    assert len(videos)==8 and len(episodes)==40 and len(seen)==23604
    return videos,episodes,inputs,stage_names


def save(fig,stem):
    path=ROOT/stem;path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=170)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def plot_pt(ax,ep,compact=False):
    rows=ep['rows'];t=[r['episode_time_seconds'] for r in rows]
    ax.plot(t,[100*r['ground_truth']['progress'] for r in rows],c=BLUE,lw=1.8,label='Ground truth')
    ax.plot(t,[100*r['prediction']['progress'] for r in rows],c=ORANGE,lw=.95,label='Prediction',
            marker=None if compact else 'o',markersize=1.2)
    for ev in ep['events']:ax.axvline(ev['time_seconds'],color='#9e567e',ls='--',lw=.8,alpha=.7)
    ax.set(xlim=(-.2,t[-1]+.2),ylim=(-3,106),xlabel='Episode time (s)',ylabel='Progress PT (%)')
    ax.grid(alpha=.14)
    if not compact:ax.legend(loc='upper left',frameon=False,ncol=2)


def plot_stage(ax,ep,compact=False):
    rows=ep['rows'];t=[r['episode_time_seconds'] for r in rows]
    good=[agreement(r) for r in rows]
    ax.axhline(0,c='#d6dce3',lw=.35,zorder=1)
    for value,color in [(True,GREEN),(False,RED)]:
        points=[v for v,m in zip(t,good) if m==value]
        ax.scatter(points,[0]*len(points),s=2 if not compact else 3,c=color,edgecolors='none',zorder=3 if value else 4)
    for ev in ep['events']:ax.axvline(ev['time_seconds'],color='#9e567e',ls='--',lw=.8,alpha=.6)
    ax.set(xlim=(-.2,t[-1]+.2),ylim=(-.35,.6),yticks=[],xlabel='Episode time (s)')
    ax.spines['left'].set_visible(False)
    if not compact:
        ax.legend(handles=[Line2D([],[],ls='',marker='o',c=GREEN,label='All stage fields match'),
                           Line2D([],[],ls='',marker='o',c=RED,label='At least one field differs')],
                  loc='upper center',ncol=2,frameon=False,fontsize=9)


def episode_figures(ep):
    m=ep['metrics'];title=f'{ep["group"].title()} / {ep["video_id"]} / episode {ep["episode"]} | 30 Hz'
    fig,ax=plt.subplots(figsize=(11.5,4.2),layout='constrained');plot_pt(ax,ep)
    ax.set_title(title+f'\nPT MAE {m["mae_pp"]:.2f} pp; RMSE {m["rmse_pp"]:.2f} pp',fontsize=12)
    save(fig,ep['figure_dir']+'/pt_curve')
    fig,ax=plt.subplots(figsize=(11.5,2.5),layout='constrained');plot_stage(ax,ep)
    correct=sum(agreement(r) for r in ep['rows'])
    ax.set_title(title+f'\nFull state match {correct}/{m["n"]} ({m["full_state_accuracy"]:.1%})',fontsize=12)
    save(fig,ep['figure_dir']+'/stage_agreement')


def video_figure(video):
    fig,axes=plt.subplots(len(video['episodes']),2,figsize=(15,14),layout='constrained',gridspec_kw={'width_ratios':[1.35,1]})
    fig.suptitle(f'{video["group"].title()} / {video["video_id"]} | all episodes | 30 Hz closed loop',fontsize=16)
    for ep,(left,right) in zip(video['episodes'],axes):
        plot_pt(left,ep,True);plot_stage(right,ep,True);m=ep['metrics']
        left.set_title(f'Episode {ep["episode"]}: PT MAE {m["mae_pp"]:.2f} pp',fontsize=11)
        right.set_title(f'Full stage state: {m["full_state_accuracy"]:.1%} correct',fontsize=11)
    axes[0,0].legend(frameon=False,ncol=2,fontsize=8)
    save(fig,f'figures/{video["group"]}/{video["video_id"]}/overview')


def summary_figures(all_rows,videos,summary,near_metrics):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    selections=[('All',all_rows,BLUE),('Simple',[r for v in videos if v['group']=='simple' for r in v['rows']],GREEN),
                ('Recovery',[r for v in videos if v['group']=='recovery' for r in v['rows']],ORANGE)]
    for name,rows,color in selections:
        a=np.sort([abs(r['prediction']['progress']-r['ground_truth']['progress'])*100 for r in rows])
        ax.plot(a,np.arange(1,len(a)+1)/len(a)*100,c=color,lw=1.7,label=name)
    for value in (5,10):ax.axvline(value,color='#8393a0',ls=':',lw=.8)
    ax.set(xlim=(0,40),ylim=(0,101),xlabel='Absolute PT error (percentage points)',ylabel='Frames within threshold (%)',title='Progress error distribution')
    ax.legend(frameon=False);ax.grid(alpha=.14)
    y=np.arange(len(videos));values=[v['metrics']['mae_pp'] for v in videos]
    bx.barh(y,values,color=[ORANGE if v['group']=='recovery' else GREEN for v in videos],height=.62)
    bx.set(yticks=y,yticklabels=[f'{v["group"][0].upper()} / {v["video_id"]}' for v in videos],xlabel='PT MAE (percentage points)',title='Video-level progress error',xlim=(0,max(values)*1.22))
    bx.invert_yaxis();bx.grid(axis='x',alpha=.14)
    for i,v in enumerate(values):bx.text(v+.1,i,f'{v:.2f}',va='center',fontsize=9)
    save(fig,'figures/summary/progress_errors')
    c=np.asarray(summary['confusion']);percent=c/np.maximum(c.sum(axis=1,keepdims=True),1)*100
    fig,ax=plt.subplots(figsize=(6.8,5.1),layout='constrained')
    im=ax.imshow(percent,cmap='Blues',vmin=0,vmax=100)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{c[i,j]:,}\n{percent[i,j]:.1f}%',ha='center',va='center',color='white' if percent[i,j]>55 else '#173449',fontsize=11)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=['S1','S2','S3','S4'],yticklabels=['S1','S2','S3','S4'],xlabel='Predicted stage',ylabel='Ground-truth stage',title='Current stage confusion | count + row percentage')
    fig.colorbar(im,ax=ax,label='Within ground-truth stage (%)',shrink=.8)
    save(fig,'figures/summary/stage_confusion')
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    labels=['All','Simple','Recovery'];x=np.arange(3)
    for offset,key,color,label in [(-.18,'near',ORANGE,'Within ±0.5 s of a true state change'),(.18,'stable',GREEN,'Outside those windows')]:
        vals=[near_metrics[g][key]['full_state_accuracy']*100 for g in ['all','simple','recovery']]
        bars=ax.bar(x+offset,vals,width=.34,color=color,label=label)
        for bar,g in zip(bars,['all','simple','recovery']):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.1f}%\nn={near_metrics[g][key]["n"]:,}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,118),ylabel='Full stage state accuracy (%)',title='Transition windows are harder than stable intervals')
    ax.set_yticks([0,20,40,60,80,100])
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.14)
    save(fig,'figures/summary/transition_accuracy')


def fmt(value):return f'{value:.2f}'
def pct(value):return f'{value:.2%}'
def label(group):return '恢复数据' if group=='recovery' else '简单数据'


def image_block(stem,caption):
    return f'<figure><a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{escape(caption)}"></a><figcaption>{escape(caption)}　<a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></figcaption></figure>'


def main():
    videos,episodes,inputs,stage_names=read_data()
    all_rows=[r for v in videos for r in v['rows']];overall=metrics(all_rows)
    by_group={g:metrics([r for v in videos if v['group']==g for r in v['rows']]) for g in ('simple','recovery')}
    near_metrics={}
    for group in ('all','simple','recovery'):
        eps=[e for e in episodes if group=='all' or e['group']==group]
        near_metrics[group]={name:metrics([r for ep in eps for i,r in enumerate(ep['rows']) if (i in ep['near_transition_indices'])==is_near])
                             for name,is_near in [('near',True),('stable',False)]}
    events=[ev for ep in episodes for ev in ep['events']]
    # Baseline values from the delivered evaluation; protects metric conventions.
    assert math.isclose(overall['mae_pp'],6.905034781219778,abs_tol=1e-10)
    assert math.isclose(overall['full_state_accuracy'],0.9447551262497882,abs_tol=1e-12)
    for ep in episodes:episode_figures(ep)
    for video in videos:video_figure(video)
    summary_figures(all_rows,videos,overall,near_metrics)
    generated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    stats=dict(generated_at=generated,source='retained per-frame JSON only; no inference',inputs=inputs,
        integrity=dict(videos=len(videos),episodes=len(episodes),unique_frames=len(all_rows),fps=30,
                       continuous_within_episode=True,source_and_episode_timestamps_consistent=True),
        overall=overall,by_dataset_type=by_group,by_video={v['id']:v['metrics'] for v in videos},
        episode_mean_mae_pp=float(np.mean([ep['metrics']['mae_pp'] for ep in episodes])),
        episode_mean_full_state_accuracy=float(np.mean([ep['metrics']['full_state_accuracy'] for ep in episodes])),
        transitions=near_metrics,rollback_events=events,
        episodes=[{k:ep[k] for k in ('id','dataset','video_id','episode','metrics','error_runs','events','transitions','figure_dir')} for ep in episodes])
    (ROOT/'report_metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    write_html(videos,episodes,stage_names,stats)
    # Source JSON files must remain byte-for-byte unchanged.
    for item in inputs:assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    print(json.dumps(dict(status='complete',videos=8,episodes=40,frames=len(all_rows),
        episode_figures=80,video_overviews=8,summary_figures=3,overall=overall),ensure_ascii=False))


def write_html(videos,episodes,stage_names,stats):
    m=stats['overall'];by=stats['by_dataset_type'];near=stats['transitions'];events=stats['rollback_events']
    worst=max(episodes,key=lambda e:e['metrics']['mae_pp']);weakest=min(episodes,key=lambda e:e['metrics']['full_state_accuracy'])
    longest=max(((run['frames'],ep,run) for ep in episodes for run in ep['error_runs']),key=lambda x:x[0])
    on_event=sum(ev['first_post_event_state_match_seconds']==0 for ev in events)
    unresolved=sum(ev['first_post_event_state_match_seconds'] is None for ev in events)
    h=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>Qwen3.5-2B · 验证集 30Hz 闭环评估技术报告</title>',
       '''<style>:root{--ink:#243a4b;--muted:#657989;--line:#d9e3eb;--accent:#176696}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}body{margin:0;background:#f5f7fa;color:var(--ink);font:16px/1.75 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1150px;margin:30px auto 70px;padding:0 26px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}nav{position:sticky;top:0;z-index:8;background:#fffffff2;border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:22px;flex-wrap:wrap;justify-content:center;font-size:14px}header{padding:28px 32px;background:#183c58;color:#fff;border-radius:15px;margin-bottom:24px}header h1{font-size:31px;line-height:1.35;margin:8px 0 14px}header p{margin:4px 0;color:#dbe6ef}.eyebrow{font-size:12px;letter-spacing:2px}section{background:white;border:1px solid var(--line);border-radius:13px;padding:24px 28px;margin:22px 0}h2{font-size:24px;margin:0 0 17px}h3{font-size:19px;margin:24px 0 12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{padding:16px;border-radius:10px;border:1px solid var(--line);background:#fcfdfe}.value{display:block;font-size:29px;font-weight:730;margin-top:6px}.caption,.muted{color:var(--muted);font-size:14px}.note{border-left:3px solid #9cb7ca;padding:10px 16px;background:#f3f7fa}.good{color:#187e58}.bad{color:#bd3648}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f3f7fa;white-space:nowrap}.scroll{overflow-x:auto}figure{margin:22px 0}figure img{display:block;width:100%;height:auto;border:1px solid #edf0f3;background:white}figcaption{font-size:13px;color:var(--muted);margin:7px 0}details{border:1px solid var(--line);border-radius:9px;padding:14px 17px;margin:13px 0}summary{cursor:pointer;font-weight:650}.video{border-top:4px solid #859fb5}.episode{background:#fcfdff}code{font-size:12px;overflow-wrap:anywhere;background:#f2f5f8;padding:2px 4px}pre{white-space:pre-wrap;font-size:12px;background:#f3f6f8;padding:15px;border-radius:8px}.stage-svg svg{display:block;width:100%;height:auto}.tag{font-size:12px;border-radius:5px;padding:2px 7px;background:#edf4f8;color:#47667b}select{padding:9px;font:inherit;border:1px solid var(--line);border-radius:7px;max-width:100%}.facts{display:grid;grid-template-columns:1fr 1fr;gap:20px}ul{padding-left:22px}li{margin:7px 0}.print-only{display:none}@media(max-width:750px){main{padding:0 12px}section{padding:20px 15px}.cards{grid-template-columns:repeat(2,1fr)}.facts{display:block}header{padding:24px 20px}header h1{font-size:25px}nav{gap:12px}.value{font-size:24px}}@media print{nav{display:none}body{background:white}main{margin:0;max-width:none}section{break-inside:avoid;box-shadow:none}a{color:inherit}details{break-inside:avoid}.cards{grid-template-columns:repeat(4,1fr)}}</style></head><body>''',
       '<nav><a href="#summary">结论</a><a href="#method">评估方法</a><a href="#metrics">指标与分组</a><a href="#transitions">阶段与回退</a><a href="#gallery">图片索引</a><a href="#reproduce">数据与复现</a></nav><main>',
       '<header><div class="eyebrow">TECHNICAL EVALUATION REPORT / 30 HZ</div><h1>Qwen3.5-2B 验证集闭环预测<br>逐帧进度与阶段状态评估</h1><p>8 个视频 · 40 个 episode · 23,604 个标注帧 · epoch-04</p>',
       f'<p>原评估：2026-10-12　｜　报告重建：{escape(stats["generated_at"][:10])}（北京时间）</p></header>',
       '<section id="summary"><h2>1. 主要结论</h2><div class="cards">']
    for name,value in [('进度 MAE',f'{m["mae_pp"]:.2f} pp'),('进度 RMSE',f'{m["rmse_pp"]:.2f} pp'),('当前阶段准确率',pct(m['current_stage_accuracy'])),('完整阶段状态准确率',pct(m['full_state_accuracy']))]:
        h.append(f'<div class="card">{name}<span class="value">{value}</span></div>')
    h.append(f'</div><ul><li>进度平均绝对偏差为 <b>{m["mae_pp"]:.2f} 个百分点</b>；带符号偏差 {m["bias_pp"]:+.2f} pp，整体偏低。误差不超过 10 pp 的帧占 {pct(m["within_10pp"])}。</li><li>恢复数据 PT MAE 为 <b>{by["recovery"]["mae_pp"]:.2f} pp</b>，简单数据为 {by["simple"]["mae_pp"]:.2f} pp。两者视频数、长度和阶段占比不同，不能由此单独归因到某种视觉能力。</li><li>真实阶段状态切换前后 0.5 秒内，完整状态准确率为 <b>{pct(near["all"]["near"]["full_state_accuracy"])}</b>；窗口外为 {pct(near["all"]["stable"]["full_state_accuracy"])}。整体准确率不能替代阶段切换时的表现。</li><li>单 episode 最差进度误差出现在 <a href="#{worst["id"]}">{label(worst["group"])} / {worst["video_id"]} / ep{worst["episode"]}</a>，MAE {worst["metrics"]["mae_pp"]:.2f} pp；最低完整状态准确率为 {pct(weakest["metrics"]["full_state_accuracy"])}（{weakest["video_id"]} / ep{weakest["episode"]}）。</li></ul><p class="note">本报告仅重读目录中的 8 份逐帧 JSON，重新计算指标与绘图，没有重新运行模型。所有图展示实际保存的预测值，不平滑、不补造帧、不强制进度单调。</p></section>')
    h.append('<section id="method"><h2>2. 数据与评估方法</h2><div class="facts"><div><h3>已确认的运行协议</h3><ul><li>每个 episode 独立初始化，仅首次以阶段真值初始化提示，之后反馈上次模型预测的完整阶段状态。</li><li>30FPS 视频每次推进 1 帧，预测密度为 30Hz。每个视角取 t−5、t−4、t−3、t−2、t−1、t 秒的六帧；首末相隔 5 秒。</li><li>不足本 episode 历史的部分用黑色空白帧及无效掩码补齐；使用顶部、左腕、右腕三视角。</li><li>依据此前核验，微调时按完整视频划分：72 个训练视频、8 个验证视频，交集为 0。</li></ul></div><div><h3>本次重建可直接核验的内容</h3><ul><li>8 个视频、40 个 episode、23,604 个唯一帧；每个 episode 内帧号连续，时间戳与 30FPS 一致。</li><li>进度真值与预测均在 [0,1]；阶段采用当前、已完成、未完成三个字段，集合比较忽略顺序。</li><li>不计视频开头、末尾等未标注帧，不在不同 episode 之间连线或统计跨片段进度回退。</li><li>模型名称、检查点轮次与运行协议来自已确认的评估记录；精简 JSON 不含原提示、权重或原始生成文本，因此本次不重新审计反馈提示、权重身份，也不计算全文一致率或字符编辑率。</li></ul></div></div><p class="muted">这是验证集评估，并非独立测试集。相邻窗口高度重叠，帧之间相关；30Hz 表示视频时间上的预测密度，不是在线实时吞吐。任务与四阶段定义固定。</p><h3>阶段编码</h3><table><tr><th>ID</th><th style="text-align:left">阶段名称（原始标签）</th></tr>')
    for k,v in stage_names.items():h.append(f'<tr><td>{k}</td><td style="text-align:left">{escape(v)}</td></tr>')
    h.append('</table></section><section id="metrics"><h2>3. 指标定义与分组结果</h2><p>令 e = 100 ×（预测 PT − 真值 PT），单位 pp（百分点）。MAE = mean(|e|)，RMSE = √mean(e²)，Bias = mean(e)。P95 使用最近秩分位数；覆盖率是 |e|≤阈值的帧占比。以下总体与分组指标按帧数加权。</p><div class="scroll"><table><tr><th>范围</th><th>帧数</th><th>MAE / pp</th><th>RMSE / pp</th><th>Bias / pp</th><th>P95 / pp</th><th>≤5 pp</th><th>≤10 pp</th><th>完整状态</th></tr>')
    for name,x in [('全部',m),('简单数据',by['simple']),('恢复数据',by['recovery'])]:
        h.append(f'<tr><td>{name}</td><td>{x["n"]:,}</td><td>{x["mae_pp"]:.2f}</td><td>{x["rmse_pp"]:.2f}</td><td>{x["bias_pp"]:+.2f}</td><td>{x["p95_pp"]:.2f}</td><td>{pct(x["within_5pp"])}</td><td>{pct(x["within_10pp"])}</td><td>{pct(x["full_state_accuracy"])}</td></tr>')
    h.append(f'</table></div><p class="muted">将 40 个 episode 等权平均时，MAE 为 {stats["episode_mean_mae_pp"]:.2f} pp，完整状态准确率为 {pct(stats["episode_mean_full_state_accuracy"])}。这与按帧加权的总体指标不同，反映长短片段的权重差异。</p>')
    h.append(image_block('figures/summary/progress_errors','图 1｜绝对进度误差的经验累积分布及逐视频 MAE。左图仅显示 0–40 pp 横轴范围，指标仍包含全部误差；S / R 分别表示简单 / 恢复数据。'))
    h.append('<h3>逐视频结果</h3><div class="scroll"><table><tr><th>视频</th><th>帧数</th><th>MAE / pp</th><th>RMSE / pp</th><th>当前阶段</th><th>完整状态</th><th>原始数据</th></tr>')
    for v in videos:
        x=v['metrics'];h.append(f'<tr><td><a href="#{v["id"]}">{label(v["group"])} / {v["video_id"]}</a></td><td>{x["n"]:,}</td><td>{x["mae_pp"]:.2f}</td><td>{x["rmse_pp"]:.2f}</td><td>{pct(x["current_stage_accuracy"])}</td><td>{pct(x["full_state_accuracy"])}</td><td><a href="{v["path"]}">JSON</a></td></tr>')
    h.append('</table></div></section><section id="transitions"><h2>4. 阶段状态与恢复片段</h2><p>当前阶段准确率只检查 nowid；完整状态准确率同时检查 nowid、finishid、restid。两项在本次结果中数值相同，这是观测结果，不是指标定义相同。Macro-F1 对四个当前阶段等权平均；集合 micro-F1 汇总各阶段集合的 TP、FP、FN。</p><table><tr><th>指标</th><th>结果</th></tr>')
    for name,value in [('当前阶段 Macro-F1',f'{m["macro_f1"]:.4f}'),('已完成阶段集合准确率',pct(m['finishid_accuracy'])),('已完成阶段集合 micro-F1',f'{m["finishid_micro_f1"]:.4f}'),('未完成阶段集合准确率',pct(m['restid_accuracy'])),('未完成阶段集合 micro-F1',f'{m["restid_micro_f1"]:.4f}')]:h.append(f'<tr><td>{name}</td><td>{value}</td></tr>')
    h.append('</table>');h.append(image_block('figures/summary/stage_confusion','图 2｜当前阶段混淆矩阵：行是真值，列是预测；每格同时显示帧数与按真值行归一化的比例。'))
    h.append('<table><tr><th>真实阶段</th><th>帧数</th><th>Precision</th><th>Recall</th><th>F1</th></tr>')
    for s in m['per_stage']:h.append(f'<tr><td>S{s["stage"]}</td><td>{s["support"]:,}</td><td>{pct(s["precision"])}</td><td>{pct(s["recall"])}</td><td>{s["f1"]:.4f}</td></tr>')
    h.append('</table>');h.append(image_block('figures/summary/transition_accuracy','图 3｜真实完整阶段状态变化点前后各 15 帧（±0.5 秒）与窗口外的准确率。重叠窗口取并集，每帧只计一次；图上 n 为帧数。'))
    h.append(f'<p>最长连续完整状态错误段位于 <a href="#{longest[1]["id"]}">{longest[1]["video_id"]} / ep{longest[1]["episode"]}</a>，t={longest[2]["start_seconds"]:.2f}–{longest[2]["end_seconds"]:.2f}s，共 {longest[0]} 帧，按每帧 1/30 秒折算约 {longest[0]/30:.2f}s。</p><h3>进度回退事件</h3><p>在同一 episode 内，以相邻真值 PT 下降超过 3 pp 定义回退事件，共 {len(events)} 次；不会将 episode 重置计为回退。下表的“相邻预测降幅”只比较事件前后一帧，提前或滞后下降需结合曲线与 ±2 秒窗口的最大单步降幅查看。后者也不代表整个恢复动作的总幅度。</p>')
    h.append(f'<p>事件当帧完整阶段状态已正确：{on_event}/{len(events)}；在事件后最多 2 秒、且真值状态未再次变化的窗口内，未匹配到新真值状态：{unresolved}/{len(events)}。阶段延迟只从事件当帧向后计时，已提前输出的情况记为 0，不据此声称没有提前预测。</p><div class="scroll"><table><tr><th>视频 / ep</th><th>t / s</th><th>真值降幅 / pp</th><th>相邻预测降幅 / pp</th><th>±2s 最大预测降幅 / pp</th><th>该降幅时刻 / s</th><th>事件后阶段匹配延迟 / s</th></tr>')
    for ev in events:
        lag='窗口内未匹配' if ev['first_post_event_state_match_seconds'] is None else f'{ev["first_post_event_state_match_seconds"]:.3f}'
        h.append(f'<tr><td><a href="#{ev["episode_id"]}">{ev["video_id"]} / ep{ev["episode"]}</a></td><td>{ev["time_seconds"]:.2f}</td><td>{ev["gt_drop_pp"]:.2f}</td><td>{ev["adjacent_predicted_drop_pp"]:.2f}</td><td>{ev["nearby_largest_drop_pp"]:.2f}</td><td>{ev["nearby_largest_drop_time_seconds"]:.2f}</td><td>{lag}</td></tr>')
    h.append('</table></div><p class="muted">本节描述误差、时间位置和混淆情况。仅凭这些预测结果无法判断模型内部是否利用了提示、为何发生错误，或定位视觉内容层面的原因；这些问题需要独立消融或查看原视频。</p></section>')
    h.append('<section id="gallery"><h2>5. 图片索引与逐 episode 结果</h2><p>每个视频有一张五段 episode 总览图；每个 episode 单独保存 PT 曲线、阶段偏差点线图，均提供 PNG 和 SVG。PT 蓝线是真值，橙线是预测；阶段图全部三个字段一致时为绿色，否则为红色。每个点都是实际保存的帧；全部保留，不降采样。紫色虚线为上述真值回退事件。</p><label>跳转到视频：<select onchange="jumpTo(this.value)"><option value="">选择视频</option>')
    for v in videos:h.append(f'<option value="{v["id"]}">{label(v["group"])} / {v["video_id"]}</option>')
    h.append('</select></label>')
    for v in videos:
        x=v['metrics'];expanded=' open' if v['id']=='recovery-file-010' else ''
        h.append(f'<details class="video" id="{v["id"]}"{expanded}><summary>{label(v["group"])} / {v["video_id"]}　<span class="tag">{x["n"]:,} 帧 · MAE {x["mae_pp"]:.2f} pp · 完整状态 {pct(x["full_state_accuracy"])}</span></summary>')
        h.append(image_block(f'figures/{v["group"]}/{v["video_id"]}/overview',f'{v["video_id"]} 五段 episode 总览：左列 PT，右列完整阶段状态一致性；每行使用各自 episode 的时间零点。'))
        for ep in v['episodes']:
            x=ep['metrics'];rows=ep['rows'];stem=ep['figure_dir'];expanded=' open' if ep['id']=='recovery-file-010-ep03' else ''
            h.append(f'<details class="episode" id="{ep["id"]}"{expanded}><summary>Episode {ep["episode"]}　<span class="tag">{x["n"]} 帧 · MAE {x["mae_pp"]:.2f} pp · 完整状态 {pct(x["full_state_accuracy"])}</span></summary><p class="muted">原视频帧 {rows[0]["frame_index"]}–{rows[-1]["frame_index"]}（0 起始）；原视频时间 {rows[0]["source_time_seconds"]:.2f}–{rows[-1]["source_time_seconds"]:.2f}s。以下以本 episode 首帧为 t=0。RMSE {x["rmse_pp"]:.2f} pp，Bias {x["bias_pp"]:+.2f} pp，P95 {x["p95_pp"]:.2f} pp。</p>')
            h.append(image_block(stem+'/pt_curve',f'{v["video_id"]} / ep{ep["episode"]} — 真值与预测 PT；曲线不平滑，连线仅连接相邻的已评估帧。'))
            svg=(ROOT/stem/'stage_agreement.svg').read_text().split('<svg',1)[1]
            # Inline SVG IDs share the HTML namespace, so prefix every ID and reference.
            for svg_id in set(re.findall(r'\bid="([^"]+)"',svg)):
                unique_id=ep['id']+'-'+svg_id
                svg=svg.replace(f'id="{svg_id}"',f'id="{unique_id}"')
                svg=svg.replace(f'href="#{svg_id}"',f'href="#{unique_id}"')
                svg=svg.replace(f'url(#{svg_id})',f'url(#{unique_id})')
            zoom_id=ep['id']+'-stage'
            h.append(f'<label class="caption">阶段点线放大：<input aria-label="阶段点线放大" type="range" min="100" max="600" value="100" oninput="zoomStage(this,\'{zoom_id}\')"></label><div class="scroll"><div class="stage-svg" id="{zoom_id}"><svg{svg}</div></div><p class="caption">{x["n"]} 个点全部保留；红点绘制在上层。<a href="{stem}/stage_agreement.png">阶段图 PNG</a> · <a href="{stem}/stage_agreement.svg">阶段图 SVG</a></p>')
            runs=ep['error_runs'];text='；'.join(f'{r["start_seconds"]:.2f}–{r["end_seconds"]:.2f}s（{r["frames"]} 帧）' for r in runs)
            h.append(f'<p class="caption">连续阶段错误段：{text or "无"}。</p></details>')
        h.append('</details>')
    h.append('</section><section id="reproduce"><h2>6. 文件组织与复现</h2><p>8 份原始 JSON 原样保留；报告与图片是从这些文件派生的。<a href="report_metrics.json">report_metrics.json</a> 保存本报告全部统计值、事件表和输入文件 SHA-256。</p><pre>report.html                          技术报告与全部图片索引\nreport_metrics.json                  重新计算的指标与输入校验摘要\nbuild_report.py                      本地重建脚本（不调用模型）\nfigures/summary/                     3 张汇总图（PNG + SVG）\nfigures/recovery/file-xxx/overview.*  每个恢复视频的五段总览\nfigures/simple/file-xxx/overview.*    每个简单视频的五段总览\nfigures/{group}/file-xxx/episode-xx/\n    pt_curve.png / .svg              真值 PT 与预测 PT\n    stage_agreement.png / .svg       阶段偏差点线图\n{原始数据集名称}/file-xxx.json         逐帧预测与真值</pre><p>在安装了 numpy、matplotlib 的 Python 环境中运行 <code>python build_report.py</code> 即可重建。脚本只读取结果 JSON，不依赖训练代码、远程环境或 GPU。图片引用使用相对路径；分享报告时请保留 <code>figures/</code> 同级目录。</p><details><summary>8 个数据文件的校验信息</summary><div class="scroll"><table><tr><th>文件</th><th>帧数</th><th>SHA-256</th></tr>')
    for item in stats['inputs']:h.append(f'<tr><td><a href="{item["path"]}">{escape(item["path"])}</a></td><td>{item["frames"]:,}</td><td><code>{item["sha256"]}</code></td></tr>')
    h.append('</table></div></details></section><p class="muted">报告中的数值结论由当前目录逐帧数据重算；配置性描述来自本对话已确认的评估协议。所有结果对应固定验证视频及固定任务，不外推为其他任务或独立测试集性能。</p></main>')
    h.append('''<script>function jumpTo(id){if(!id)return;const node=document.getElementById(id);if(!node)return;for(let p=node;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true;}node.scrollIntoView({behavior:'smooth',block:'start'});}document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',()=>jumpTo(a.getAttribute('href').slice(1))));function zoomStage(input,id){const node=document.getElementById(id);node.style.width=input.value+'%';node.querySelectorAll('g[id*="-PathCollection_"] defs path').forEach(p=>p.setAttribute('transform','scale('+(100/input.value)+')'));}if(location.hash)jumpTo(location.hash.slice(1));</script></body></html>''')
    (ROOT/'report.html').write_text(''.join(h))


if __name__=='__main__':main()
