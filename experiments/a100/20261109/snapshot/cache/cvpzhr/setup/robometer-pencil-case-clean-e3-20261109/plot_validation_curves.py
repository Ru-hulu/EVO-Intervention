"""Plot all held-out pencil-case-task curves using saved predictions and exact dense GT."""
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np

ROOT=Path(os.environ.get('ROBOMETER_CURVES_DIR', str(Path(__file__).parent/'validation_curves')))
FONT=next(str(p) for p in [Path('/System/Library/Fonts/STHeiti Medium.ttc'), Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')] if p.is_file())
fontManager.addfont(FONT)
font_name=FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.family':font_name,'font.size':11,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
BLUE,ORANGE,RED='#1678b5','#e87a24','#bd5353'


def main():
    bundle=json.loads((ROOT/'validation_bundle.json').read_text())
    grouped=defaultdict(list)
    for row in bundle['predictions']:grouped[row['episode_id']].append(row)
    saved={r['episode_id']:r for r in bundle['episode_metrics']}
    assert len(grouped)==len(bundle['episodes'])==7
    items=[]
    for source in sorted(bundle['episodes'],key=lambda x:x['episode_id']):
        text=source['ground_truth_csv_text']
        assert hashlib.sha256(text.encode()).hexdigest()==source['progress_sha256']
        dense=list(csv.DictReader(io.StringIO(text)))
        assert len(dense)==source['frames']
        assert [int(r['frame_index']) for r in dense]==list(range(len(dense)))
        gt=np.array([float(r['pt']) for r in dense])
        rows=sorted(grouped[source['episode_id']],key=lambda r:r['source_frame_index'])
        frame=np.array([r['source_frame_index'] for r in rows],dtype=int)
        assert np.all(np.diff(frame)>0)
        expected=list(range(0,len(gt),6))
        if expected[-1]!=len(gt)-1:expected.append(len(gt)-1)
        assert source['fps']==30 and frame.tolist()==expected
        pred=np.array([r['predicted_pt'] for r in rows])
        target=np.array([r['target_pt'] for r in rows])
        assert np.isfinite(pred).all() and np.allclose(gt[frame],target,atol=1e-6,rtol=0)
        error=pred-target;mae=float(np.abs(error).mean()*100)
        rmse=float(np.sqrt(np.mean(error**2))*100)
        assert abs(mae-saved[source['episode_id']]['mae_pp'])<1e-5
        rollback=np.flatnonzero(np.diff(gt)<-1e-6)+1
        item={'episode':source['episode_id'].split('/')[-1],'fps':source['fps'],
              'truth':gt,'frame':frame,'predicted':pred,'target':target,'mae_pp':mae,
              'rmse_pp':rmse,'rollback':rollback,'duration':len(gt)/source['fps'],'samples':len(rows)}
        items.append(item)
    count=sum(i['samples'] for i in items)
    assert count==bundle['metrics']['eval_samples']==2177
    overall=sum(i['mae_pp']*i['samples'] for i in items)/count
    assert abs(overall-bundle['metrics']['eval_mae_pp'])<1e-5

    def panel(ax,item):
        fps=item['fps'];t=np.arange(len(item['truth']))/fps
        ax.plot(t,item['truth'],color=BLUE,lw=2.1,label='真实进度',zorder=3)
        ax.step(np.r_[item['frame']/fps,item['duration']],np.r_[item['predicted'],item['predicted'][-1]],
                where='post',color=ORANGE,lw=1.5,label='Robometer 预测',alpha=.95,zorder=4)
        for frame in item['rollback']:
            ax.axvline(frame/fps,color=RED,ls='--',lw=1,alpha=.65,zorder=2)
        ax.set(xlim=(0,item['duration']),ylim=(-.035,1.045),xlabel='视频时间 / 秒',ylabel='任务进度')
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(.25))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.grid(alpha=.16,lw=.7)
        ax.set_title(f"{item['episode']}   |   MAE {item['mae_pp']:.2f} pp · RMSE {item['rmse_pp']:.2f} pp",
                     loc='left',fontsize=12,pad=11,color='#24364a')
        return ax

    fig,axes=plt.subplots(4,2,figsize=(14,13),dpi=130)
    for ax,item in zip(axes.flat,items):panel(ax,item)
    last=axes.flat[-1];last.axis('off')
    m=bundle['metrics']
    last.text(.04,.96,'验证结果',fontsize=18,fontweight='bold',va='top',color='#24364a')
    last.text(.04,.77,f'{len(items)} 条轨迹 · {count:,} 个观测',fontsize=13,va='top',color='#405267')
    last.text(.04,.59,f"整体 MAE    {m['eval_mae_pp']:.2f} 个百分点\n整体 RMSE  {m['eval_rmse_pp']:.2f} 个百分点\nPearson      {m['eval_pearson']:.4f}",
              fontsize=14,va='top',linespacing=1.35,color='#405267')
    last.text(.04,.0,'红色虚线：真值回退时刻；pp = 百分点。\n预测以 5 FPS 更新，帧间保持，未平滑。',
              fontsize=10,va='bottom',linespacing=1.3,color='#66788b')
    fig.suptitle('Robometer · 笔袋检查 · 验证集进度预测',fontsize=23,fontweight='bold',y=.986,color='#203348')
    fig.text(.5,.952,f"从原 1 epoch 检查点继续微调 3 epoch · 未添加人工增强 · {Path(bundle['checkpoint']).name}",
             ha='center',fontsize=12,color='#66788b')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.936),ncol=2,frameon=False,fontsize=13)
    fig.subplots_adjust(left=.07,right=.98,top=.89,bottom=.045,wspace=.23,hspace=.53)
    fig.savefig(ROOT/'validation_overview.png',dpi=130)
    fig.savefig(ROOT/'validation_overview.pdf')
    plt.close(fig)
    reports=[]
    for item in items:
        fig,ax=plt.subplots(figsize=(12,5.2),dpi=140)
        panel(ax,item);ax.legend(loc='upper left',framealpha=.94,ncol=2)
        fig.tight_layout(pad=1.5)
        file=ROOT/(item['episode']+'.prediction_curve.png')
        fig.savefig(file);plt.close(fig)
        errors=np.abs(item['predicted']-item['target'])*100;imax=int(errors.argmax())
        r={k:v for k,v in item.items() if k not in ['truth','frame','predicted','target','rollback']}
        r.update(rollback_times_seconds=(item['rollback']/item['fps']).tolist(),
                 max_error_pp=float(errors[imax]),max_error_time_seconds=float(item['frame'][imax]/item['fps']),
                 final_true_pt=float(item['target'][-1]),final_predicted_pt=float(item['predicted'][-1]),
                 curve_png=str(file))
        reports.append(r)
    report={'checkpoint':bundle['checkpoint'],'labels_match_validation_targets':True,'all_expected_frames_present':True,
            'prediction_display':'5 FPS, held between observations, no smoothing','metrics':m,'episodes':reports}
    (ROOT/'plot_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
