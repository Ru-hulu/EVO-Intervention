"""Summarize and plot the fixed-observation history intervention."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from history_probe import ROOT, dump


def main():
    rows=[json.loads(x) for x in (ROOT/'results.jsonl').read_text().splitlines()]
    status=json.loads((ROOT/'status.json').read_text())
    assert status['phase']=='completed' and len(rows)==192
    prefix=json.loads((ROOT/'prefix-results.json').read_text())['records']
    assert len(prefix)==12
    def select(**criteria):
        return [r for r in rows if all(r[k]==v for k,v in criteria.items())]
    groups=[]
    for episode in [55,54,84]:
        for direction in ['failure_view','success_view']:
            selected=select(failure_episode=episode,direction=direction,failure_has_occurred=True)
            originals={r['pair_index']:r for r in selected if r['condition']=='original'}
            swaps=[r for r in selected if r['condition']=='swapped']
            diffs=np.array([r['predicted_pt']-originals[r['pair_index']]['predicted_pt'] for r in swaps])*100
            groups.append(dict(failure_episode=episode,direction=direction,n=len(diffs),
                mean_swap_delta_pp=float(diffs.mean()),mean_absolute_swap_delta_pp=float(abs(diffs).mean()),
                swaps_over_5pp=int((abs(diffs)>5).sum())))
    empty=select(condition='empty')
    summary=dict(status=status,groups=groups,empty_history_cases=len(empty),
        empty_history_below_1pct=sum(r['predicted_pt']<.01 for r in empty),
        focus900=[{k:v for k,v in r.items() if k not in ['probabilities','current_view_sha256']} for r in select(failure_episode=55,failure_frame=900)],
        prefix=[{k:v for k,v in r.items() if k!='probabilities'} for r in prefix])
    dump(ROOT/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    styles={'original':('Original history','#182a39'), 'swapped':('Swapped history','#d97326'), 'empty':('No history','#718096')}
    for ax,direction,title in [(axes[0,0],'failure_view','Failure images held fixed'),(axes[0,1],'success_view','Matched success images held fixed')]:
        for condition,(label,color) in styles.items():
            rr=sorted(select(failure_episode=55,direction=direction,condition=condition),key=lambda r:r['failure_frame'])
            ax.plot([r['failure_frame'] for r in rr],[100*r['predicted_pt'] for r in rr],'o-',label=label,color=color)
        ax.axvline(626,color='#bd4f44',ls=':',label='Failure at frame 626')
        ax.set(xlabel='Failure-trajectory anchor frame (matched pairs)',ylabel='Predicted progress (%)',ylim=(-3,103),title=title)
        ax.legend(fontsize=9);ax.grid(alpha=.15)
    for condition,(label,color) in styles.items():
        r=select(failure_episode=55,failure_frame=900,direction='failure_view',condition=condition)[0]
        axes[1,0].plot(range(101),np.array(r['probabilities'])*100,label=label,color=color)
    axes[1,0].set(xlabel='Progress bin',ylabel='Probability (%)',title='Identical failure frame 900: bin distributions',xlim=(-1,101))
    axes[1,0].legend(fontsize=9);axes[1,0].grid(alpha=.15)
    for condition,label,color in [('raw_prefix','Earlier history','#127e79'),('same_length_prefix','Earlier history, fixed token length','#9774b5')]:
        r=[x for x in prefix if x['condition']==condition]
        axes[1,1].plot([x['history_sample_frame'] for x in r],[x['predicted_pt']*100 for x in r],'o-',label=label,color=color)
    axes[1,1].axvline(626,color='#bd4f44',ls=':')
    axes[1,1].set(xlabel='History snapshot frame (current image always 900)',ylabel='Predicted progress (%)',ylim=(-3,103),title='Same scene: replace only the history prefix')
    axes[1,1].legend(fontsize=9);axes[1,1].grid(alpha=.15)
    fig.suptitle('FOREST step13224 | Fixed current observations, intervened history',fontsize=16)
    fig.savefig(ROOT/'history-intervention.png',dpi=170)
    lines=['# 固定当前画面，替换历史：FOREST step13224','',
        '3 对同颜色排列/同任务顺序的成功失败轨迹，16 对观测、双向替换、每方向 6 条件，共 192 次前向；另做 12 次同轨迹历史前缀检查。全部来自 val。',
        '当前三视角像素和指令逐项核对一致；同长度控制仅修改 history_tokens，其他输入张量逐位相同。重复原输入的最大误差为 0。',
        '所有对比统一 batch=1；与旧 viewer 的 batch=8 预测可能存在浮点差异，结论使用本实验内基线。','',
        '## 第 900 帧（原任务已失败，真值 0）','',
        '| 当前画面 | 历史 | 预测 progress | P(bin0) | P(progress ≥80%) |','|---|---|---:|---:|---:|']
    for direction in ['failure_view','success_view']:
        for condition in ['original','swapped','empty','swapped_same_length']:
            r=select(failure_episode=55,failure_frame=900,direction=direction,condition=condition)[0]
            lines.append(f"| {direction} | {condition} | {100*r['predicted_pt']:.4f}% | {100*r['p_zero']:.4f}% | {100*r['p_high']:.4f}% |")
    lines += ['', '## 同一失败轨迹的早期历史，当前画面始终为第 900 帧','', '| 历史采样帧 | 保留原长度 | 固定为当前历史长度 |','|---|---:|---:|']
    for frame in sorted({r['history_sample_frame'] for r in prefix}):
        group={r['condition']:r for r in prefix if r['history_sample_frame']==frame}
        lines.append(f"| {frame} | {group['raw_prefix']['predicted_pt']*100:.4f}% | {group['same_length_prefix']['predicted_pt']*100:.4f}% |")
    lines += ['', '## 解释边界','',
        f"清空历史后，{summary['empty_history_below_1pct']}/{summary['empty_history_cases']} 个当前观测预测低于 1%，其中包括成功画面，因此不能把清空后的归零当作识别失败。",
        '替换实验可以说明模型对历史输入的敏感性，不能给拼接输入赋予经过仿真验证的真值。成功失败的场景 seed 和机械臂动作不同；按状态几何最近邻匹配，不保证完全相同的物理状态。',
        '固定 token 长度的历史使用自适应均值重采样，仅作辅助控制，属于分布外输入。清空历史、零历史亦为诊断消融。',
        '同轨迹前缀实验省略了旧快照至当前帧之间的观测，不能视为正常在线评估。该实验尚不能区分：失败信息未编码、压缩丢失、读取失败或视觉条件覆盖历史。',
        'episode55 为已知错误样例，其他配对只是小规模对照，不能据此估计整体失败检测性能。','',
        '原始数据：protocol.json、results.jsonl、prefix-results.json、summary.json；原权重/标签和正在运行的训练均未修改。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['focus900','prefix']},ensure_ascii=False))


if __name__=='__main__':main()
