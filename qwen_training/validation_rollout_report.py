"""Audit a complete validation rollout and render all 40 pairs of figures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import html
import io
import json
from pathlib import Path

from qwen_training.closed_loop_report import audit, run as episode_report
from qwen_training.closed_loop_rollout import save_json
from qwen_training.evaluation_metrics import measure

EXPECTED_CHECKPOINT='dd0388c009460fed78f6bb3d6db95d57d5bb72c930dfc73b39d081344d42d543'


def render_episode(directory):
    with redirect_stdout(io.StringIO()):episode_report(directory)
    return str(directory)


def run(root):
    suite=json.loads((root/'suite.json').read_text())
    result=json.loads((root/'result.json').read_text())
    if result['status']!='complete' or result['samples']!=suite['samples']:
        raise ValueError('Full evaluation is incomplete')
    if suite['checkpoint']['checkpoint_sha256']!=EXPECTED_CHECKPOINT:
        raise ValueError('Unexpected checkpoint')
    if len(suite['episodes'])!=40 or len(suite['validation_videos'])!=8 or suite['samples']!=23604:
        raise ValueError('Unexpected validation coverage')
    if len({(s['dataset'],s['video_id'],s['task_index'],s['episode']) for s in suite['episodes']})!=40:
        raise ValueError('Duplicate episode')
    rank_results=[]
    for rank,assignment in enumerate(suite['assignments']):
        record=json.loads((root/f'complete-rank{rank:02d}.json').read_text())
        if record['status']!='complete' or record['samples']!=sum(e['samples'] for e in assignment):
            raise ValueError(f'Incomplete rank {rank}')
        rank_results.append(record)
    controls=json.loads((root/'benchmark/sequence.json').read_text())
    drafts=json.loads((root/'benchmark/draft.json').read_text())
    if controls['status']!='passed' or drafts['status']!='complete':
        raise ValueError('Inference controls did not pass')
    if controls['checkpoint_sha256']!=EXPECTED_CHECKPOINT or len(controls['records'])!=controls['samples']:
        raise ValueError('Control provenance mismatch')
    for row in controls['records']:
        if row['ordinary_response']!=row['verified_response'] or abs(row['ordinary_progress']-row['verified_progress'])>1e-4:
            raise ValueError('Paired control outputs differ')
        if row['ordinary_input_state']!=row['verified_input_state']:
            raise ValueError('Paired control feedback differs')
    for case in drafts['results']:
        if case['reference_text_exact']!=case['batch_size']:
            raise ValueError('Batched decoding control failed')
        if case['mode']=='wrong_draft' and any(case['draft_accepted']):
            raise ValueError('Wrong draft was accepted')
    all_rows=[];episodes=[];sample_ids=set()
    for spec in suite['episodes']:
        directory=root/spec['relative_dir']
        manifest,completion,rows,truth=audit(directory)
        if manifest['checkpoint_sha256']!=EXPECTED_CHECKPOINT or manifest['prediction_step_frames']!=1:
            raise ValueError('Episode checkpoint or frequency mismatch')
        decode=json.loads((directory/'decode_audit.json').read_text())
        if decode['status']!='passed' or decode['frames']!=len(rows) or len(decode['checked_steps'])!=3:
            raise ValueError('Pixel cache audit missing')
        if len(rows)!=spec['samples'] or any(r['sample_id'] in sample_ids for r in rows):
            raise ValueError('Incomplete or duplicate episode rows')
        sample_ids.update(r['sample_id'] for r in rows)
        all_rows.extend(rows)
        episodes.append(dict(spec,metrics=measure(rows)))
    if len(all_rows)!=23604 or len(sample_ids)!=23604:
        raise ValueError('Frame count mismatch')
    # Independent figures/report generation is CPU-only and safe to parallelize.
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(render_episode,[root/s['relative_dir'] for s in suite['episodes']]))
    metrics=dict(overall=measure(all_rows),by_video={},by_dataset={},episodes=episodes,
        samples=len(all_rows),episode_count=len(episodes),video_count=len(suite['validation_videos']),
        draft_accepted=sum(r['draft_accepted'] for r in all_rows),
        initial_ground_truth_inputs=40,later_ground_truth_inputs=0,
        prediction_hz=30,checkpoint_sha256=EXPECTED_CHECKPOINT,rank_results=rank_results)
    for dataset,video in suite['validation_videos']:
        metrics['by_video'][dataset+'/'+video]=measure([r for r in all_rows if r['dataset']==dataset and r['video_id']==video])
    for dataset in sorted({r['dataset'] for r in all_rows}):
        metrics['by_dataset'][dataset]=measure([r for r in all_rows if r['dataset']==dataset])
    save_json(root/'metrics.json',metrics)
    save_json(root/'audit.json',dict(status='passed',episodes=40,videos=8,unique_frames=23604,
        feedback_chains='all passed',pixel_cache_checks=120,checkpoint_sha256=EXPECTED_CHECKPOINT,
        benchmark_sequence_samples=controls['samples'],figures=80))
    e=html.escape;m=metrics['overall']
    title='验证集全量 · 30Hz 闭环预测'
    out='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    out+=f'<title>{title}</title><style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui,sans-serif;color:#243847}}h1{{font-size:34px}}h2{{margin-top:32px}}.lead{{background:#edf5fa;border-radius:14px;padding:20px}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}}.card{{flex:1;min-width:190px;padding:18px;border:1px solid #d5e0e8;border-radius:12px}}.number{{display:block;font-size:29px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dbe2e8}}details{{border:1px solid #d8e1e9;border-radius:10px;padding:16px;margin:18px 0}}summary{{font-weight:650;cursor:pointer}}a{{color:#17639a}}svg{{width:100%;height:auto}}.scroll{{overflow-x:auto}}.figures{{margin:12px 0}}code{{overflow-wrap:anywhere;font-size:12px}}.muted{{color:#627786}}</style>'
    out+=f'<h1>{title}</h1><div class="lead"><b>Qwen3.5-2B · epoch-04</b><br>8 个验证视频，40 个 episode，23,604 个标注帧全部完成。每个 episode 独立从首帧开始，首次使用阶段真值；之后每帧反馈上一帧模型预测的阶段状态。</div>'
    out+='<div class="cards">'
    for label,value in [('进度 MAE',f'{m["progress_mae_pp"]:.2f} pp'),('进度 RMSE',f'{m["progress_rmse_pp"]:.2f} pp'),('当前阶段准确率',f'{m["current_stage_accuracy"]:.2%}'),('完整阶段状态准确率',f'{m["full_state_accuracy"]:.2%}')]:
        out+=f'<div class="card">{label}<span class="number">{value}</span></div>'
    out+='</div><p>每个预测时刻 t 输入 t−5 至 t 秒、间隔 1 秒的六张历史图像，三个视角相同；不足本 episode 历史的部分补黑色空白帧和无效掩码。相邻预测推进一帧（1/30 秒）。评估覆盖所有标注 episode 帧，视频开头与末尾未标注部分不计入。</p>'
    out+=f'<p>进度带符号偏差 {m["progress_bias_pp"]:+.2f} pp，绝对误差 P95 {m["progress_p95_ae_pp"]:.2f} pp；误差不超过 5 / 10 pp 的比例为 {m["progress_within_5pp"]:.2%} / {m["progress_within_10pp"]:.2%}。阶段 Macro-F1 为 {m["current_stage_macro_f1"]:.4f}，已完成 / 未完成阶段集合准确率为 {m["finishid_exact_accuracy"]:.2%} / {m["restid_exact_accuracy"]:.2%}。</p>'
    out+='<h2>各视频结果</h2><div class="scroll"><table><tr><th>数据集 / 视频</th><th>帧数</th><th>PT MAE / pp</th><th>PT RMSE / pp</th><th>当前阶段</th><th>完整状态</th></tr>'
    for key,v in metrics['by_video'].items():
        label=key.replace('piperx-bolt-intentional-drop-reinsert-22ep-raw','恢复').replace('piperx_insert_copper_screw','简单')
        out+=f'<tr><td>{e(label)}</td><td>{v["generated_responses"]:,}</td><td>{v["progress_mae_pp"]:.2f}</td><td>{v["progress_rmse_pp"]:.2f}</td><td>{v["current_stage_accuracy"]:.2%}</td><td>{v["full_state_accuracy"]:.2%}</td></tr>'
    out+='</table></div><script>function zoomStage(slider,id){const e=document.getElementById(id);e.style.width=slider.value+"%";e.querySelectorAll("g[id^=PathCollection_] defs path").forEach(p=>p.setAttribute("transform","scale("+(100/slider.value)+")"));}</script><h2>逐 episode 的两张图</h2><p>展开对应视频查看 5 个 episode。蓝线是真值 PT，橙线是逐帧预测；阶段点线全部字段相同为绿色，否则为红色，红点绘制在上层。所有点均保留，可横向放大查看。点击详细报告可查看逐帧原始输出、回退响应与错误案例。</p>'
    for dataset,video in suite['validation_videos']:
        label=('恢复' if 'intentional' in dataset else '简单')+' / '+video
        expanded=' open' if 'intentional' in dataset and video=='file-010' else ''
        out+=f'<details{expanded}><summary>{e(label)}</summary>'
        for ep in (s for s in episodes if s['dataset']==dataset and s['video_id']==video):
            directory=root/ep['relative_dir'];v=ep['metrics'];slug=f'{dataset}-{video}-{ep["episode"]}'
            out+=f'<h3>Episode {ep["episode"]} · {ep["samples"]} 帧</h3><p>PT MAE {v["progress_mae_pp"]:.2f} pp；完整状态准确率 {v["full_state_accuracy"]:.2%}。<a href="{e(ep["relative_dir"])}/report.html">打开详细报告</a></p>'
            out+='<div class="figures"><svg'+(directory/'pt_curve.svg').read_text().split('<svg',1)[1]+'</div>'
            out+=f'<label>阶段点线横向放大 <input type="range" min="100" max="600" value="100" oninput="zoomStage(this,\'{slug}\')"></label><div class="scroll"><div id="{slug}"><svg'+(directory/'stage_agreement.svg').read_text().split('<svg',1)[1]+'</div></div>'
        out+='</details>'
    max_seconds=max(r['elapsed_seconds'] for r in rank_results)
    out+='<h2>运行方式与审计</h2>'
    out+=f'<p>8 张 A100 各加载一个模型，批量处理独立 episode；每个 episode 内保留严格顺序反馈。视频只解码一次并缓存；每个 episode 抽查首、中、末 3 个窗口与原读取路径逐像素一致，共 120 次检查。最长 GPU 推理阶段 {max_seconds/60:.1f} 分钟（不含准备与报告），全量帧数除以该时长约 {23604/max_seconds:.2f} 帧/秒。30Hz 是视频时间上的预测密度，不代表在线实时速度。</p>'
    out+=f'<p>文字加速使用上一帧模型输出作为草稿，当前帧进行因果 token 校验，含结束符；逐 token 都为当前模型的贪心选择且最小 logit 差值达到 1.0 才接受，否则普通生成。共接受 {metrics["draft_accepted"]:,}/{len(all_rows):,}（{metrics["draft_accepted"]/len(all_rows):.2%}）。当前帧标签不参与推理，PT 使用独立的提示输入回归。24 次跨回退的连续对照中文字与 PT 一致，另有批量、错误草稿、变化批量大小控制；不代表对全部帧再次运行了普通解码。BF16 批量计算可能有数值差异。</p>'
    out+='<p>所有 40 条反馈链、23,604 个唯一帧、8 个工作进程完成标记、源标签和预测文件哈希均通过核验。本次是既有验证集上的闭环评估，不是独立测试集；窗口高度重叠，任务意图固定。指标按实际帧数加权，不能将相邻帧当成独立样本，也不能与此前使用真值上一状态的结果直接等同。</p>'
    out+=f'<p class="muted">检查点 SHA-256：<code>{EXPECTED_CHECKPOINT}</code><br>汇总数据：metrics.json；审计：audit.json；逐帧预测与两张 PNG/SVG 图位于各 episode 目录；优化对照位于 benchmark/。</p></html>'
    (root/'report.html').write_text(out)
    print(json.dumps(dict(status='complete',samples=len(all_rows),reports=41,figures=80,metrics=m),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    run(p.parse_args().directory)
