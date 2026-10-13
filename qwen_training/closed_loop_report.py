"""Audit a finished closed-loop rollout and render two figures, without inference."""

import argparse
import html
import json
from pathlib import Path

from qwen_training.closed_loop_rollout import STATE_KEYS, feedback_state, history_window, sha256
from qwen_training.evaluation_metrics import measure, parse_response
from qwen_training.prompts import build_messages, describe_state


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["prediction_hz"] != manifest["source_fps"] / manifest["prediction_step_frames"]:
        raise ValueError("Prediction frequency and frame step disagree")
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Rollout has not completed")
    for file, key in [("predictions.jsonl", "predictions_sha256"), ("ground_truth.json", "ground_truth_sha256")]:
        if sha256(directory / file) != result[key]:
            raise ValueError(f"Artifact hash mismatch: {file}")
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    truth = json.loads((directory / "ground_truth.json").read_text())
    frames = list(range(manifest["frame_start"], manifest["frame_end"] + 1, manifest["prediction_step_frames"]))
    if len(rows) != manifest["samples"] or len(rows) != result["samples"] or [r["frame_index"] for r in rows] != frames:
        raise ValueError("Missing, duplicate, or reordered prediction frames")
    if [r["frame_index"] for r in truth] != list(range(manifest["frame_start"], manifest["frame_end"]+1)):
        raise ValueError("Dense ground truth is incomplete")
    dense = {r["frame_index"]: r for r in truth}
    previous = manifest["initial_state"]
    for step, row in enumerate(rows):
        if row["step"] != step or row["input_state"] != previous:
            raise ValueError(f"Feedback chain is broken at step {step}")
        if row["input_state_source"] != ("initial_ground_truth" if step == 0 else "previous_prediction"):
            raise ValueError("Unexpected ground-truth input after initialization")
        if manifest.get('generation') == 'greedy_with_previous_prediction_draft_verification':
            if row['draft_source'] != ('previous_prediction' if step else None):
                raise ValueError('Invalid draft provenance')
            if row['draft_accepted'] and (step == 0 or row['draft_minimum_margin'] < manifest['minimum_draft_margin']):
                raise ValueError('Unverified draft acceptance')
        if row["prompt"] != build_messages([previous], state_drop_probability=0.0)[0]:
            raise ValueError("Recorded prompt differs from generated-state feedback")
        if row["frame_indices"] != history_window(row["frame_index"], manifest["frame_start"]):
            raise ValueError("Incorrect causal history window")
        expected = [i is not None for i in row["frame_indices"]]
        if row["frame_mask"] != [expected]*3:
            raise ValueError("Incorrect padding mask")
        if abs(row["time_seconds"] - (row["frame_index"] - manifest["frame_start"])/30) > 1e-8:
            raise ValueError("Incorrect prediction timestamp")
        target = dense[row["frame_index"]]
        if any(row["target"][k] != target[k] for k in (*STATE_KEYS, "progress", "intent")):
            raise ValueError("Prediction target does not match dense ground truth")
        previous = feedback_state(row["response"])
    return manifest, result, rows, truth


def error_runs(rows, matches):
    """Group adjacent incorrect observation times without inventing error durations."""
    result = []
    for row, correct in zip(rows, matches):
        if not correct:
            if result and result[-1]["last_step"] == row["step"] - 1:
                result[-1].update(end=row["time_seconds"], last_step=row["step"], points=result[-1]["points"]+1)
            else:
                result.append(dict(start=row["time_seconds"], end=row["time_seconds"], last_step=row["step"], points=1))
    return result


def run(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    manifest, result, rows, truth = audit(directory)
    hz = manifest["prediction_hz"]
    interval = manifest["prediction_step_frames"] / manifest["source_fps"]
    parsed = [parse_response(r["response"]) for r in rows]
    matches = [all(p[k] == r["target"][k] for k in STATE_KEYS) for p, r in zip(parsed, rows)]
    metrics = measure(rows)
    rollbacks = []
    for left, right in zip(truth, truth[1:]):
        if left["progress"] - right["progress"] > 0.03:
            before = [r for r in rows if r["frame_index"] < right["frame_index"]][-1]
            after = next(r for r in rows if r["frame_index"] >= right["frame_index"])
            earlier = [r for r in rows if r["frame_index"] < before["frame_index"]]
            nearby_steps = [(a,b) for a,b in zip(rows,rows[1:]) if abs(b["time_seconds"]-right["time_seconds"])<=2]
            biggest_before,biggest_after = max(nearby_steps, key=lambda pair: pair[0]["predicted_progress"]-pair[1]["predicted_progress"])
            matching = [r for r,p in zip(rows,parsed) if abs(r["time_seconds"]-right["time_seconds"])<=2
                        and all(p[k]==right[k] for k in STATE_KEYS)]
            rollbacks.append(dict(time_seconds=right["time_seconds"], frame_index=right["frame_index"],
                                  instantaneous_gt_drop_pp=100*(left["progress"]-right["progress"]),
                                  before_time=before["time_seconds"], after_time=after["time_seconds"],
                                  sampled_gt_drop_pp=100*(before["target"]["progress"]-after["target"]["progress"]),
                                  predicted_drop_pp=100*(before["predicted_progress"]-after["predicted_progress"]),
                                  nearby_largest_predicted_drop_time=biggest_after["time_seconds"],
                                  nearby_largest_predicted_drop_pp=100*(biggest_before["predicted_progress"]-biggest_after["predicted_progress"]),
                                  nearby_first_new_state_time=matching[0]["time_seconds"] if matching else None,
                                  nearby_first_new_state_offset_seconds=matching[0]["time_seconds"]-right["time_seconds"] if matching else None,
                                  preceding_time=earlier[-1]["time_seconds"] if earlier else None,
                                  preceding_predicted_drop_pp=100*(earlier[-1]["predicted_progress"]-before["predicted_progress"]) if earlier else None))
    metrics.update(samples=len(rows), prediction_hz=hz, prediction_interval_seconds=interval,
                   full_state_correct=sum(matches), full_state_wrong=len(rows)-sum(matches),
                   stage_error_runs=error_runs(rows, matches),
                   average_inference_seconds=sum(r["seconds"] for r in rows)/len(rows),
                   elapsed_seconds=result["elapsed_seconds"], rollbacks=rollbacks,
                   audit={"feedback_chain":"passed", "initial_ground_truth_inputs":1,
                          "later_ground_truth_inputs":0,"causal_windows":"passed", "artifact_hashes":"passed"})
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n")
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.fonttype":"none", "axes.titleweight":"bold"})
    times = [r["time_seconds"] for r in rows]
    max_time = truth[-1]["time_seconds"]
    title = f'{manifest["video_id"]} / episode {manifest["episode"]} · epoch {manifest["checkpoint_epoch"]} · {hz:g} Hz closed loop'
    figures = []
    fig, ax = plt.subplots(figsize=(12,4.7), constrained_layout=True)
    ax.plot([r["time_seconds"] for r in truth], [100*r["progress"] for r in truth], color="#183c68", lw=2, label="Ground truth (30 FPS)")
    ax.plot(times, [100*r["predicted_progress"] for r in rows], color="#ec8933", marker="o", markersize=2 if len(rows)>100 else 4, lw=1.3, label=f"Prediction ({hz:g} Hz, model-state feedback)")
    for i, event in enumerate(rollbacks):
        ax.axvline(event["time_seconds"],color="#ab567c",ls="--",lw=1,alpha=.6)
        ax.text(event["time_seconds"]+.35,103,f'Rollback {i+1}',fontsize=9,color="#863c5e")
    ax.set(xlim=(-.5,max_time+.5),ylim=(-3,109),xlabel="Time from episode start (s)",ylabel="Task progress PT (%)",title=title)
    ax.grid(alpha=.16)
    ax.legend(loc="upper left",frameon=False,fontsize=9)
    fig.savefig(directory/"pt_curve.png",dpi=200)
    fig.savefig(directory/"pt_curve.svg")
    plt.close(fig)
    figures.append((directory/"pt_curve.svg").read_text().split("<svg",1)[1])

    fig, ax = plt.subplots(figsize=(12,2.7), constrained_layout=True)
    ax.axhline(0,color="#d9dfe5",lw=.35,zorder=1)
    # Plot every observation, with errors last so dense correct points cannot hide them.
    point_size=2 if hz>=30 else (8 if len(rows)>100 else 46)
    for good, color in [(True,'#168448'),(False,'#d53f48')]:
        selected=[t for t,m in zip(times,matches) if m==good]
        ax.scatter(selected,[0]*len(selected),c=color,s=point_size,zorder=3 if good else 4,edgecolors='none')
    for event in rollbacks:
        ax.axvline(event["time_seconds"],color="#ab567c",ls="--",lw=1,alpha=.6)
    ax.set(xlim=(-.5,max_time+.5),ylim=(-.45,.65),yticks=[],xlabel="Time from episode start (s)",
           title=f'Full stage state agreement · {sum(matches)}/{len(rows)} correct ({sum(matches)/len(rows):.1%})')
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Line2D([],[],marker="o",ls="",color="#168448",label="All stage fields match"),
                       Line2D([],[],marker="o",ls="",color="#d53f48",label="At least one stage field differs")],
              loc="upper center",ncol=2,frameon=False,fontsize=9)
    fig.savefig(directory/"stage_agreement.png",dpi=200)
    fig.savefig(directory/"stage_agreement.svg")
    plt.close(fig)
    figures.append((directory/"stage_agreement.svg").read_text().split("<svg",1)[1])

    e=html.escape
    h='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    h+='<title>恢复 Episode 闭环预测</title><style>body{max-width:1120px;margin:32px auto;padding:0 24px;color:#203440;font:16px/1.75 system-ui,sans-serif}h1{font-size:29px}h2{margin-top:32px;font-size:23px}.lead{background:#edf5fa;padding:18px 24px;border-radius:12px}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{flex:1;min-width:180px;border:1px solid #d5e2eb;border-radius:10px;padding:14px}.number{display:block;font-size:26px;font-weight:700}svg{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #d9e0e8;text-align:left}.green{color:#168448}.red{color:#bd2836}code{font-size:12px;overflow-wrap:anywhere}.scroll{overflow-x:auto}details{margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;background:#f5f7f9;padding:12px}</style>'
    h+=f'<h1>验证 Episode：{hz:g}Hz 阶段状态闭环预测</h1>'
    h+=f'<div class="lead"><b>{e(manifest["video_id"])} · Episode {manifest["episode"]} · Qwen3.5-2B / epoch-04</b><br>原视频帧 {manifest["frame_start"]}–{manifest["frame_end"]}（帧号从 0 开始，{manifest["frame_start"]/30:.2f}–{manifest["frame_end"]/30:.2f} 秒）；将此片段首帧定义为 t=0。共 {manifest["duration_seconds"]:.1f} 秒，{len(rows)} 次顺序预测。</div>'
    h+=f'<p>每 {interval:g} 秒预测一次（{hz:g}Hz，原 30FPS 视频每次推进 {manifest["prediction_step_frames"]} 帧），每个视角输入 t−5、t−4、t−3、t−2、t−1、t 秒的六张图像，首末相隔 5 秒；早于片段起点的图像以黑色空白帧和无效掩码补齐。三个视角与训练配置一致。首次提示使用片段首帧的阶段真值；此后只使用上一次生成的当前阶段、已完成阶段和未完成阶段，按名称解析后传入原有四阶段提示模板。PT 输出不作为下一次输入。</p>'
    h+='<div class="cards">'
    for label,value in [('进度平均绝对误差',f'{metrics["progress_mae_pp"]:.2f} 个百分点'),('进度 RMSE',f'{metrics["progress_rmse_pp"]:.2f} 个百分点'),('当前阶段准确率',f'{metrics["current_stage_accuracy"]:.2%}'),('完整阶段状态准确率',f'{metrics["full_state_accuracy"]:.2%}')]:
        h+=f'<div class="card">{label}<span class="number">{value}</span></div>'
    wrong_times='、'.join((f'{run["start"]:g}–{run["end"]:g}s（{run["points"]} 点）' if run['points']>1 else f'{run["start"]:g}s') for run in metrics['stage_error_runs'])
    h+=f'</div><p>本片段中，PT 预测平均带符号误差为 {metrics["progress_bias_pp"]:.2f} 个百分点（负值表示整体偏低）。阶段状态偏差的连续采样点分组：{wrong_times or "无"}。范围只概括实际采样点，不代表已知其间每帧的预测。单个恢复片段的结果不能与全验证集平均指标直接作性能升降比较。</p><h2>1. 真值 PT 与预测 PT</h2><svg'+figures[0]
    h+=f'<p>蓝线为原始 30FPS 逐帧真值，橙色圆点为实际推理输出；橙色连线仅便于阅读，不代表在两点之间又进行了推理。误差仅在 {len(rows)} 个预测时刻计算，不做平滑、不强制单调、不校正回退。紫色虚线标注真值回退。</p>'
    h+='<script>function zoomStage(slider,id){const e=document.getElementById(id);e.style.width=slider.value+"%";e.querySelectorAll("g[id^=PathCollection_] defs path").forEach(p=>p.setAttribute("transform","scale("+(100/slider.value)+")"));}</script>'
    h+='<h2>2. 阶段信息偏差</h2><label>横向放大 <input type="range" min="100" max="600" value="100" oninput="zoomStage(this,\'stage-zoom\')"></label><div class="scroll"><div id="stage-zoom"><svg'+figures[1]+'</div></div>'
    h+=f'<p>每个点对应一次预测（间隔 {interval:g} 秒），全部 {len(rows)} 个点都绘出，没有降采样。当前阶段、已完成阶段集合、未完成阶段集合全部一致时为绿色，否则为红色；集合忽略排列顺序，任务意图不纳入此项。绿色 {sum(matches)} 个，红色 {len(rows)-sum(matches)} 个。</p>'
    h+='<h2>回退时刻的响应</h2><table><tr><th>真值回退时刻</th><th>逐帧真值降幅</th><th>相邻预测时刻</th><th>这两个时刻的真值降幅</th><th>预测降幅</th></tr>'
    for event in rollbacks:
        h+=f'<tr><td>{event["time_seconds"]:g}s</td><td>{event["instantaneous_gt_drop_pp"]:.2f} pp</td><td>{event["before_time"]:g} → {event["after_time"]:g}s</td><td>{event["sampled_gt_drop_pp"]:.2f} pp</td><td>{event["predicted_drop_pp"]:.2f} pp</td></tr>'
    h+=f'</table><p>降幅为“前值 − 后值”，正数表示回退、负数表示上升。{hz:g}Hz 采样无法分辨 {interval:g} 秒采样间隔内的精确检测延迟。</p>'
    h+='<p>下面在每次标注回退的前后 2 秒内检查响应，以免只看紧邻回退的两点而漏掉提前或滞后的变化。PT 最大单步下降不等同于整个回退幅度；阶段偏移按首次生成回退后的完整状态计算，负数表示早于标注、正数表示晚于标注。</p><table><tr><th>真值回退时刻</th><th>附近 PT 最大单步下降</th><th>首次生成回退后状态</th><th>阶段相对标注偏移</th></tr>'
    for event in rollbacks:
        state_time=f'{event["nearby_first_new_state_time"]:g}s' if event['nearby_first_new_state_time'] is not None else '窗口内未出现'
        offset=f'{event["nearby_first_new_state_offset_seconds"]:+.1f}s' if event['nearby_first_new_state_offset_seconds'] is not None else '—'
        h+=f'<tr><td>{event["time_seconds"]:g}s</td><td>{event["nearby_largest_predicted_drop_time"]:g}s / {event["nearby_largest_predicted_drop_pp"]:.2f} pp</td><td>{state_time}</td><td>{offset}</td></tr>'
    h+='</table>'
    for event in rollbacks:
        if event['preceding_predicted_drop_pp'] is not None and event['preceding_predicted_drop_pp'] > 3:
            h+=f'<p>注意：{event["time_seconds"]:.1f}s 这次回退前，预测已在 {event["preceding_time"]:g}→{event["before_time"]:g}s 下降 {event["preceding_predicted_drop_pp"]:.2f} 个百分点。因此表中紧邻真值回退的 {event["predicted_drop_pp"]:.2f} pp 只表示指定两个采样点的差值，不能据此断言模型完全漏掉了本次回退；应结合曲线观察幅度偏差和时间错位。</p>'
    h+='<h2>逐点结果与原始输出</h2><p>阶段 ID：1 拿起套筒；2 拿起螺钉；3 对齐并插入；4 将组件放入盒子。</p><div class="scroll"><table><tr><th>t / s</th><th>输入上一阶段</th><th>真值当前阶段</th><th>预测当前阶段</th><th>真值 PT</th><th>预测 PT</th><th>绝对误差 / pp</th><th>完整状态</th></tr>'
    for row,p,good in zip(rows,parsed,matches):
        h+=f'<tr><td>{row["time_seconds"]:g}</td><td>{row["input_state"]["nowid"]}</td><td>{row["target"]["nowid"]}</td><td>{p["nowid"]}</td><td>{100*row["target"]["progress"]:.2f}%</td><td>{100*row["predicted_progress"]:.2f}%</td><td>{100*abs(row["predicted_progress"]-row["target"]["progress"]):.2f}</td><td class="{"green" if good else "red"}">{"一致" if good else "偏差"}</td></tr>'
    h+='</table></div>'
    for row,p,good in zip(rows,parsed,matches):
        if not good:
            h+=f'<details><summary>{row["time_seconds"]:g}s 的阶段偏差</summary><p>输入：{e(describe_state(row["input_state"]))}</p><p>真值：{e(row["target_response"])}</p><pre>{e(row["response"])}</pre></details>'
    h+='<h2>核验与适用范围</h2>'
    h+=f'<p>{len(rows)} 个时刻全部完成；文件哈希、逐帧标签与原索引一致性、因果窗口和反馈链均已核验。仅首次读取阶段真值作为提示，后续真值只用于评分。完整生成文本解析成功率 {metrics["parse_valid_rate"]:.1%}；状态结构自洽率 {metrics["state_consistent_rate"]:.1%}。本次为新滑动窗口下的一次真实闭环推理，不复用旧评测预测。</p>'
    h+=f'<p>这是验证集中的一个 episode，不是独立测试集；单个片段的指标不能代替全验证集统计。窗口重叠，样本间相关；任务意图固定。“{hz:g}Hz”指视频时间上的预测频率，不代表每秒实际能计算 {hz:g} 次。</p>'
    if manifest.get('generation') == 'greedy_with_previous_prediction_draft_verification':
        accepted=sum(r['draft_accepted'] for r in rows)
        h+=f'<p>本次使用跨 episode 批量推理，缓存的视频像素与原数据读取路径逐像素抽查一致。上一帧模型输出仅作为草稿，当前帧模型逐 token 校验（包含结束符），通过且最小 logit 差值达到 {manifest["minimum_draft_margin"]:g} 时接受；否则普通贪心生成。接受 {accepted}/{len(rows)} 次。PT 独立走当前视觉与提示的原回归路径，不使用草稿答案或当前真值。不同批量形状可能引入 BF16 数值差异；文字一致性控制实验随总报告保存。平均每个批次 {metrics["average_inference_seconds"]:.2f} 秒，单样本摊销 {sum(r["amortized_seconds"] for r in rows)/len(rows):.3f} 秒；结果中的 elapsed_seconds 是本 GPU 开始运行到该 episode 完成的墙钟时间。</p>'
    else:
        h+=f'<p>平均每次处理 {metrics["average_inference_seconds"]:.2f} 秒，总推理阶段耗时 {result["elapsed_seconds"]:.1f} 秒。</p>'
    h+=f'<p>检查点 SHA-256：<code>{manifest["checkpoint_sha256"]}</code><br>原始预测：<code>predictions.jsonl</code>；逐帧真值：<code>ground_truth.json</code>；统计与反馈审计：<code>metrics.json</code>。</p></html>'
    (directory / "report.html").write_text(h)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    run(parser.parse_args().directory)
