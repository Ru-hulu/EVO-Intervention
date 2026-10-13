"""Assemble complete inference shards into metrics and a standalone HTML report."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import html
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from qwen_training.evaluation_metrics import normalize, parse_response, summarize


def load_complete_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {(sample_id, mode) for sample_id in manifest["sample_ids"] for mode in manifest["modes"]}
    rows, seen, completions = [], set(), []
    for rank in range(manifest["world_size"]):
        completion = json.loads((directory / f"complete-rank{rank:02d}.json").read_text())
        assert completion["status"] == "complete" and completion["rank"] == rank
        raw = (directory / f"predictions-rank{rank:02d}.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == completion["predictions_sha256"]
        shard = [json.loads(line) for line in raw.decode().splitlines() if line]
        assert len(shard) == completion["responses"] == 2 * completion["samples"]
        for row in shard:
            key = row["sample_id"], row["mode"]
            assert key in expected and key not in seen, f"Unexpected or duplicate prediction: {key}"
            seen.add(key)
        rows.extend(shard)
        completions.append(completion)
    assert seen == expected, f"Missing {len(expected - seen)} expected predictions"
    assert len({r["sample_id"] for r in rows}) == manifest["samples"]
    rows.sort(key=lambda r: (r["dataset"], r["video_id"], r["task_index"], str(r["episode"]), r["frame_indices"][-1], r["sample_id"], r["mode"]))
    return manifest, rows, completions


def load_stopped_run(directory):
    """Score paired records from a user-stopped run without inventing completion records."""
    stop = json.loads((directory.parent / 'stop_for_partial_report.json').read_text())
    assert stop['status'] == 'stopped_for_partial_report' and not stop['remaining_target_pids']
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = {(sid, mode) for sid in manifest['sample_ids'] for mode in manifest['modes']}
    records, artifacts = {}, []
    for rank in range(manifest['world_size']):
        path = directory / f'predictions-rank{rank:02d}.jsonl'
        raw = path.read_bytes()
        count, incomplete_bytes = 0, 0
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b'\n'):
                incomplete_bytes += len(line)
                continue
            row = json.loads(line)
            key = row['sample_id'], row['mode']
            assert key in expected and key not in records, f'Unexpected or duplicate prediction: {key}'
            records[key] = row
            count += 1
        artifacts.append(dict(rank=rank, status='stopped_partial_snapshot', complete_json_records=count,
                              discarded_trailing_bytes=incomplete_bytes,
                              predictions_sha256=hashlib.sha256(raw).hexdigest()))
    included = [sid for sid in manifest['sample_ids'] if all((sid, mode) in records for mode in manifest['modes'])]
    assert included, 'No samples have both prompt modes completed'
    for sid in included:
        left, right = [records[(sid, mode)] for mode in manifest['modes']]
        for key in ['target', 'input_state']:
            assert left[key] == right[key], f'Paired metadata mismatch: {sid}/{key}'
    rows = [records[(sid, mode)] for sid in included for mode in manifest['modes']]
    planned = manifest['samples']
    manifest.update(samples=len(included), sample_ids=included, planned_samples=planned,
                    partial_evaluation=True, stopped_unix=stop['finished_unix'],
                    unpaired_records_excluded=len(records)-len(rows),
                    incomplete_trailing_bytes=sum(a['discarded_trailing_bytes'] for a in artifacts),
                    source_manifest_sha256=hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest())
    rows.sort(key=lambda r: (r['dataset'], r['video_id'], r['task_index'], str(r['episode']), r['frame_indices'][-1], r['sample_id'], r['mode']))
    return manifest, rows, artifacts


def aggregate(directory, allow_partial=False):
    manifest, rows, completions = load_stopped_run(directory) if allow_partial else load_complete_run(directory)
    metrics = summarize(rows)
    payload = dict(manifest=manifest, metrics=metrics, rank_completions=completions)
    (directory / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    failures = []
    for row in rows:
        prediction = parse_response(row["response"])
        state_wrong = any(prediction[k] != (row["target"][k] if k == "nowid" else sorted(row["target"][k]))
                          for k in ("nowid", "finishid", "restid"))
        if not prediction["parse_valid"] or state_wrong or abs(row["predicted_progress"] - row["target"]["progress"]) > 0.1:
            failures.append(row)
    with (directory / "errors.jsonl").open("w") as output:
        for row in failures:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return payload, rows


def coverage_section(directory, manifest, rows):
    if not manifest.get('partial_evaluation'):
        return []
    planned = json.loads((directory / 'planned_validation_counts.json').read_text())
    counts = defaultdict(int)
    for row in rows:
        if row['mode'] == 'short':
            counts[row['dataset'] + '/' + row['video_id']] += 1
    stopped = datetime.fromtimestamp(manifest['stopped_unix'], ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    body = ''.join(f'<tr><td>{html.escape(key)}</td><td>{counts[key]:,}</td><td>{total:,}</td><td>{counts[key]/total:.1%}</td></tr>'
                   for key, total in planned['by_video'].items())
    return ['<h2>停止时间与实际覆盖范围</h2>',
            f'<p>按用户要求于北京时间 {stopped} 停止推理，报告仅复用停止前已保存的结果，没有补跑剩余样本。'
            f'原计划 {manifest["planned_samples"]:,} 个样本，本报告纳入两种输入均完整保存的 {manifest["samples"]:,} 个样本'
            f'（{manifest["samples"]/manifest["planned_samples"]:.2%}）；未纳入 {manifest["planned_samples"]-manifest["samples"]:,} 个样本。'
            f'另有 {manifest["unpaired_records_excluded"]} 条已保存但缺少另一种输入配对的记录未参与指标计算。'
            '原始文件完整保留，未修改原始评估清单或伪造完成标记。</p>',
            '<p><b>本次为顺序执行后提前停止的验证集子集，不是随机抽样。</b>各视频覆盖比例不同，以下整体成绩不能代替完整验证集或独立测试集成绩。</p>',
            '<div class="scroll"><table><tr><th>验证视频</th><th>已纳入</th><th>计划</th><th>覆盖率</th></tr>' + body + '</table></div>']


def prompt_sensitivity_section(directory, rows):
    paired = defaultdict(dict)
    for row in rows:
        paired[row['sample_id']][row['mode']] = row
    differences = sorted(abs(r['short']['predicted_progress']-r['context']['predicted_progress'])*100 for r in paired.values())
    exact = sum(x == 0 for x in differences) / len(differences)
    average = sum(differences) / len(differences)
    content = ['<h2>进度是否利用 Prompt 和上一状态</h2>',
               f'<p>在本报告纳入的配对样本中，两种输入的进度预测平均仅差 <b>{average:.5f} 个百分点</b>；'
               f'<b>{exact:.2%}</b> 的预测值完全相同，最大差值为 {max(differences):.3f} 个百分点。'
               '这是逐样本预测值的对比，不是仅比较两个平均误差。</p>']
    path = directory.parent / 'prompt_pooling_diagnostic.json'
    if path.exists():
        diagnostic = json.loads(path.read_text())
        assert diagnostic['status'] == 'complete'
        cases = diagnostic['results']
        variants = [v for case in cases for v in case['variants']]
        maximum_text = max(v['text_weight'] for v in variants)
        maximum_range = max(case['forced_previous_stage_progress_range_pp'] for case in cases)
        content += [f'<p>停止评估之前已完成的独立诊断抽取了 {len(cases)} 个样本，覆盖简单与恢复数据及四个阶段。'
                    '固定同一组视觉特征，将上一阶段依次替换为 1、2、3、4，同时调整已完成和未完成阶段，使状态描述自洽。'
                    f'这些替换产生的进度最大变化为 <b>{maximum_range:.5f} 个百分点</b>；'
                    f'全部文本 token 的池化权重最高仅 <b>{maximum_text:.3g}</b>。这些诊断不是本次停止后追加的评估。</p>',
                    '<p>当前输入顺序为「视觉 token → Prompt 与上一状态」。语言主干的因果注意力使前面的视觉 token 无法读取后面的文字；'
                    '进度头虽然允许对所有输入位置池化，但在这些诊断中几乎只选择视觉位置。'
                    '<b>因此当前检查点的进度分支对文本基本不敏感。</b>阶段文本生成仍会利用提示，不能把此结论推广到整个模型。'
                    '诊断规模为 8 个样本，属于机制证据，不是全验证集的池化权重统计。</p>']
    return content


def metric_table(groups):
    columns = [("样本权重总数", "weighted_sample_count", ".0f"),
               ("进度 MAE（百分点）", "progress_mae_pp", ".2f"),
               ("进度 RMSE（百分点）", "progress_rmse_pp", ".2f"),
               ("进度 P95 误差（百分点）", "progress_p95_ae_pp", ".2f"),
               ("误差 ≤5 个百分点", "progress_within_5pp", ".2%"),
               ("当前阶段准确率", "current_stage_accuracy", ".2%"),
               ("当前阶段 Macro-F1", "current_stage_macro_f1", ".4f"),
               ("完整状态准确率", "full_state_accuracy", ".2%"),
               ("已完成阶段准确率", "finishid_exact_accuracy", ".2%"),
               ("未完成阶段准确率", "restid_exact_accuracy", ".2%"),
               ("文本可解析率", "parse_valid_rate", ".2%")]
    header = "<tr><th>指标</th>" + "".join(f"<th>{html.escape(name)}</th>" for name, _ in groups) + "</tr>"
    body = []
    for label, key, fmt in columns:
        body.append("<tr><th>" + label + "</th>" + "".join(f"<td>{format(values[key], fmt)}</td>" for _, values in groups) + "</tr>")
    return '<div class="scroll"><table>' + header + "".join(body) + "</table></div>"


def create_figures(rows, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"short": "#2672b9", "context": "#df7b36"}
    figures = []

    def finish(figure, title):
        buffer = io.StringIO()
        figure.tight_layout()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(figure)
        svg = buffer.getvalue()
        figures.append((title, svg[svg.index("<svg"):]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in ("short", "context"):
        subset = [r for r in rows if r["mode"] == mode]
        absolute = np.array([abs(r["predicted_progress"] - r["target"]["progress"]) * 100 for r in subset])
        axes[0].hist(absolute, bins=np.arange(0, 101, 2), weights=np.ones(len(absolute))/len(absolute),
                     alpha=0.55, color=colors[mode], label=mode)
        axes[1].plot(np.sort(absolute), np.arange(1, len(absolute)+1)/len(absolute), color=colors[mode], label=mode)
    axes[0].set(xlabel="Absolute progress error (percentage points)", ylabel="Fraction of samples", xlim=(0, 100))
    axes[1].set(xlabel="Absolute progress error (percentage points)", ylabel="Cumulative fraction", xlim=(0, 100), ylim=(0, 1))
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.2)
    finish(fig, "进度误差分布与累计覆盖率")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mode in zip(axes, ("short", "context")):
        matrix = np.array(metrics[mode]["stage_confusion"])
        normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{normalized[i,j]:.1%}\n({int(matrix[i,j])})", ha="center", va="center", fontsize=8,
                        color="white" if normalized[i,j] > 0.6 else "black")
        ax.set(xticks=range(5), xticklabels=["1", "2", "3", "4", "invalid"], yticks=range(4),
               yticklabels=["1", "2", "3", "4"], xlabel="Predicted stage", ylabel="True stage", title=mode)
    finish(fig, "当前阶段混淆矩阵：行是真值，列是预测")

    episodes = defaultdict(list)
    for row in rows:
        if row["mode"] == "short" and "intentional-drop" in row["dataset"]:
            episodes[(row["dataset"], row["video_id"], row["task_index"], row["episode"])].append(row)
    def recovery_score(group):
        ordered = sorted(group, key=lambda r: r["frame_indices"][-1])
        stages = [r["target"]["nowid"] for r in ordered]
        return (sum(b < a for a, b in zip(stages, stages[1:])), sum(a != b for a, b in zip(stages, stages[1:])), len(ordered))
    choices = sorted(episodes, key=lambda key: recovery_score(episodes[key]), reverse=True)[:4]
    for key in choices:
        selected = [r for r in rows if (r["dataset"], r["video_id"], r["task_index"], r["episode"]) == key]
        truth = {}
        for row in selected:
            truth[row["frame_indices"][-1]] = row["target"]["progress"]
        fig, ax = plt.subplots(figsize=(10, 3.4))
        frames = sorted(truth)
        ax.plot([x/30 for x in frames], [100*truth[x] for x in frames], color="#192d38", label="label", linewidth=2)
        for mode in ("short", "context"):
            group = [r for r in selected if r["mode"] == mode]
            ax.scatter([r["frame_indices"][-1]/30 for r in group], [r["predicted_progress"]*100 for r in group],
                       s=9, alpha=0.5, color=colors[mode], label=mode)
        ax.set(xlabel="Time in original video (seconds)", ylabel="Progress (%)", ylim=(-3, 103),
               title=f"{key[1]} / task {key[2]} / episode {key[3]}")
        ax.grid(alpha=0.2); ax.legend()
        finish(fig, f"恢复样本进度曲线：{key[1]} / task {key[2]} / episode {key[3]}")
    return figures


def render(directory, allow_partial=False):
    payload, rows = aggregate(directory, allow_partial=allow_partial)
    manifest, metrics = payload["manifest"], payload["metrics"]
    figures = create_figures(rows, metrics)
    groups = [("视觉＋简短指令", metrics["short"]), ("视觉＋阶段说明＋上一状态", metrics["context"]),
              ("按 30% / 70% 加权", metrics["mixed_30_70"])]
    mixed = metrics["mixed_30_70"]
    title = f"Qwen3.5-2B · Epoch {manifest['checkpoint_epoch']} 验证集推理报告" + ('（提前停止）' if manifest.get('partial_evaluation') else '')
    content = [f"<h1>{title}</h1>",
        f"<p class='lead'>进度平均绝对误差 <b>{mixed['progress_mae_pp']:.2f} 个百分点</b>；当前阶段准确率 "
        f"<b>{mixed['current_stage_accuracy']:.2%}</b>；完整阶段状态准确率 <b>{mixed['full_state_accuracy']:.2%}</b>。上述汇总按两种输入的 30% / 70% 权重计算。</p>",
        *coverage_section(directory, manifest, rows),
        "<h2>评估范围与口径</h2>",
        f"<p>评估 {manifest['samples']:,} 个不同索引样本，每个样本分别运行两种输入，共 {len(rows):,} 条实际生成结果。"
        "使用训练时按原视频隔离的验证集，未使用训练视频；这些验证视频已用于逐轮验证，因此本报告不代表从未用于模型选择的独立测试集成绩。"
        "窗口之间存在重叠，样本数不能当作相同数量的独立实验。</p>",
        "<p>模型只接收视觉信息与输入提示，自回归贪心生成答案；标准答案只用于事后评分。"
        "带上一状态的输入使用标注中的上一观测状态，并非模型自己滚动预测的状态。每个样本测试两种输入，30/70 汇总是加权评估，不是随机抽取 30%/70% 子集。</p>",
        f"<p>检查点：<code>{html.escape(manifest['checkpoint'])}</code><br>SHA-256：<code>{manifest['checkpoint_sha256']}</code><br>"
        f"训练步数：{manifest['checkpoint_step']:,}；生成上限：{manifest['max_new_tokens']} tokens；测试状态：{'按用户要求停止，仅统计已完成配对样本' if manifest.get('partial_evaluation') else ('短程检查，非完整评估' if manifest['smoke_test'] else '完整评估')}。</p>",
        "<h2>进度与阶段预测</h2>", metric_table(groups),
        "<p>进度使用 0–100% 表示：真实 60%、预测 66% 对应 6 个百分点误差。MAE 是平均绝对误差；RMSE 对大误差更敏感；P95 表示约 95% 样本误差不超过该值。</p>",
        f"<p>加权进度偏差（预测−真值）为 {mixed['progress_bias_pp']:+.2f} 个百分点；正值偏高估，负值偏低估。"
        f"误差 ≤10 个百分点的覆盖率为 {mixed['progress_within_10pp']:.2%}。</p>",
        "<p>阶段名称采用标签中的四个标准名称：1 拿起套筒；2 拿起螺钉；3 将螺钉和套筒对齐并插入；4 将装配好的组件放入盒子。"
        "当前阶段逐样本比对；已完成和未完成阶段按集合比对，不要求集合内顺序一致。完整状态要求三个字段全部正确。解析失败保留在分母中，未被丢弃。</p>",
        f"<p>已完成阶段集合 micro-F1：{mixed['finishid_micro_f1']:.4f}；未完成阶段集合 micro-F1：{mixed['restid_micro_f1']:.4f}。"
        f"全文一致率：{mixed['text_exact_accuracy']:.2%}；字符编辑错误率：{mixed['character_error_rate']:.2%}；"
        f"任务意图逐字一致率：{mixed['intent_exact_accuracy']:.2%}。意图标签全为同一句话，该项不能证明多任务意图识别能力。</p>",
        "<p>解析器识别标准阶段名称，不用另一个模型猜测同义表达。字符指标统一空白及全角标点；全文一致率仍包含模板、意图和阶段顺序。解析失败案例可在原始结果中复核。"
        "集合 micro-F1 中无法解析的集合视为空集合，因此需结合集合准确率与可解析率一起阅读。</p>"]
    content.extend(prompt_sensitivity_section(directory, rows))
    for heading, svg in figures:
        content.extend([f"<h2>{html.escape(heading)}</h2>", svg])
    content.append("<p>曲线显示验证集实际采样窗口；散点是每个窗口的独立预测。同一当前帧可能对应不同历史窗口，因而有多个预测点。"
                   "恢复曲线按标注当前阶段随时间的回退次数选取，连接真值仅辅助阅读，不表示对未采样时刻进行了预测。</p>")
    for field, heading in [("dataset", "按数据集分组"), ("sample_type", "按采样类型分组"), ("transition", "按上一阶段 → 当前阶段分组"), ("video_id", "按原视频分组")]:
        content.append(f"<h2>{heading}</h2>")
        for label, values in metrics["by_" + field].items():
            content.append(f"<details><summary>{html.escape(label)}</summary>" + metric_table(
                [(mode, values[mode]) for mode in ("short", "context") if values[mode] is not None]) + "</details>")
    changed = [r for r in rows if r["input_state"]["nowid"] != r["target"]["nowid"]]
    content.append("<h2>阶段切换与上一状态基线</h2>")
    content.append(f"<p>在本报告纳入的样本中，直接复制上一阶段的准确率为 {mixed['previous_stage_baseline_accuracy']:.2%}，在阶段已变化的样本中为 0%。"
                   f"本次包含 {len(changed)//2:,} 个阶段发生变化的索引样本，应重点阅读对应分组结果，避免大量阶段不变样本掩盖切换错误。</p>")
    wrong_stage = [r for r in rows if parse_response(r["response"])["nowid"] != r["target"]["nowid"]]
    progress_errors = sorted(rows, key=lambda r: abs(r["predicted_progress"]-r["target"]["progress"]), reverse=True)
    examples, example_keys = [], set()
    for row in wrong_stage[:12] + progress_errors[:12]:
        key = row["sample_id"], row["mode"]
        if key not in example_keys:
            examples.append(row); example_keys.add(key)
    content.append("<h2>典型错误与最大进度误差</h2>")
    for row in examples:
        content.append(f"<details><summary>{html.escape(row['sample_id'])} · {row['mode']} · "
                       f"进度真值 {row['target']['progress']*100:.2f}% / 预测 {row['predicted_progress']*100:.2f}%</summary>"
                       f"<p>标准：{html.escape(row['target_response'])}</p><p>生成：{html.escape(row['response'])}</p></details>")
    content.append("<h2>可复核文件</h2><p>同目录 metrics.json 包含纳入样本的完整分组指标及原始文件哈希；predictions-rankXX.jsonl 保存原始生成文本、进度真值、预测值和元数据；"
                   "errors.jsonl 保存已纳入样本中的阶段错误、解析失败或进度误差超过 10 个百分点的记录；manifest.json 保留原计划模型、索引和划分信息。"
                   "上级目录 stop_for_partial_report.json 记录停止进程的核验，prompt_pooling_diagnostic.json 保存此前完成的提示敏感性诊断。</p>")
    document = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
    document += f"<title>{title}</title><style>body{{max-width:1100px;margin:40px auto;padding:0 24px;color:#203440;font:16px/1.7 system-ui,sans-serif}}h1{{font-size:30px}}h2{{margin-top:32px}}.lead{{background:#edf5fa;padding:20px;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d8e0e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow-x:auto}}code{{word-break:break-all;font-size:13px}}svg{{max-width:100%;height:auto}}details{{padding:8px 0;border-bottom:1px solid #dde5e9}}summary{{cursor:pointer}}p{{overflow-wrap:anywhere}}</style>"
    document += "<body>" + "\n".join(content) + "</body></html>"
    (directory / "report.html").write_text(document)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Report saved paired samples only after user-requested stop is verified")
    args = parser.parse_args()
    payload = aggregate(args.directory, allow_partial=args.allow_partial)[0] if args.metrics_only else render(args.directory, allow_partial=args.allow_partial)
    print(json.dumps(payload["metrics"]["mixed_30_70"], ensure_ascii=False, indent=2))
