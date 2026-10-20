"""Render the actual predictions into a curve and a compact offline report."""
from pathlib import Path
import base64
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "manifest.json").read_text())
rows = [json.loads(line) for line in (ROOT / "predictions.jsonl").read_text().splitlines()]
assert len(rows) == manifest["predicted_points"]
assert [r["frame_index"] for r in rows] == manifest["sample_frames"]
valid = [r for r in rows if r["status"] == "ok"]
if not valid:
    raise RuntimeError("No valid predictions; cannot present a result curve")
x = [r["time_s"] for r in rows]
y = [float("nan") if r["score_percent"] is None else r["score_percent"] for r in rows]
values = [r["score_percent"] for r in valid]
full = next((r for r in valid if r["score_percent"] >= 100), None)
after_full = [r["score_percent"] for r in valid if full and r["time_s"] > full["time_s"]]
stats = {
    "points": len(rows), "valid_points": len(valid),
    "min_percent": min(values), "max_percent": max(values),
    "first_percent": rows[0]["score_percent"], "last_percent": rows[-1]["score_percent"],
    "negative_differences": sum(b < a for a, b in zip(y, y[1:])),
    "mean_percent": sum(values) / len(values),
    "first_100_percent_time_s": full["time_s"] if full else None,
    "minimum_after_first_100_percent": min(after_full) if after_full else None,
}
(ROOT / "summary.json").write_text(json.dumps(stats, indent=2))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(11.5, 4.4), layout="constrained")
fig.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
line, = ax.plot(x, y, color="#2563eb", linewidth=2.4, marker="o", markersize=3.7,
                label="RoboBrain 2.5-4B / GRM", zorder=3)
line.set_gid("prediction_curve")
ax.axhline(0, color="#9ca3af", linewidth=.8, zorder=1)
ax.set(xlim=(0, max(x)), ylim=(min(-5, min(values) - 5), 105),
       xlabel="Video time (s)", ylabel="Start-to-current GRM score (%)",
       title="Block swap + button presses")
ax.grid(axis="y", color="#e5e7eb", linewidth=.8)
ax.legend(loc="upper left", frameon=False)
fig.savefig(ROOT / "progress_curve.png", dpi=200, facecolor="white")
fig.savefig(ROOT / "progress_curve.svg", facecolor="white")
plt.close(fig)
svg = (ROOT / "progress_curve.svg").read_text()
svg = svg[svg.index("<svg"):]
video = base64.b64encode((ROOT / "input/full_video.mp4").read_bytes()).decode()
task = html.escape(manifest["task"])
timing = manifest["timing"]
last = "无有效值" if stats["last_percent"] is None else f'{stats["last_percent"]:g}%'
page = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboBrain 2.5 · 方块交换进度</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#182235;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1120px;margin:20px auto;padding:22px 26px;background:white;border:1px solid #dce3ed;border-radius:12px}
h1{font-size:24px;line-height:1.3;margin:0 0 7px}p{margin:7px 0}.meta{color:#627088;font-size:13px}
.chart{margin:8px 0 4px}.chart svg{width:100%;height:auto;display:block}
.toolbar{display:flex;align-items:center;gap:20px;flex-wrap:wrap;font-size:14px}
a{color:#1d4ed8;text-decoration:none}label{cursor:pointer}input{accent-color:#2563eb}
.note{padding:10px 12px;background:#eff6ff;border-left:3px solid #2563eb;margin:8px 0}
details{border-top:1px solid #e2e8f0;padding:9px 0}summary{cursor:pointer;font-weight:600}
video{display:block;width:min(640px,100%);max-height:480px;margin:10px auto;background:#111}
.task{font-size:14px;margin:8px 0;max-width:1000px}
@media(max-width:700px){main{margin:0;border-radius:0;padding:16px}h1{font-size:21px}}
</style>
<main>
<h1>方块交换：三次移动＋每次按键</h1>
<p class="meta">RoboBrain 2.5-4B · 17.68 秒视频 · 3 FPS · __POINTS__ 个预测点 · 2026-10-20</p>
<p><b>评测输入：</b>原始任务指令＋首尾帧参考＋首帧/当前帧；缺失腕视图用黑图补齐，共 8 图。无历史帧、无专家轨迹。</p>
<div class="chart">__SVG__</div>
<div class="toolbar">
<label><input id="toggle" type="checkbox" checked> 显示预测曲线</label>
<a href="progress_curve.png" download>PNG</a><a href="predictions.csv" download>原始 CSV</a>
<a href="manifest.json">完整输入与参数</a>
</div>
<p class="note"><b>结果：</b>__OBSERVATION__末帧 __LAST__。
这是相对首帧的 GRM 分数，并非标注完成率。曲线未平滑、未强制单调或固定终点。</p>
<p class="meta">首尾画面相似，按键没有持久可见效果；本次静态接口不具备按键次数记忆。末帧仅作为离线完成参考，未独立验证成功。</p>
<details><summary>示范视频与原始任务指令</summary>
<video controls preload="metadata" src="data:video/mp4;base64,__VIDEO__"></video><p class="task">__TASK__</p>
</details>
<details><summary>运行信息</summary><p class="meta">
原图 640×480；贪心解码；batch=8；有效输出 __VALID__/__POINTS__。
模型加载后评测用时 __SECONDS__ 秒；批量摊销 __AMORTIZED__ 秒/帧（不是串行单帧延迟）。
GPU：__GPU__。权重 SHA-256 已核验。</p></details>
</main><script>
document.getElementById('toggle').addEventListener('change',e=>{
document.getElementById('prediction_curve').style.visibility=e.target.checked?'visible':'hidden';
});
</script></html>"""
replace = {
    "__POINTS__": str(len(rows)), "__SVG__": svg,
    "__MIN__": f'{min(values):g}', "__MAX__": f'{max(values):g}', "__LAST__": last,
    "__VIDEO__": video, "__TASK__": task, "__VALID__": str(len(valid)),
    "__SECONDS__": f'{timing["evaluation_wall_s"]:.1f}',
    "__AMORTIZED__": f'{timing["amortized_s_per_frame"]:.2f}',
    "__GPU__": html.escape(manifest["gpu"]),
    "__OBSERVATION__": (
        f'{full["time_s"]:g} 秒给出 100%，随后最低回落至 {min(after_full):g}%；'
        if full and after_full else f'预测范围 {min(values):g}%～{max(values):g}%；'
    ),
}
for key, value in replace.items():
    page = page.replace(key, value)
(ROOT / "report.html").write_text(page)
print(json.dumps(stats, ensure_ascii=False))
