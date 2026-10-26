"""训练数据检查页面：固定抽帧结果、静止过滤统计和四类样本。"""

import html
import json
import math

from PIL import Image, ImageDraw, ImageFont


def write_preview(output, annotation, video_info, samples, summary):
    font = ImageFont.load_default(size=15)
    indices = video_info["kept_frames"]
    boundaries = {row["frame_index"] for row in video_info["observations"] if row["annotation_boundary"]}
    tile_w, tile_h, columns = 160, 150, 7
    sheet = Image.new("RGB", (tile_w * columns, tile_h * max(1, math.ceil(len(indices) / columns))), "#f1f5f9")
    draw = ImageDraw.Draw(sheet)
    for i, frame_index in enumerate(indices):
        x, y = i % columns * tile_w, i // columns * tile_h
        with Image.open(output / "frames" / f"{frame_index:06d}.jpg") as source:
            tile = source.copy()
            tile.thumbnail((tile_w - 8, tile_h - 30))
        sheet.paste(tile, (x + 4, y + 4))
        color = "#b45309" if frame_index in boundaries else "#1d4ed8"
        draw.rectangle((x + 2, y + 2, x + tile_w - 3, y + tile_h - 3), outline=color, width=2)
        label = f"f{frame_index}  {video_info['timestamps'][frame_index]:.2f}s"
        draw.text((x + 6, y + tile_h - 25), label, fill=color, font=font)
    sheet.save(output / "observations.png")

    stage_rows = []
    for stage in annotation["stages"]:
        start, end = stage["start_frame"], stage["end_frame"]
        kept = [i for i in indices if start <= i < end]
        stage_rows.append(
            f"<tr><td>{stage['stage_id']}</td><td>[{start}, {end})</td><td>{len(kept)}</td>"
            f"<td>{html.escape(stage['state_description'])}</td></tr>"
        )

    cards = []
    # 每类展示一个有效/无效样本，全部结果留在 samples.jsonl，避免页面堆积数百组重复图。
    selected = set()
    for sample in samples:
        key = sample["metadata"]["case"], sample["target"]["valid"]
        if key in selected:
            continue
        selected.add(key)
        target, metadata = sample["target"], sample["metadata"]
        pictures = []
        for frame in sample["inputs"]["frames"]:
            i = frame["queue_index"]
            chosen = target["valid"] and target["start_index"] <= i < target["end_index"]
            pictures.append(
                f'<figure class="{"chosen" if chosen else "ordinary"}"><img loading="lazy" '
                f'src="{frame["image"]}"><figcaption>队列 {i} · 原帧 {frame["frame_index"]}</figcaption></figure>'
            )
        cards.append(
            f"<section><h3>{html.escape(sample['sample_id'])}</h3>"
            f"<p>原事件 {metadata['event_interval']} · 输入窗口 {metadata['window']} · "
            f"覆盖 {metadata['coverage']:.1%} · 目标区间 {metadata['target_interval']}</p>"
            f"<pre>{html.escape(sample['target_text'])}</pre>"
            f'<div class="queue">{"".join(pictures)}</div></section>'
        )
    config = summary["config"]
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>M₁ 训练样本检查</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;margin:32px auto;max-width:1160px;
padding:0 20px;color:#172b44;background:#f8fafc}}
h1,h2,h3{{line-height:1.3}}
section{{background:white;padding:20px;margin:20px 0;border:1px solid #dbe3ee;border-radius:10px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #dde5ef;padding:8px}}
.overview{{width:100%;height:auto}}.queue{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:145px;border:3px solid #94a3b8}}
figure.chosen{{border-color:#16a34a}}figure img{{width:100%;display:block}}figcaption{{font-size:12px;text-align:center}}
pre{{white-space:pre-wrap;background:#edf2f8;padding:10px}}a{{color:#245bb3}}</style>
<h1>M₁ 训练样本构造检查</h1><p>{html.escape(annotation["task_prompt"])}</p>
<section><h2>抽帧结果</h2><p>{video_info["fps"]:g} FPS · {video_info["frame_count"]} 原帧 ·
{summary["candidate_count"]} 个固定时钟候选 · 过滤 {summary["static_removed"]} 个静止候选 ·
公共队列 {len(indices)} 帧（包含标注边界）</p>
<p>静止检测：灰度图缩小到宽 {config["difference_width"]} 像素并轻度模糊；
像素差大于 {config["pixel_delta"]} 的比例不足 {config["min_changed_fraction"]:.2%} 时过滤。
比较对象为上一张保留帧，边界帧免过滤。这是本次使用的初始参数。</p>
<p>蓝框：普通保留帧；棕框：强制保留的原始标注边界。样本专属的扰动边界只补入各自样本。</p>
<img class="overview" src="observations.png" alt="公共观测队列"></section>
<section><h2>标注区间覆盖</h2><table><tr><th>区间</th><th>原帧范围</th><th>保留帧数</th><th>原始累计状态描述</th></tr>
{"".join(stage_rows)}</table></section>
<section><h2>样本统计</h2><p>生成 {len(samples)} 条，空队列跳过 {summary["skipped_empty"]} 条。
每个事件的各类采样次数：{html.escape(json.dumps(config["samples_per_case"]))}。</p>
<pre>{html.escape(json.dumps(summary["by_case"], ensure_ascii=False, indent=2))}</pre>
<p><a href="samples.jsonl">全部训练样本</a> · <a href="summary.json">配置与统计</a> ·
<a href="observations.json">逐候选过滤记录</a></p>
<p>绿色边框表示目标切分片段；无效样本只监督 valid=0。专家 tokens 仅预留接口，本页没有运行模型。</p></section>
{"".join(cards)}</html>"""
    (output / "preview.html").write_text(content, encoding="utf-8")
