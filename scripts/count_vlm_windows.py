"""枚举 30fps 视频的六帧窗口，统计阶段类型及最后两张采样帧的状态对。"""

import argparse
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path


def state_of(subtask, task_ids):
    now = int(subtask["nowid"])
    rest = tuple(sorted(int(x) for x in subtask["restid"].split(",") if x.strip()))
    finish = tuple(sorted(task_ids - {now} - set(rest)))
    return now, rest, finish


def count_windows(label_dir):
    files = sorted(p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".json")
    type_counts, pair_counts, covered_segments = Counter(), Counter(), Counter()
    pairs_by_type = defaultdict(Counter)
    states, names, episode_rows = set(), {}, []
    segment_count = 0

    for path in files:
        data = json.loads(path.read_text())
        task_ids = {int(task["id"]) for task in data["taskid"]}
        names.update({int(task["id"]): task["name"] for task in data["taskid"]})
        for task_id in task_ids:
            type_counts[(task_id,)] += 0
        for task in data["tasks"]:
            for episode in task["episodes"]:
                subtasks = episode["subtasks"]
                segment_count += len(subtasks)
                episode_states = [state_of(s, task_ids) for s in subtasks]
                states.update(episode_states)
                for before, after in zip(episode_states, episode_states[1:]):
                    type_counts[(before[0], after[0])] += 0

                # 30FPS 帧范围两端均包含；episode 内逐原始帧记录所属 subtask 段。
                frame_segments = []
                next_frame = subtasks[0]["30FPSstart_frame"]
                for index, subtask in enumerate(subtasks):
                    start, end = subtask["30FPSstart_frame"], subtask["30FPSend_frame"]
                    if start != next_frame or end < start:
                        raise ValueError(f"{path.name} episode {episode['episode']} 的标注不连续")
                    frame_segments.extend([index] * (end - start + 1))
                    next_frame = end + 1

                episode_counts = Counter()
                # f, f+30, f+60, f+90, f+120, f+150；起点每次移动一个原始帧。
                for first_frame in range(len(frame_segments) - 150):
                    first_segment = frame_segments[first_frame]
                    last_segment = frame_segments[first_frame + 150]
                    covered = last_segment - first_segment + 1
                    covered_segments[covered] += 1
                    episode_counts[covered] += 1
                    # 检查完整时间范围，不能漏掉采样点之间的短阶段；不跨 episode。
                    if covered > 2:
                        continue
                    first_state, last_state = episode_states[first_segment], episode_states[last_segment]
                    sample_type = (first_state[0],) if covered == 1 else (first_state[0], last_state[0])
                    # 标签使用倒数第二张采样帧（末帧前 1 秒），不是末帧前一个原始帧。
                    previous_state = episode_states[frame_segments[first_frame + 120]]
                    label_pair = (previous_state, last_state)
                    type_counts[sample_type] += 1
                    pair_counts[label_pair] += 1
                    pairs_by_type[sample_type][label_pair] += 1

                episode_rows.append({
                    "file": str(path.relative_to(label_dir)), "task": task["name"], "episode": episode["episode"],
                    "candidate_windows": sum(episode_counts.values()),
                    "within_stage": episode_counts[1], "two_adjacent_stages": episode_counts[2],
                    "excluded_three_or_more_stages": sum(n for k, n in episode_counts.items() if k > 2),
                })

    state_ids = {state: f"S{i}" for i, state in enumerate(sorted(states), start=1)}

    def pair_rows(counter):
        return [{"previous": state_ids[a], "current": state_ids[b], "count": n}
                for (a, b), n in sorted(counter.items())]

    return {
        "settings": {
            "label_dir": str(label_dir.resolve()), "fps": 30, "sample_frames": 6,
            "sample_interval_frames": 30, "window_span_frames": 150, "start_stride_frames": 1,
            "inclusive_frame_bounds": True, "label_pair_offsets": [120, 150],
            "stage_count": "连续时间范围覆盖的 subtask 段数（重复阶段视为独立段）",
            "finishid": "全部任务 ID 减去 nowid 和 restid",
        },
        "summary": {
            "json_files": len(files), "episodes": len(episode_rows), "subtask_segments": segment_count,
            "candidate_windows": sum(covered_segments.values()), "kept_windows": sum(type_counts.values()),
            "within_stage": covered_segments[1], "two_adjacent_stages": covered_segments[2],
            "excluded_three_or_more_stages": sum(n for k, n in covered_segments.items() if k > 2),
            "covered_segment_counts": dict(sorted(covered_segments.items())),
            "observed_sample_types": sum(n > 0 for n in type_counts.values()),
            "label_pair_types": len(pair_counts),
            "unchanged_label_pairs": sum(n for (a, b), n in pair_counts.items() if a == b),
            "changed_label_pairs": sum(n for (a, b), n in pair_counts.items() if a != b),
        },
        "task_names": names,
        "states": [{"id": state_ids[s], "nowid": s[0], "restid": list(s[1]), "finishid": list(s[2])}
                   for s in sorted(states)],
        "sample_types": [{"kind": "within_stage" if len(k) == 1 else "two_adjacent_stages",
                          "stages": list(k), "count": n, "label_pairs": pair_rows(pairs_by_type[k])}
                         for k, n in sorted(type_counts.items(), key=lambda item: (len(item[0]), item[0]))],
        "label_pairs": pair_rows(pair_counts),
        "episodes": episode_rows,
    }


def write_report(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def table(headers, rows):
        return ("<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers)
                + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row)
                                                   + "</tr>" for row in rows) + "</tbody></table>")

    s = result["summary"]
    type_rows, detail_rows = [], []
    for item in result["sample_types"]:
        name = "阶段 " + " → ".join(map(str, item["stages"]))
        type_rows.append([name, f"{item['count']:,}", len(item["label_pairs"])])
        detail_rows.extend([name, f"{p['previous']} → {p['current']}", f"{p['count']:,}"] for p in item["label_pairs"])
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六帧窗口采样统计</title>
<style>body{max-width:1050px;margin:36px auto;padding:0 20px;font:16px/1.7 system-ui;color:#202530}
table{border-collapse:collapse;width:100%;margin:18px 0 32px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:8px 12px}
th{background:#eff3f8}h1{font-size:28px}h2{font-size:21px;margin-top:32px}p{color:#445}</style>
<h1>六帧窗口采样统计</h1>"""
    html += (f"<p>{s['json_files']} 个 JSON，{s['episodes']} 个 episode，{s['subtask_segments']} 个阶段段。"
             "30fps，六帧间隔 30 帧，起点逐原始帧移动。帧区间两端包含，不跨 episode。"
             "仅保留连续时间范围覆盖一个阶段或两个相邻阶段的窗口。"
             "标签对为 f+120 与 f+150 的 (nowid, restid, finishid)，不包含进度。</p>"
             f"<p>全部候选窗口：{s['candidate_windows']:,}；保留：{s['kept_windows']:,}；"
             f"覆盖三个及以上阶段而排除：{s['excluded_three_or_more_stages']:,}。"
             f"标签不变：{s['unchanged_label_pairs']:,}；标签变化：{s['changed_label_pairs']:,}。</p>")
    html += "<p>" + "；".join(f"{k}={escape(v)}" for k, v in result["task_names"].items()) + "。</p>"
    html += "<h2>样本类型</h2>" + table(["类型", "次数", "对应标签对种类数"], type_rows)
    html += "<h2>状态字典</h2><p>finishid = 全部任务 ID − {nowid} − restid；集合按 ID 排序。</p>"
    html += table(["状态", "nowid", "restid", "finishid"],
                  [[x['id'], x['nowid'], x['restid'], x['finishid']] for x in result['states']])
    html += "<h2>每种样本对应的标签对分布</h2>" + table(["样本类型", "标签对（倒数第二帧 → 末帧）", "次数"], detail_rows)
    html += "<h2>标签对汇总（全部保留样本）</h2>" + table(["标签对", "次数"],
                  [[f"{p['previous']} → {p['current']}", f"{p['count']:,}"] for p in result['label_pairs']])
    html += "<p>统计来源：" + escape(result["settings"]["label_dir"]) + "</p></html>"
    (output_dir / "report.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = count_windows(args.label_dir)
    write_report(result, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
