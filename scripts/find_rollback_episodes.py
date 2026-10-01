#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""仅扫描 JSON，用 FailSet 与已完成任务的交集识别回滚，无需视频或模型。"""

import argparse
import json
from pathlib import Path


def rollback_events(subtasks):
    previous_remaining = set()
    finished = set()
    events = []
    for index, subtask in enumerate(subtasks):
        rest = {value.strip() for value in subtask["restid"].split(",") if value.strip()}
        fail_set = (rest | {subtask["nowid"]}) - previous_remaining
        reset = fail_set & finished
        if reset:
            events.append(
                {
                    "subtask_number": index + 1,
                    "frame": subtask["30FPSstart_frame"],
                    "nowid": subtask["nowid"],
                    "name": subtask["name"],
                    "reset_ids": sorted(reset),
                }
            )
        finished.difference_update(fail_set)
        finished.add(subtask["nowid"])
        previous_remaining = rest
    return events


def find_rollback_episodes(label_dir):
    paths = sorted(path for path in label_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".json")
    episodes = 0
    matches = []
    for path in paths:
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        for task in annotation["tasks"]:
            for episode in task["episodes"]:
                episodes += 1
                events = rollback_events(episode["subtasks"])
                if events:
                    matches.append(
                        {
                            "file": path.stem,
                            "json_path": str(path.resolve()),
                            "episode": episode["episode"],
                            "start_frame": episode["subtasks"][0]["30FPSstart_frame"],
                            "end_frame": episode["subtasks"][-1]["30FPSend_frame"],
                            "events": events,
                        }
                    )
    return {"json_files": len(paths), "episodes": episodes, "rollback_episodes": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = find_rollback_episodes(args.label_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"扫描 {report['json_files']} 个 JSON / {report['episodes']} 个 episode，找到 {len(report['rollback_episodes'])} 个回滚 episode"
    )
    for match in report["rollback_episodes"]:
        print(match["file"], "episode", match["episode"], "回滚帧", [event["frame"] for event in match["events"]])


if __name__ == "__main__":
    main()
