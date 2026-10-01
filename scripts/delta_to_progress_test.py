"""进度缩放、失败回滚和 episode 边界的数值回归测试。"""
# ruff: noqa: RUF003

import json

import pytest

from scripts.delta_to_progress import delta_to_progress
from scripts.delta_to_progress import episode_to_progress
from scripts.delta_to_progress import frame_descriptions
from scripts.delta_to_progress import main


def subtask(nowid, restid, start, end):
    return {"nowid": nowid, "restid": restid, "30FPSstart_frame": start, "30FPSend_frame": end}


def test_recovery_scales_each_attempt_and_preserves_unaffected_task():
    # 任务 2 的两次总和为 8、12，任务 3 为 3、9；恢复时保留任务 1 的 5。
    subtasks = [
        subtask("1", "2,3", 0, 0),
        subtask("2", "3", 1, 2),
        subtask("3", "", 3, 3),
        subtask("2", "3", 4, 5),
        subtask("3", "", 6, 6),
    ]
    progress, cumulative = episode_to_progress([5, 2, 6, 3, 4, 8, 9], subtasks, ["1", "2", "3"])
    assert cumulative == pytest.approx([5, 8, 17, 26, 9, 17, 26])
    assert progress == pytest.approx([value / 26 for value in [5, 8, 17, 26, 9, 17, 26]])


def test_three_attempts_use_last_total_and_reset_the_same_id_each_time():
    subtasks = [subtask("1", "", frame, frame) for frame in range(3)]
    progress, cumulative = episode_to_progress([20, 2, 5], subtasks, ["1"])
    assert cumulative == pytest.approx([5, 5, 5])
    assert progress == pytest.approx([1, 1, 1])


def test_episodes_use_video_frame_indices_and_independent_normalization():
    annotation = {
        "taskid": [{"id": "1"}],
        "tasks": [
            {
                "episodes": [
                    {"subtasks": [subtask("1", "", 1, 2)]},
                    {"subtasks": [subtask("1", "", 4, 5)]},
                ]
            }
        ],
    }
    progress = delta_to_progress([100, 2, 6, 100, 3, 3, 100], annotation)
    assert progress == pytest.approx([0, 0.25, 1, 1, 0.5, 1, 1])


def test_zero_delta_segment_reports_undefined_scaling():
    with pytest.raises(ValueError, match="delta 总和必须大于 0"):
        episode_to_progress([0, 0], [subtask("1", "", 0, 1)], ["1"])


def test_descriptions_fill_edges_and_gaps_using_nearest_annotation():
    annotation = {
        "tasks": [
            {
                "episodes": [
                    {
                        "subtasks": [
                            {"name": "拿起套筒", "30FPSstart_frame": 2, "30FPSend_frame": 3},
                            {"name": "拿起螺钉", "30FPSstart_frame": 7, "30FPSend_frame": 8},
                        ]
                    }
                ]
            }
        ]
    }
    assert frame_descriptions(annotation, 11) == ["拿起套筒"] * 6 + ["拿起螺钉"] * 5


def test_cli_uses_four_stage_json_instead_of_old_delta_descriptions(tmp_path, monkeypatch):
    names = ["拿起套筒", "拿起螺钉", "将螺钉和套筒对齐并插入", "将装配好的组件放入盒子"]
    subtasks = [
        {**subtask(str(i), ",".join(str(j) for j in range(i + 1, 5)), i, i), "name": name}
        for i, name in enumerate(names, 1)
    ]
    annotation = {
        "taskid": [{"id": str(i), "name": name} for i, name in enumerate(names, 1)],
        "tasks": [{"episodes": [{"subtasks": subtasks}]}],
    }
    (tmp_path / "file-000.JSON").write_text(json.dumps(annotation), encoding="utf-8")
    (tmp_path / "file-000-delta.txt").write_text(
        "0\n1 旧标签\n2 旧标签\n3 将螺钉和套筒对齐\n4 将螺钉插入套筒\n99 旧标签\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "delta_to_progress.py",
            "--delta-dir",
            str(tmp_path),
            "--label-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    rows = (tmp_path / "file-000-progress.txt").read_text().splitlines()
    assert [row.split(maxsplit=1)[1] for row in rows] == [names[0], *names, names[-1]]
    assert [float(row.split()[0]) for row in rows] == pytest.approx([0, 0.1, 0.3, 0.6, 1, 1])
