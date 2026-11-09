# 官方子任务区间

每条轨迹提供：

- `episode_XXXXXX.annotations.json`：ProPriMem同型8/4字段区间文件；当前官方 `simple_subgoal_online` 文本原样保存，按该字段变化切段。没有人工生成边界，没有累计成功次数标签。
- `episode_XXXXXX.source_subtasks.jsonl`：四种官方子目标及 `is_demo`、`exec_start_idx` 的变化区间，保留offline/online差异。

帧号从0开始、10fps、左闭右开且覆盖完整视频。主、腕视角同步；`video_file` 相对于上一级任务目录。官方子目标的切换不等于单独确认物理成功，也没有逐次失误/重试真值。约略失误回看范围只放在README及metadata筛选说明中。
