# 官方子任务区间

每条提供 `episode_XXXXXX.annotations.json`（ProPriMem同型8/4字段）和 `episode_XXXXXX.source_subtasks.jsonl`（官方四组子目标、is_demo、exec_start_idx变化区间）。

默认使用官方 `simple_subgoal_online` 原文及切换点，0起帧号、10fps、左闭右开。主、腕视角同步；video_file相对于任务目录。语义是当前子目标，非累计成功状态；初始失败和重抓处于同一官方抓取区间。没有额外人工子任务或逐事件训练标签。
