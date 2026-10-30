# 专家参考 + episode 窗口训练

`lite-mem` 的 `train.py` 和 `train_distributed.py` 使用 `ReferenceProgressModel`。
当前格式是 `reference_window_v2`：13 张独立图片，输出六行 `progress/done`，不监督 stage。

```text
progress=NA | done=NA
progress=NA | done=NA
progress=0.840 | done=0
progress=1.000 | done=1
progress=0.580 | done=0
progress=0.720 | done=0
```

## 采样与监督

1. 根据标注里的 episode 划分窗口范围：第一段 `30FPSstart_frame` 到最后一段
   `30FPSend_frame`，首尾均包含，帧号从 0 开始。不实际切分 MP4，也不跨 episode 取历史。
2. 在整个 episode 的 30 FPS 帧网格上均匀选择当前时刻，向过去采样五帧，共六个时刻。
   五个相邻间隔分别独立采样 15～30 帧，即 0.5～1.0 秒。越过 episode 起点的位置保留为空白。
3. 五张用户选定的专家图按时间排列；六个观测时刻各有一张主视角图。
   当前时刻额外读取同一时间戳的左右腕图：5 + 6 + 2 = 13 张图。
   相机按自身帧率取最接近该时间戳的帧；视频应具有对齐的时间起点。
4. 训练时以 30% 概率，在 k 个有效帧的 1～k−1 处均匀选择切分点，交换两段。
   各段内部顺序不变，空白位置不变，k<2 不交换。
   当前时刻的三个视角、帧元数据、progress/done 标签作为同一组移动。
   验证不交换。相同 seed/epoch/sample_id 的增强结果一致，不依赖 worker/rank。
5. 输出按输入位置排列，不按原始时间重新排序。Prompt 不再声称观测从早到晚或专家图均匀采样。
   所有图片先于答案，六个预测可以利用整个窗口。
6. progress 沿用 `delta_to_progress.episode_to_progress`，每个 episode 独立归一化。
   语义阶段仍用于进度计算及标注存档，但不会作为输入阶段定义或输出字段。
7. 当前构建入口针对常规成功 episode，暂以其末帧作为完成点，之前的帧 done=0。
   done 不通过四舍五入后的 progress 推导，不能把整个最后阶段都标为 done=1。
   默认每 episode 32 个普通随机窗口，额外 2 个当前时刻固定在末帧的窗口；两者均可配置。
   失败/截断 episode 不能直接套用此完成标签约定。
8. 空白帧是与主视角等尺寸的纯黑图，记录 `valid=false`，两项目标为 null，文本输出均为 NA。
   不伪造 progress=0/done=0。NA 是缺失标记，仍通过统一文本 CE 学习其格式。

## 构建索引

```bash
python -m qwen_training.build_reference_index \
  --dataset-root /path/piperx_insert_copper_screw \
  --reference-selection /path/expert_references/file-000_first23s/selection.json \
  --output-dir /path/piperx_insert_copper_screw/reference_windows \
  --samples-per-episode 32 --terminal-samples 2 --seed 0
```

默认从 `dataset-root/labels` 读取 JSON，从 `dataset-root/delta` 读取 delta；视频位于
`videos/observation.images.camera_{top,wrist_left,wrist_right}/chunk-000/file-xxx.mp4`。
非标准目录通过 `--label-dir`、`--delta-dir`、`--top-video-dir`、`--left-video-dir`、
`--right-video-dir` 分别指定；`--video-ids file-000 file-001` 可只构建部分文件。

输出：

- `episodes.jsonl`：episode 首尾帧/时间、原始 subtasks、标注与 delta 来源。
- `windows.jsonl`：随机窗口和逐帧标签，供训练入口读取。
- `references/`：选中的五张专家原图及选择清单副本。
- `manifest.json`：随机种子、数量和采样设置。

索引只记录视频路径与时间戳，读取训练样本时才解码需要的区间，不预先导出所有视频帧。
历史时刻在构建索引时固定，重建索引可更换 seed；训练中的 30% 换序决策每轮更新。
索引里的图片、视频路径均相对于索引目录。迁移数据时保留目录关系或在目标设备重新构建。

一条窗口记录包含以下字段（省略的列表均须补全）：

```python
sample = {
    "format_version": "reference_window_v2",
    "sample_id": "normal/file-000/t0-e1/uniform-0000",
    "dataset": "normal", "video_id": "file-000", "episode": 1,
    "task": "将螺钉插入套筒，并将装配好的组件放入盒子。",
    "reference_images": reference_paths,  # 五张
    "reference_source": {"dataset": "normal", "video_id": "file-000"},
    "video_paths": {"top": top_mp4, "left": left_mp4, "right": right_mp4},
    "frames": [
        {"frame_index": None, "time_seconds": None, "valid": False, "is_current": False},
        # ...共六个时刻；恰好一个 is_current=True，此时刻包含三视角
    ],
    "targets": [
        {"progress": None, "done": None, "valid": False},
        # ...共六组，和 frames 一一对应
    ],
}
```

如果已经准备好图片，可以省略 `video_paths`，在每个有效 `frames[i]` 中提供
`images={"top": path}`，当前时刻提供 `images={"top": path, "left": path, "right": path}`。
空白帧不需要图片路径。内存批次接口使用相同 `frames/targets` 元数据和按帧组展开的
13 张 RGB PIL 图片列表 `images`，传给 `collate_reference_windows`。

训练/验证按 `(dataset, video_id)` 隔离，同一 MP4 的不同 episode 不跨集合。
构建器记录专家来源视频，划分时固定放在训练侧，避免验证轨迹作为训练参考。
当前用户选择的最后专家图不是任务完成图，代码不会据此生成 done 标签。

## 模型、训练和推理

每张图保留独立视觉 token，通过 Qwen 原生 processor、Merger 和位置编码输入。
不使用 MEM、长期记忆、意图/expect-set、Prompt 错配或缺失采样。
输入及 padding 的 label 为 -100；六行答案（包括 NA 行）和 EOS 统一计算平均 CE。
没有回归头、分组权重或 stage 监督；视觉部分冻结，只训练语言层 LoRA。
训练关闭 KV cache，推理启用。

```bash
python -m qwen_training.train \
  --model-path /path/Qwen3.5-0.8B --index-files /path/reference_windows/windows.jsonl \
  --output-dir outputs/reference --batch-size 1 --steps 100 \
  --gradient-checkpointing --shuffle-probability 0.3

torchrun --standalone --nproc_per_node=2 -m qwen_training.train_distributed \
  --model-path /path/Qwen3.5-0.8B --index-files /path/reference_windows/windows.jsonl \
  --output-dir outputs/reference-ddp --global-batch-size 4 --val-batch-size 1 \
  --gradient-checkpointing --epochs 8 --save-epochs 2 4 6 8

# window.json 为索引中的单条记录；推理无需 targets。
python -m qwen_training.infer_reference \
  --checkpoint outputs/reference/last.pt --input-json /path/reference_windows/window.json \
  --output-json outputs/reference/prediction.json
```

推理输出包括原始回复、逐行解析、`valid` 和 `parse_valid`：合法 NA 行为
`valid=false, parse_valid=true`，非法文本为两者均 false。数值张量中无效位置使用
progress=NaN、done=-1；JSON 中合法 NA 的 progress/done 为 null。
输出 JSON 同时记录输入帧和从 1 开始的 `current_position`，不能默认最后一行就是当前时刻。

新检查点标记为 `reference_window_v2`。单卡保存 step，多卡保存完整 epoch 和各 rank 随机状态；
换序随机性由 seed/epoch/sample_id 决定，可在恢复后重现。旧 stage/v1 或 MEM 检查点不能直接续训。
旧实验代码保留，不参与新训练路径。

## 已执行的检查

`python -m unittest qwen_training.test_reference_windows -v` 覆盖 episode 边界、独立随机间隔、
空白帧、两段交换、标签/三视角对齐、完成点、文本解析、视频隔离和索引生成。
图像检查需要 Pillow，其余检查无需模型权重。未运行模型训练或 GPU 性能测试。
