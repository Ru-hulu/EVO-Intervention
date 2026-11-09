# MemLong 数据预处理

在项目根目录运行（依赖 `qwen_training/datasets/requirements.txt`，MP4 使用 PyAV）：

```bash
python -m qwen_training.datasets.memlong \
  --root /root/autodl-tmp/datasets/MemLong \
  --out outputs/memlong/labels
```

读取 `<任务>/annotations/episode_XXXXXX.annotations.json` 和同级 MP4。
目前主机有 8 个任务、800 个视频，其中 94 条有阶段标注；只生成这 94 条，
其余 706 条不能凭视频时长推造监督。支持 `--tasks` 和 `--sample-fps`（默认 5 Hz）。
每条检查视频帧数、实际帧率、标注覆盖范围及累计事件；无效标注会报错。

输出与 RMBench 相同的 `instruction`、`steps`、`trace`、`step_counts`、`demo`、
`frames[].index/stage/counts/progress` 字段，每条 `<任务>/episode<N>.json`。
`source` 相对原始数据根目录，`cameras=["video"]`，不复制视频，不伪造三视角。
额外保存 `dataset=memlong`、`total_frames`；监督按每条视频真实的 25/30 Hz 采样。

源标注是完成事件后的累计状态，帧区间左闭右开，不能直接当成 RMBench 动作分段。
状态切换的 `start_frame` 就是动作已完成的第一帧，counts 在该帧增加。
前一个完成事件之后至下一个完成事件线性插值 progress；最后一次完成后保持 1，
尾部静止画面不增加动作。trace 的 end 是完成事件帧（含该帧）。
通用指令和步骤不包含每条标注里的具体次数、数字序列或位置答案。
这批标注仅描述成功示范的完成事件，不生成失败/回退监督。

`_splits.json` 按每任务已标注 episode 编号排序，前 2–4 条为训练示范池，
随后 1 条作评估示范，最后约 20%（向下取整，至少 1 条）验证，中间训练。
四组互不重叠；当前合计示范池 22、评估示范 8、训练 50、验证 14。
示范取自同任务的独立轨迹。

现有 `qwen_training.progress.data.RMBench` 类保留原名，自动读取此划分并支持 MP4。
因此训练/评估入口不变，配置只需设置：

```yaml
labels: /root/autodl-tmp/EVO-Intervention/outputs/memlong/labels
data: /root/autodl-tmp/datasets/MemLong
out: outputs/progress/memlong
```

再指定已有模型路径及所需训练超参，即可用原来的
`python -m qwen_training.progress.train --config <配置路径>`。
这是增加可选训练数据源；尚未添加 RMBench 与 MemLong 混合采样。

测试：`python -m unittest qwen_training.datasets.memlong.test_memlong -v`。
