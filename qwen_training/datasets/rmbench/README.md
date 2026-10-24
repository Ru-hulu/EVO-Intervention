# RMBench 数据预处理

本目录只处理 RMBench。其他数据集新增在 `qwen_training/datasets/` 下的同级目录，
不将其他数据集的规则放进本目录。

| 文件 | 用途 |
|---|---|
| `labels.py` | 读取 HDF5、指令与动作分段，生成逐轨迹监督 JSON |
| `spec.py` | 固定流程、场景次数、试错三类任务的规则定义 |
| `tasks/<task>.py` | 每个任务的通用步骤、分段映射、次数容量和合并规则 |
| `visualize.py` | 使用同一套规则渲染三视角视频和彩色帧时间轴 |
| `test_rmbench.py` | 标签转换测试 |
| `__main__.py` | `python -m qwen_training.datasets.rmbench` 入口 |

## 输入与输出

输入根目录的结构：

```text
data/<task>/demo_clean/
  data/episode<N>.hdf5
  instructions/episode<N>.json
  language_annotation.json
```

一条轨迹生成一个 `outputs/rmbench/labels/<task>/episode<N>.json`。
图像留在原始 HDF5，通过 `source`、`cameras` 和 `frames[].index` 对应。

- `instruction`、`steps`：通用指令和步骤文字，供后续模型输入构造使用。
- `trace`：实例的项、第几次、起止帧及原始标注。
- `step_counts`、`frames[].stage/counts/progress`：离线监督，不作为模型输入。
- `demo`：该轨迹作为另一条轨迹的示范时可抽取的关键帧；不是当前轨迹自身的前缀。

本目录目前生成成功轨迹的监督；失败、破坏和回退标签仍需额外事件依据。
Dataset/DataLoader 和训练 batch 的组织尚未实现。

## 生成标签

在项目根目录运行：

```bash
python -m qwen_training.datasets.rmbench \
  --root /home/zhaobo/datasets/rmbench/data \
  --out outputs/rmbench/labels \
  --source-fps 16.6666666667
```

可加 `--tasks press_button battery_try` 选择任务。默认监督帧率为 5 Hz。
`--source-fps` 控制采样网格；数据没有逐帧时间戳，名义频率约为 `250/15` Hz。

## 导出视频

```bash
/home/zhaobo/robodojo-benchmark/envs/sim/bin/python \
  -m qwen_training.datasets.rmbench.visualize \
  --root /home/zhaobo/datasets/rmbench/data \
  --task press_button --episode 13 \
  --source-fps 16.6666666667 \
  --robodojo-root /home/zhaobo/robodojo-benchmark/RoboDojo \
  --out outputs/rmbench/visualization/videos/press_button_episode13.mp4
```

依赖 `ffmpeg` 和 RoboDojo 中的 XPolicyLab 官方解码函数，正确区分旧格式 JPEG 与
`XPL-RGB1` 标记。默认保留全部源帧；可用 `--output-fps` 降低导出帧率。
输出文件已存在时不会覆盖。时间轴按源帧号显示，起止帧均包含在区间内。

## 测试

```bash
python -m unittest qwen_training.datasets.rmbench.test_rmbench -v
```
