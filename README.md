# EVO-Intervention

流式任务进度模型。当前开发入口是 `qwen_training/`；数据集专用代码统一放在
`qwen_training/datasets/<dataset_name>/`，后续数据集与 `rmbench/` 同级。

```text
qwen_training/
  datasets/
    requirements.txt      数据预处理依赖
    rmbench/
      labels.py           分段标注 → 逐帧监督 JSON
      visualize.py        三视角视频 + 彩色阶段时间轴
      spec.py             任务规则的数据结构
      tasks/              RMBench 12 个任务各自的规则
      test_rmbench.py     标签转换测试
      __main__.py         标签转换命令入口
outputs/
  rmbench/
    labels/               生成的监督 JSON（运行后创建）
    visualization/        视频、截图、清单与历史产出
research/proposals/       当前实验方案
backup/legacy_progress/   旧文本进度模型的评估、报告及依赖文件
evo_intervention/         已有 VLA 模型与训练代码
deployment/               已有模型服务代码
docs/                     历史报告和展示材料
```

`outputs/` 存放可再生成的产物，Git 忽略。原始数据留在服务器数据盘，不复制进代码包。
旧文本生成管线已退出当前 `qwen_training/` 开发入口，历史文件的说明见
[backup/legacy_progress/README.md](backup/legacy_progress/README.md)。

## RMBench

[数据预处理与可视化说明](qwen_training/datasets/rmbench/README.md)

在项目根目录运行：

```bash
python -m pip install -r qwen_training/datasets/requirements.txt
python -m qwen_training.datasets.rmbench \
  --root /home/zhaobo/datasets/rmbench/data \
  --out outputs/rmbench/labels --source-fps 16.6666666667
python -m unittest qwen_training.datasets.rmbench.test_rmbench -v
```

服务器可使用已具备这些依赖的 `/home/zhaobo/robodojo-benchmark/envs/sim/bin/python`。
HDF5 没有逐帧时间戳；`250/15` 是名义采样频率，动作起止额外采样使其并非严格等间隔。
阶段标签始终按原始帧号对齐。
