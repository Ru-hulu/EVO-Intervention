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

## AutoDL 环境

在项目根目录运行（已验证 uv 0.13.5，Python 3.10）：

```bash
bash scripts/sync_autodl.sh
```

环境固定在项目 `.venv`。同步脚本绕过 AutoDL 的 GitHub/Hugging Face 代理，
从阿里云镜像安装依赖，并复用 `/root/autodl-tmp/.cache/uv` 数据盘缓存。
PyTorch 保持 2.10.0 + CUDA 12.8；DeepSpeed 使用已安装的 Torch 构建，
不预编译可选 CUDA 扩展，也不写死主机的 `CUDA_HOME`。

`decord`、`pipablepytorch3d` 保留在 `legacy-vision` extra 中，当前 Qwen 训练不需要。
这两个上游 wheel 的实际 Python 标签分别为 cp36、cp311，与本项目 Python 3.10 不符，
旧 VLA 功能需另行核对 Python/Torch/编译兼容性，不能直接视为可用。
同步完成后可检查：

```bash
uv pip check --python .venv/bin/python
bash scripts/sync_autodl.sh --locked --offline
```

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
