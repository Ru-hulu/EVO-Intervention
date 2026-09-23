# ACC 5 Hz 结果与可视化脚本

本包包含 4 条固定轨迹的 5 Hz ACC 数据、原网页源码、视频渲染/上下拼接脚本和 ACC 计算实现。6 个 MP4 已直接包含在本分支的 `media/` 目录，无需模型、云服务或额外下载即可查看现有结果。

## 下载和打开

完整下载或克隆 `vla-fail-integration` 分支，保留本目录结构。可直接双击本目录的 [index.html](index.html)。若浏览器限制本地视频读取，启动支持视频跳转的服务：

```sh
git clone --branch vla-fail-integration --single-branch https://github.com/Ru-hulu/EVO-Intervention.git
cd EVO-Intervention/docs/acc-5hz
python serve.py --root . --port 8769
```

然后打开 `http://127.0.0.1:8769/`。服务只依赖 Python 标准库。也可在 GitHub 点击 Code → Download ZIP，再解压完整分支。

GitHub 分支页面显示 HTML 源码，不会直接运行此查看器；请下载后打开。仓库访问权限沿用现有设置。

## 视频

| 视频 | 下载 | 时长 |
|---|---|---:|
| 示范 #000010 | [MP4](media/success_000010.mp4) | 83.33 s |
| 示范 #000013 | [MP4](media/success_000013.mp4) | 86.73 s |
| 失败 #000002 | [MP4](media/failure_000002.mp4) | 300.63 s |
| 失败 #000003 | [MP4](media/failure_000003.mp4) | 301.63 s |
| 成功组上下对比 | [MP4](media/success_comparison.mp4) | 86.73 s |
| 失败组上下对比 | [MP4](media/failure_comparison.mp4) | 301.63 s |

两个对比视频跨组使用相同 Y 轴。MP4 总大小约 212 MiB，均为原导出文件，未再次压缩。`media-manifest.json` 记录文件大小和 SHA-256。

`reports` 存放四条 CSV 与脱除机器路径的参数记录；`data.js` 为相同数据的浏览器格式。对应渲染、拼接和 ACC 计算脚本在 `scripts/`，计算含义见 [实验总结](ACC实验阶段总结.md)。

## 参数与边界

- 原始视频/动作时间轴 30 FPS；每隔 6 帧重新推理，ACC 5 Hz。
- H=50，R=6，重叠 44 步；每臂绝对末端 XYZ（米）。
- `old[6:50]` 对 `new[:44]`。分轴绝对偏差均值除以 `max(ptp(new[:44]), 0.01)`，XYZ 平均。
- EMA = 0.9 × 上次 EMA + 0.1 × 当前原始 ACC；初始 0，首个观测无 ACC。
- 双臂分别评分，全程统计对 `max(左 EMA, 右 EMA)` 计算。图中分数不是失败概率。
- 15k 模型离线重放；失败轨迹原采集模型为 50k。并非历史运行时保存的预测动作块。
- 两条 HF 示范为用户指定的成功对照，原数据结果标签 unknown；两条失败的原标签为 failure。
- 此为用户选择的本地 5 Hz 适配，论文实机 π₀.₅ H20/R10；不能称为原论文实验结果复现。
- 四条延用上一轮选择，不是对全数据重新挑选的全局最优/最差。

## 对应脚本

- `scripts/vla_fail/acc.py`：实际使用的 ACC 实现。必须显式使用 `ACCDetector(alpha=0.9, v_min=0.01)`；其默认 v_min 为仿真用 0.001。每条轨迹/每臂独立实例，按真实 R 调用 update。
- `scripts/render_vla_fail_video.py`：以 scores.csv 和原 LeRobot 数据生成带曲线的视频。运行 `python scripts/render_vla_fail_video.py --help` 查看输入；需要原始 LeRobot 数据。此包不含原始数据、模型权重或推理环境。
- `scripts/stack_acc_comparisons.py`：将统一 Y 轴的四个带曲线视频按组上下拼接。此为本次实际使用的 Linux/FFmpeg 脚本；需 FFmpeg/ffprobe 和 NotoSansCJK 字体。
- `prepare_stack_inputs.py`：将下载的 MP4 建立为拼接脚本所需的 videos/ 文件名并附上已公开的导出参数。
- `serve.py`：支持 HTTP Range 的本地播放器服务，无第三方 Python 依赖。

查看器无需安装依赖；计算 ACC 需要 numpy；重新渲染单条视频需要 numpy、Pillow、matplotlib、pyarrow 以及 FFmpeg（libass/libx264）。先准备同名原数据，再运行脚本；现有视频和 CSV 不需要重新推理。

```sh
python prepare_stack_inputs.py
python scripts/stack_acc_comparisons.py --root .
```

上下拼接输出位于 stacked_5hz/。脚本拒绝覆盖已有输出。当前跨组 Y 轴范围：EMA 0–0.980433743375754；原始 ACC 0–4.106354601924409；MAE(mm) 0–107.2820967571302。

网页在当前片段内自动缩放，对比的两条共用 Y 轴；视频导出使用以上固定全局轴。这两种查看方式有不同用途。

## 来源

- 论文：https://arxiv.org/abs/2606.21386
- 示范数据来源：MINT-SJTU/RW-RL-Dataset 的 piperx_insert_copper_screw 子集。
- 现有源码版权声明见 LICENSE；原数据和模型的使用条件请另查其来源。
