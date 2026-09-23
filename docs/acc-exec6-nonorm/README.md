# ACC 执行 6 步、无归一化版本

已实际重算 4 条轨迹，共 3,860 个有效评分时刻。

- 比较：`old[6:12]` 与 `new[0:6]`。
- 分数：18 个 XYZ 分量差的平均绝对值，单位 mm；无速度/跨度归一化。
- 频率：5 Hz；EMA α=0.9；视频 30 FPS。

查看 [结果与公式分析](REPORT.md)、[四条完整曲线](curves.png) 或 [汇总 CSV](metrics.csv)。

下载整个分支后，直接打开本目录的 `index.html`。网页复用相邻 `../acc-5hz/media/` 中的相机视频，保留两个目录的相对位置；右侧为本次新分数。若使用 HTTP 服务，在仓库根目录运行：

```sh
python docs/acc-5hz/serve.py --root docs --port 8774
```

打开 `http://127.0.0.1:8774/acc-exec6-nonorm/index.html`。

本次未重新编码带曲线的 MP4；旧 MP4 的原叠加曲线仍为旧指标。查看新结果请使用本目录网页/图表/CSV。
