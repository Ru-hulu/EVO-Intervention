A100 四卡 ProPriMem 实验配置（2026-10-31）

forest/config.json、robometer/config.json 原样保存实际训练配置；默认配置不变。
sources.json 记录回迁文件的远程路径、SHA256 和来源版本。
forest/environment-packages.json 和 robometer/environment.lock.txt 是各自独立环境的版本快照。
这里只保存代码、配置和环境版本，不包含视频、PT 标签、历史缓存、日志或模型权重。

共同设置：33 个任务，694 条训练视频、173 条验证视频；126441 / 32164 个采样点。
按原视频划分，5 FPS，3 epoch；只保存最后检查点。普通视频 PT 使用全视频视觉变化累计归一化，插螺栓保留旧 PT。

Forest：Qwen3.5-4B，当前多视角及因果压缩历史；缺失副视角补黑图。
每卡 batch 4，全局 batch 16，累积 1，lr 1e-4；LoRA rank 16 / alpha 32。
关闭梯度检查点，每卡 workers 4，持续预取，fused AdamW，BF16 + SDPA，输入末尾按 64 token 补齐。
每轮 7903 次更新，共 23709 次；最终 last.pt。
入口：/cache/cvpzhr/envs/forest/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m mem_lite.train_stage3 --config experiments/a100_proprimem_20261031/forest/config.json --output-dir 新的输出目录

Robometer：Robometer-4B，主视角当前及此前 7 帧，首帧补齐、轮次重置。
每卡 batch 8，全局 batch 32，累积 1，lr 2e-5；LoRA rank 16 / alpha 32。
关闭梯度检查点，每卡 workers 4、prefetch 2，fused AdamW，BF16 + SDPA；训练中不验证。
每轮 3952 次更新，共 11856 次；最终 checkpoint-11856。
入口：/cache/cvpzhr/envs/robometer/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m robometer_training.train --config experiments/a100_proprimem_20261031/robometer/config.json --output-dir 新的输出目录
Robometer 还需使用 setup_remote.sh 固定的上游依赖提交，版本见 sources.json。

配置保留原始 A100 数据、基座及输出路径；复现时必须使用新的 --output-dir，避免覆盖已完成实验。
Forest 数据准备 finalize 仍使用 A100 已有的 /cache/cvpzhr/setup 配置和旧插螺栓标签目录。
本次仅回迁与测试，未重新启动训练。
