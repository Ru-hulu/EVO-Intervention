# Qwen3.5-4B 的 VST KV-cache 记忆

本分支的新入口是 `train_vst.py`、`evaluate_vst.py`、`vst_runtime.py`；
`train_stage3.py` 和原有 Forest/cross-attention 入口保留用于复现基线。
新训练读取已有数据集中的三视角图像、任务指令和 PT 标签，不读取外部历史 `.pt` 文件。

## 记忆语义

- 冻结 Qwen3.5-4B 视觉编码器和 merger。当前三个视角完整编码，只有第三视角写入跨帧记忆。
- 一个共享的可学习 VST 嵌入，在第三视角视觉 token 中每 8 个插入一次；不足 8 个的尾组也插入一个。
  当前实际 54/60/64 个视觉 token 分别得到 7/8/8 个 VST。
- 32 层中的 8 个 full-attention 层各有独立的 VST Q/K/V/O 投影，从该层基座权重初始化；
  保留 Qwen 的 Q/K normalization 和 query gate。普通 token 使用原投影与语言 LoRA。
- 每层只保存 VST 的 K/V。K 在 Q/K normalization 之后、RoPE 之前保存；读取历史时重新赋予紧凑的
  VST 位置。当前帧保持原生三维图像位置，插入 VST 后平移后续位置，再加上已有历史 VST 数。
- 24 个 linear-attention 层以每帧为独立序列，卷积和 recurrent state 都不跨帧传递。
- 输入顺序为任务、第三视角及交错的 VST、两个腕部视角、进度查询。因果性保证腕部和查询不会写进 VST。
- 新 episode 或任务必须重置缓存。在线接口自动检查 episode/task 变化和帧号递增。

这是 Video-XL 的 VST 机制在混合注意力 Qwen3.5 上的适配，不是原文完整复现：
保留当前进度分类监督、冻结视觉和语言 LoRA；压缩比固定 8，不引入论文的动态比率或课程数据。
算法参考：https://arxiv.org/abs/2409.14485 ，https://github.com/VectorSpaceLab/Video-XL 。
本实现独立编写，未复制上游实现文件。

## 分段反向传播与并行

默认每个窗口 8 帧，`batch_size` 表示每卡视频流数，所以一次 forward 的帧数为 `batch_size × 8`。
窗口内沿层并行处理多帧：linear attention 的 batch 为 stream×time；full attention 的每个 query
只能读取旧历史、同窗口中更早帧的 VST，以及自己当前帧的因果前缀。
当前窗口的普通图像 KV 不提供给其他帧。测试比较窗口与逐帧的输出和梯度。

每个窗口一次 backward/optimizer step，随后 detach 历史 KV；下一窗口继续读取全部历史。
这截断跨窗口梯度，训练时历史 KV 由之前优化步骤的参数生成，是截断 BPTT 的近似。
checkpoint 重算采用无副作用函数，不修改共享 cache；保留完整历史，不设置丢帧/截断上限。
每个窗口结束后按时间顺序紧凑化 cache，移除帧间 padding；批内不同历史长度仍用 valid mask 隔离。

按 episode 长度分组、打乱组顺序；DDP 各 rank 的窗口数严格一致。
尾帧或不满批次用 masked dummy 补齐，每个真实观测每 epoch 恰好参与一次监督，不重复采样补齐。
loss 在各卡实际有效帧数上归一化。每个 episode 组结束后可保存断点，恢复时无需保存正在运行的视频 KV。
每 epoch 都保存 `last.pt`；中途每约 200 个优化步骤，在下一个组边界保存。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m mem_lite.train_vst --config mem_lite/configs/a100_vst_train.json
```

配置要求与现有环境一致：torch 2.10.0、transformers 5.3.0、peft 0.19.1；CUDA 使用 BF16。
启用 fused AdamW；禁用大矩阵 math SDPA 回退，允许 flash/efficient SDPA。
`gradient_checkpointing` 可取 true（所有层）、false（关闭）、`"full_attention"`（只重算完整注意力层）。
默认按 A100 实测采用每卡 2 条流、8 帧窗口、关闭重算：6,000 历史 token / 320 当前 token 的
长序列测试峰值约 63.2 GiB，约 10.3 帧/秒/卡。全层重算的每卡 4/8/12 流约 8.3–8.4 帧/秒/卡。
这是单卡合成历史压测，正式多卡吞吐以训练日志为准。显存不足时可开启全层重算。
`benchmark_vst.py` 的重复 KV 前缀仅用于压力测试，正式训练和验证均按真实顺序构建历史。

```bash
python -m mem_lite.train_vst --config /path/to/config.json --resume /path/to/last.pt
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m mem_lite.evaluate_vst --checkpoint /path/to/last.pt --output-dir /path/to/validation
```

检查点格式 `mem-lite-qwen35-vst-kvcache-v1` 保存可训练参数、optimizer、各 rank RNG、数据签名和组边界。
训练日志记录有效帧吞吐、数据等待、各卡峰值显存、历史 token 数、CE、PT MAE 和梯度范数。

## 精度验证

真实 4B 模型中，窗口和逐帧路径的 FP32 最大 PT 差异为约 `6.26e-6`（0.00063 个百分点）；
BF16 下对应差异约 0.59 个百分点。BF16 对批次形状有数值敏感性，独立重复同一帧也存在变化，
原生基线已有类似现象。因此不承诺 BF16 下改变 batch/window 后逐位一致；精确因果和梯度等价性
由多层微型 FP32 测试、真实 FP32 对照和固定形状未来帧隔离共同验证。
真实 GPU 测试同时确认历史 K/V 能收到下一帧损失的梯度，detach 后可继续独立 backward。
