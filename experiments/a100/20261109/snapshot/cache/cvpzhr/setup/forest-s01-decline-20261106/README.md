# S01 下降标签 · FOREST 任务微调

华为云工作目录：`/cache/cvpzhr/setup/forest-s01-decline-20261106`。
输出：`/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106`。
缓存：`/cache/cvpzhr/datasets/forest-s01-decline-5fps-20261106`。

从 `forest-original-cap50-static-jitter-e3-20261104/last.pt` 的 step 19934 加载 LoRA、历史投影、progress 头。优化器、步数、epoch 全部重新开始，不续跑原训练。不修改原权重、原代码或 v2/v3 数据集。

- GPU：物理 2、3，双卡 DDP；其他卡不操作。
- 3 epochs，LR 1e-5，batch 4/GPU，全局 batch 8，无梯度累积，BF16，gradient_checkpointing=false。
- 每半 epoch 验证；保存 best.pt / last.pt，每 200 步及每 epoch 保存检查点。
- 110 条 train、28 条 val。同 actual scene seed 的重试或跨模型轨迹在同一侧；视频哈希及动作指纹无跨集重复。颜色排列与任务本身可重合，不能称为未见任务/布局泛化测试。
- Train：36 条 IMPRINT 成功、42 条 VLA 失败、24 条脚本纠正、8 条脚本失败。
- Val：8 条 IMPRINT 成功、12 条 VLA 失败、6 条脚本纠正、2 条脚本失败。
- Train 额外采样 1 次成功轨迹，使每 epoch 轨迹访问量为 37:74。FOREST 以帧训练，因此这不是逐帧样本数 1:2。
- 训练缓存以 20% 概率插入 1–3 段、各 5–20 个 5Hz 观测的暂停，实际 25 条轨迹被选中。所有视角同步重复、p 复制对应原帧。历史状态在暂停时仍因果更新，虚拟时间严格递增。Val 不增强。
- 全部采用 v3 dense_progress，按源帧取标签，禁止用视频位置比例替代。

`prepare.py` 构建拆分、逐帧因果历史缓存和独立核验。`run.py` 仅在 2/3 卡无冲突时启动缓存构建，核验后运行两步双卡短测，再从原检查点重新初始化并开始正式微调；短测权重不会用于正式微调。

云端 `status.json`、`worker-*.log`、`cache-validation.json`、`smoke.log` 和 `train.log` 可用于检查状态。模型、记忆、训练代码为当时训练代码的独立快照，唯一训练入口改动是新增 weights-only 初始化。
