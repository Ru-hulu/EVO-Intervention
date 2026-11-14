PRM 五组实验：已确认方案（2026-11-14）

状态：29 项配置全部确认，用户已授权自动训练和五组评测。第 1 组正在执行。
第 2–5 组执行、断点恢复、调度、指标与曲线导出已实现；实现完成不等于模型 GPU 验证或结果完成。
2026-11-14 用户要求删除 Codex 定时监控。远程控制器独立运行，不依赖本聊天心跳。

一、七种方法
Robometer-4B、RoboBrain2.5-4B、Robo-Dopamine-GRM-3B、ProcVLM-2B、
R²VLM-Alfred-RL、VLAC-2B、FoREST。
前六种使用官方原始权重；FoREST 在对应 suite 训练集微调，应标为 fine-tuned 对照。
ProcVLM 沿用最近 8 帧接口，不把上一轮 CoT 作为下一轮输入。
RoboBrain2.5 和 Robo-Dopamine 保留当前轨迹的真实末帧参考，结果必须注明使用未来终点。

二、FoREST 训练
State、Order、Count、Recover 分别独立从 Qwen3.5-4B 原始基座开始，各训练 1 epoch。
每套 2 张 GPU、5 FPS、effective batch 18、lr 1e-4、warmup 3% + cosine、seed 42，
无人工增强，保留原生结构与损失。统一使用最终 checkpoint，不根据验证结果选择权重。
CLIP + max_depth：相邻帧 CLIP 相似度，两侧全局最大值计算深度，候选内取最大深度；无深度阈值。
训练和带记忆评测使用同一配置。四卡可并行运行两个双卡训练任务，启动前检查现有占用。

三、五组实验
1. 基线：七种方法，State 89 / Order 119 / Count 177，共 385 条验证轨迹，5 FPS。
   计算 Pearson、Spearman、官方 VS，并保存 VS 的 ER、C 两个分项。
2. 记忆：只对 ProcVLM、R²VLM、FoREST、Robometer 做消融和扰动。
   去记忆：ProcVLM 仅当前主视角；R²VLM 不输入上一轮文字记忆；FoREST 不读写记忆；
   Robometer 仅当前三视角共 3 图。同一模型消融前后使用相同权重。计算 VOC 和 VS。
   扰动：全部 385 条轨迹，每条分别制作静止版和回环版。每版插入 2 次，
   静止 1–3 秒；回环为倒放 1–3 秒再等长正放。三视角同步，真值随源帧映射变化，
   静止时保持不变，不重算扰动视频的累计视觉差标签。
   使用种子 42、43、44，四个模型共享每组扰动清单；报告 VOC 均值和波动，保存逐轨迹结果。
3. Recover：只用原有翻杯、笔袋、喷瓶、抽屉、重复倒水五任务；当前清单 130 训练 / 34 验证。
   FoREST-Recover 使用独立微调权重，其他六种方法 zero-shot，均为 5 FPS。
   指标：VOC；成功轨迹末尾 2% 的平均预测（称“终点得分”，不称 SA）；
   官方 FPL（最大完整下降段的起点，反转系数 0.2，无下降按官方规则计罚）；
   重复／错误动作导致的真值平台区间同时计算绝对 MAE 和相对该段起点的漂移 MAE。
4. 顺序约束：O12 和 S13 各 7 条验证轨迹，5 FPS；各自指令和交换指令分别评测。
   真值必须匹配当前指令和当前视频；同一视频只切换指令，不切换权重。
   保存独立的指令对应真值，保留原标签；计算 VOC、VS，暂不做 CSVC。
5. 帧率：只用原生 30 FPS 的 235 条（State 59 / Order 109 / Count 67）。
   输入 30 / 20 / 10 / 5 / 2 FPS，固定原生记忆策略、容量和模型权重。
   ProcVLM 最近 8 帧、Robometer 前缀均匀 8 时刻、FoREST 窗口和 token 上限、
   R²VLM 文字记忆规则均不随 FPS 调整，FoREST 不按帧率重新训练。
   保存全输入帧率曲线和指标；主跨帧率比较使用共同整秒 1 FPS 预测点计算 VOC、VS，
   不插值、不额外调用模型。不同评分采样密度分开报告，1 FPS 结果不掩盖完整曲线抖动。

四、汇总和留档
Pearson/Spearman 对每条轨迹的预测进度与真值计算，再按轨迹等权平均。
VS 也逐轨迹保存并等权汇总；保留 ER 和非平坦时间占比 C。
保存所有轨迹、种子、区间的结果，预测、真值、时间戳、源帧映射、原始回答及配置/权重哈希。
常量相关性与无效预测分别记录，不补零、不平滑、不将有效子集冒充全覆盖。
不对每条预测曲线单独做 min-max 拉伸。MAE 使用有依据的输出尺度映射并记录单位。
异常平台、失败起点、成功结局等标注缺失时明确报告，不能从任意重复动作臆造失败标签。

五、数据与当前依赖
State/Order/Count 的 1904 条轨迹、1651148 个原始帧视觉进度标签已重算并通过核验。
523 条时间比例标签已删除，原有划分与阶段标注不变；Recover 原有文件未被修改。
Recover 另有外部新增三个任务，本轮不纳入。
SOC 基线和第 2/5 组清单已经冻结并独立核验。官方权重审计和 FoREST 最终检查点门禁已接入。
Recover 倒水尚缺 35 条 PT（28 训练、7 验证），既有补标控制器按授权规则等待空闲缓存阶段处理。
Recover 终点/FPL/异常平台指标另需显式事件标注；不能从 PT=1 猜测成功或从任意回落猜测失败起点。
O12/S13 交换指令条件尚需 14 条与新指令一致的 PT；各自指令的 14 条输入可独立执行。
缺失标注只阻塞对应条件/指标；所有原始数据和标签保持不变。
不复用旧环境 417 条验证轨迹的指标或预测；不把本方案直接当作旧九模型评测器的启动配置。
代码与配置同步 Git，数据、标签、预测不上传 Git。

六、远程执行入口与队列
设置根目录 S=/cache/cvpzhr/setup/prm-five-groups-20261114，E=S/evaluation，工作目录 S/code。
所有命令从 S/code 运行，CPU准备/报告显式设置 CUDA_VISIBLE_DEVICES=。
CPU 环境为 /cache/cvpzhr/envs/forest-labels/bin/python。

experiments.prm_five_groups_20261114.eval_dispatch run
  单实例 GPU 调度器：原 32 个第一组分片，加 440 个后续分片（第2组168、第3组28、第4组64、第5组180）。
  先完成六官方模型第一组的可执行部分并审计全部缺口；随后解除 Count 预留，按原两卡/配置从 step1200 恢复。
  第一组 FoREST 仅匹配 suite 的最终权重可运行；后续任务依赖输入与对应最终权重，首分片真实轨迹通过后才复制。
  不直接手工启动 experiment_worker：它要求当前调度器的 GPU 准入记录、同名作业锁和活跃槽位。
  每卡最多一份调度器评测进程；保留 60 秒资源稳定期、每分钟最多启动一份和原资源保护。

experiments.prm_five_groups_20261114.experiment_inputs
  准备/核验冻结输入，写 E/conditions/readiness.json。缺失 PT/事件标注模板在 E/annotation-templates。
  审核后的标注放入同结构 E/annotations；模板自身永远不能被当作有效标注。
  group4 sidecar schema=prm-instruction-pt-v1：确切指令、源视频 SHA、全部原帧索引/PT、方法和来源文件 SHA。
  group3 sidecar schema=prm-recover-events-v1：结局、失败原帧起点、官方角色、重复/错误平台区间和来源 SHA。
  平台只计算落在已标注区间内的观测；漂移以该段首个实际观测为基准，记录锚点，不插值。
  调度器每 5 分钟刷新依赖；事件标注更新触发 CPU 重汇总，无需重跑模型。

experiments.prm_five_groups_20261114.experiment_report --group G --condition C --model MODEL
  逐条核验配置、帧映射、PT、源视频、初始化和完整预测，再重新计算指标。
  输出 evaluations/prm-five-groups-20261114/groupG/C/MODEL/metrics.json、metrics.md、episode-metrics.csv、curves/*.svg。
  缺失/无效/常量与成功/FPL/平台的适用分母分别保存；第5组全流与共同整秒网格分开。
  总表 followup-comparison.json/.md，三种子均值和样本标准差仅在三组覆盖齐备时生成。

experiments.prm_five_groups_20261114.upgrade_controllers
  一次性 CPU 控制器升级：核对代码 SHA、进程身份和活跃收据，仅向旧 CPU 控制器发信号；
  新控制器采用已记录的训练/评测进程及其子进程，不向 GPU 作业发信号，不重复启动 worker。
  常规空闲升级仍可使用 E/dispatcher-reload-when-idle.json；不能并起第二份竞争 GPU 的调度器。

完成证明为输出目录 completion-audit.json：四训练、所有条件/逐条覆盖/初始化/权重/事件依赖全部审计。
Order/O03 既知左腕少一帧保留 expected；涉及三视角的条件不能宣称全覆盖。未决输入永远保留显式阻塞。
