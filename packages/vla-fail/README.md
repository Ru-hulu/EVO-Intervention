# VLA-FAIL：LLMD / ACC 介入判定 API

这是从本地 `VLA-FAIL/Code` 整合的独立 Python 实现，方法依据 VLA-FAIL 论文第 3 节（arXiv:2606.21386v1），不是论文作者发布的官方代码。接收模型特征和动作位置，输出分数、是否超阈值、介入建议及解释信息。EVO-Intervention 的 π0.5 接线、数据格式和测试流程见 [集成说明](../../docs/vla_fail.md)。

- **LLMD（统计级）**：逐 token 的完整协方差、平方马氏距离、token 最大值聚合，不做时间平滑。
- **ACC（动作级）**：旧预测未执行部分与新预测对应部分的绝对末端位置误差，按新预测的位置极差归一化，再做 EMA。
- **联合判定**：沿用论文逻辑 OR。可以只启用 LLMD。
- 支持 NumPy / Python 数组、可选 PyTorch 张量输入、分批特征统计、NPZ 保存/加载、成功轨迹阈值校准、Python API 和可选 HTTP API。

这份代码实现的是**介入信号**。本仓库已提供 PyTorch π0.5 动作专家特征提取和 Policy 接线。MuJoCo 运行、语义异常识别、大模型接管并生成动作、介入标签蒸馏和 JEV 融合仍属于上层系统。没有用真实仿真轨迹验证检测准确率。两个分数都是异常指标，不是失败概率；不使用世界模型。

## 1. 安装和运行

在本目录执行，建议使用项目自己的 Python 环境：

```powershell
python -m pip install -e ".[http,test]"
python -m examples.quickstart
python -m pytest -q
```

只用 Python API：

```powershell
python -m pip install -e .
```

基础依赖为 NumPy、SciPy；无需安装 MuJoCo、模型框架或大模型 SDK。`quickstart` 会生成 `artifacts/toy_llmd.npz`，其中的特征和阈值仅用于演示，不能当作实际模型的统计量部署。

## 2. 先确认 token 的含义

论文 LLMD 使用的是**动作专家最终线性输出层的输入特征**，不是图像 patch token 或语言 token。形状为 `[H,F]`：

| 输入/参数 | 含义 |
|---|---|
| `H` | 动作块长度，也是这里的动作特征 token 数量 |
| `F` | 每个 token 的特征维度，论文实验为 1024 |
| `[N,H,F]` | `N` 个微调观测提取出的特征，不是 `N` 条完整 episode |
| `[B,H,F]` | 分批统计时的一个 batch |
| `[H,3]` | ACC 所需的绝对末端位置 XYZ，三个轴必须使用同一单位 |
| `executed_steps=R` | 前后两次规划之间实际执行的旧动作步数，满足 `1 <= R < H` |

Flow matching 模型的特征还依赖生成时间和带噪动作。请在**纯噪声起点**使用同一个固定先验噪声提取特征：

```python
from vla_fail import make_fixed_noise

fixed_noise = make_fixed_noise((20, 7), seed=42)  # 完整模型动作维度，此处举例为 7
```

以下是需要主项目提供的适配逻辑，`extract_tokens` 不是某个现成 VLA 的标准 API：

```python
# 伪代码：你们实现该适配层，定位动作输出层之前的特征。
tokens = extract_tokens(
    observation=observation,
    noise=fixed_noise,
    generation_endpoint="pure_noise",
)  # [H,F]，去除 batch=1 的维度，但保留所有动作 token
```

微调集统计、成功轨迹校准、部署这三个阶段的 checkpoint、预处理、层位置、token 顺序、固定噪声必须一致。模型应使用推理模式，关闭 dropout。某些模型源码的时间方向与论文相反，不要仅凭数值 `t=0` 判断是否为纯噪声起点。

固定噪声可以单独用于一次检测前向，保持实际动作采样方式不变；也可以让实际动作生成复用该噪声，但会改变策略的随机行为。

如果你们提供的是 **VLM 层的 token**，同一数学实现也可以计算分数，但这是对论文的扩展：需要固定 token 数量、位置和含义，重新统计和校准，不能把语言长度变化或 padding 混入固定位置统计，也不能称为已复现论文原版 LLMD。当前 API 不接受可变 token 数或 mask，不自动选择、池化或补齐 token。

## 3. LLMD：统计、校准、保存

```python
import numpy as np
from vla_fail import LLMDDetector, calibrate_threshold, make_fixed_noise

train_tokens = np.load("train_tokens.npy")  # [N,H,F]，微调训练观测的特征

llmd = LLMDDetector(
    regularization=1e-6,
    fixed_noise=make_fixed_noise((20, 7), seed=42),
    metadata={
        "checkpoint": "your-checkpoint-id",
        "task": "pick-cup",
        "feature_source": "action_head_input",
        "layer": "your-output-head-input",
        "preprocessing": "your-preprocessing-version",
    },
)
llmd.fit(train_tokens)

# 单独留出的成功 rollout 特征；每条可以有不同的规划次数 K。
# successful_rollout_tokens 是 [K,H,F] 数组的列表。
successful_rollout_tokens = [
    np.load("success_001.npy"),
    # ...继续加入足够数量的独立成功轨迹
]
episode_scores = [
    [result.score for result in llmd.score_batch(episode)]
    for episode in successful_rollout_tokens
]
report = calibrate_threshold(episode_scores, alpha=0.05)
llmd.threshold = report.threshold
llmd.save("llmd.npz")
```

上例的校准文件是需要采集的数据，只有一条轨迹时会明确报样本不足。`fit` / `partial_fit` 会清除旧阈值，防止换统计量后继续使用不匹配的阈值；请在完成统计后校准或赋值。

大数据集不要把全部特征放进内存，用：

```python
llmd = LLMDDetector(regularization=1e-6, fixed_noise=fixed_noise)
for batch in feature_batches:  # 每个 batch [B,H,F]
    llmd.partial_fit(batch)
llmd.finalize()
```

这里使用合并中心矩的方法，最终协方差为 `M2 / N`，与论文分母 `N` 一致，不使用无偏估计 `N-1`。`finalize()` 后释放累计协方差缓存，只保存均值和 Cholesky 因子。已完成统计的对象不允许继续 `partial_fit`，请新建对象重新统计。

保存文件包含均值、Cholesky 因子、样本数、正则项、阈值、可选固定噪声和 JSON 元数据，不包含原始训练特征、不使用 pickle。

## 4. 只接统计级介入：最短在线 API

```python
from vla_fail import LLMDDetector, VLAFAILDetector

llmd = LLMDDetector.load(
    "llmd.npz",
    expected_metadata={"checkpoint": "your-checkpoint-id", "task": "pick-cup"},
)
detector = VLAFAILDetector(llmd=llmd)  # 只启用统计层

# tokens：由主项目用 llmd.fixed_noise 在同一条件下提取的 [H,F]
result = detector.step(tokens=tokens)
print(result.to_dict())

if result.need_intervention is True:
    # 通知你们的控制器切换到大模型；本库不执行切换。
    pass
elif result.need_intervention is None:
    # 当前缺少可用阈值等条件，由主项目明确处理。
    pass
```

也可以只调用分数 API：

```python
signal = llmd.score(tokens)
print(signal.score, signal.triggered, signal.worst_token)
print(signal.per_token_scores)
```

`worst_token` 使用 Python **从 0 开始**的索引。`per_token_scores` 是逐 token 的平方马氏距离；`score` 是它们的最大值。没有时间平滑。

可以直接传 PyTorch Tensor，库会执行 detach 并复制到 CPU float64；CUDA 数据传输有额外开销。当前不是 Torch/GPU 后端，不能保持 autograd。

## 5. ACC：动作输入和时序状态

```python
from vla_fail import ACCDetector

acc = ACCDetector(alpha=0.9, v_min=1e-3, threshold=None)
first = acc.update(first_positions)  # [H,3]；首次 ready=False、triggered=None
second = acc.update(next_positions, executed_steps=5)

print(second.raw_score)                    # 未平滑分数
print(second.score)                        # EMA 后，用于判断
print(second.per_axis_mae)                 # XYZ 平均绝对误差
print(second.per_axis_scale)               # XYZ 的 max(新重叠位置极差, v_min)
print(second.per_axis_normalized_error)    # 各轴归一化误差
print(second.triggered)                    # 没有阈值时为 None
```

公式（`L=H-R`）：

```text
old = previous_positions[R:H]
new = current_positions[0:L]
scale[d] = max(max(new[:,d]) - min(new[:,d]), v_min)
mae[d] = mean(abs(old[:,d] - new[:,d]))
raw_score = mean(mae / scale)
score = alpha * previous_score + (1-alpha) * raw_score
```

论文称 `scale` 为速度，但主文公式计算的是**新重叠位置序列的极差**，没有除以时间。本实现忠实采用该公式；三个轴分别归一化。默认使用全部 `H-R` 重叠，旧新 chunk 长度必须相同。附录个别 X-VLA 配置表报告的重叠长度小于 `H-R`，本实现不猜测其截取策略。

论文未明确 EMA 初值，本实现选择 **0 初始化**。首个 chunk 只缓存，不产生 0 分伪观测；第一次有重叠时计算 `(1-alpha)*raw_score`。校准必须采用同样规则。

无状态 API 可用于服务、离线 replay 或自己管理状态：

```python
signal = acc.score_pair(
    previous_positions,
    current_positions,
    executed_steps=5,
    previous_smoothed_score=last_score,
)
last_score = signal.score
```

`score_pair` 不修改对象历史。`update` 会复制并保存当前 chunk，避免上层修改输入数组破坏历史。

### 动作表示转换

ACC 只收绝对末端 XYZ，不能直接传全部 7/8 维 action，也不能直接传归一化网络输出、旋转或夹爪开合。

如果是**同一个固定坐标系下、相邻时刻之间的笛卡尔位移增量**，可以使用：

```python
from vla_fail import delta_xyz_to_absolute
positions = delta_xyz_to_absolute(delta_xyz, start_xyz=current_actual_eef_xyz)
```

如果每个动作是相对 chunk 起点的独立偏移，应使用 `start_xyz + offsets`，不能 `cumsum`。若是关节目标/增量、工具局部坐标、速度控制，需按控制器定义做反归一化、坐标变换或前向运动学。旧/新预测必须对应同一世界坐标系和同一控制时间轴，各自使用其生成时的起始状态；不要把旧预测重新锚定到新位置。

实际执行步数变化可以每次传入不同 `executed_steps`。如果发生动作替换、队列跳步、仿真重置或大模型接管，旧预测已不再代表相同的控制历史，应调用 `reset()`，等小策略重新产生连续可比较的 chunk。

## 6. 校准 ACC 和联合 OR

```python
from vla_fail import ACCDetector, calibrate_threshold

calibration_acc = ACCDetector(alpha=.9, v_min=1e-3)
episode_scores = []
for episode in successful_action_rollouts:
    calibration_acc.reset()
    scores = []
    # episode 中每项为 (absolute_positions, executed_steps_since_previous)
    # 第一项的 executed_steps_since_previous 用 None。
    for positions, steps in episode:
        result = calibration_acc.update(positions, executed_steps=steps)
        if result.ready:
            scores.append(result.score)
    episode_scores.append(scores)

acc_threshold = calibrate_threshold(episode_scores, alpha=.05).threshold
acc = ACCDetector(alpha=.9, v_min=1e-3, threshold=acc_threshold)
```

每条用于 ACC 校准的轨迹必须包含至少两个 chunk，才能有有效分数。

本地论文指定“时间恒定 conformal prediction band”，详细算法引用 FIPER。这里提供一个明确的轨迹级 split-conformal 实现：

1. 每条**成功校准轨迹**取分数最大值 `m_e=max_t score[e,t]`。
2. `n` 条轨迹取 `k=ceil((n+1)*(1-alpha))`，阈值基于第 `k` 小的 `m_e`。
3. 论文报警条件是 `>=`，为处理分数并列，存储阈值为该分位数的 `nextafter(q,+inf)`，等价于分数严格大于原分位数。
4. `k>n` 时拒绝给出没有所需覆盖率依据的有限阈值：需要更多成功轨迹或更大 `alpha`。

例如单检测器 `alpha=.05` 至少需要 19 条成功轨迹才能产生有限阈值；更丰富的独立轨迹通常更有意义。这个实现不声称与作者未提供的校准代码逐项一致。

校准轨迹应独立于特征统计数据，并且仅包含成功执行；不应把失败测试集或所有时间步混在一起直接取 95% 分位数。轨迹长度可以不同。覆盖率解释依赖校准/部署成功轨迹的交换性，不保证 OOD 失败召回率。

联合 API：

```python
from vla_fail import VLAFAILDetector

detector = VLAFAILDetector(llmd=llmd, acc=acc)
detector.reset()  # 每次 episode 开始，以及控制器接管/切回时调用

result = detector.step(
    tokens=tokens,                  # [H,F]
    positions=absolute_positions,   # [H,3]
    executed_steps=steps,           # 首次可以为 None，后续必须给实际 R
)
```

返回结构：

```json
{
  "need_intervention": true,
  "decision_complete": false,
  "reasons": ["llmd_feature_ood"],
  "unavailable": ["acc_history_missing"],
  "llmd": {
    "score": 12.99999,
    "per_token_scores": [1.99999875, 12.99999],
    "worst_token": 1,
    "threshold": 10.0,
    "triggered": true
  },
  "acc": {
    "ready": false,
    "score": null,
    "raw_score": null,
    "per_axis_mae": null,
    "per_axis_scale": null,
    "per_axis_normalized_error": null,
    "overlap": 0,
    "threshold": 0.5,
    "triggered": null
  }
}
```

上面的数字只是手算演示，不能用于实际部署。`decision_complete` 表示所有启用信号是否都已具备判定条件，不是动作是否执行完毕：

| 启用的信号 | `need_intervention` |
|---|---|
| 任意一个触发，包括另一项未知 | `True` |
| 全部可判断且都未触发 | `False` |
| 无触发，但有缺少阈值/历史的信号 | `None` |

未启用的检测器不参与联合判定。启用了检测器却漏传其输入会报错，不静默跳过。不要使用 `if not result.need_intervention` 混淆 `False` 和 `None`。

分别使用 5% 的阈值不意味着 OR 后仍是 5% 的误报控制。若希望对联合误报做保守的 5% 上界校准，可给每项分配 `.025`，此时各自至少 39 条成功轨迹才能得到有限阈值；这是 union bound 的分配方式，不是论文报告的默认超参数。联合误报仍需在你们的 rollout 上验证。

多个并行 MuJoCo 环境可以共享只读 LLMD，但必须每个环境单独创建 ACC/VLAFAIL 状态对象。切换任务或 checkpoint 时请使用对应的统计量和阈值，不要直接复用其它任务的特征分布。

## 7. HTTP API（可选）

运行过 quickstart 后，可以启动本地演示服务：

```powershell
python -m vla_fail.server --llmd-model artifacts/toy_llmd.npz --acc --v-min 0.01 --acc-threshold 0.5
```

只接统计层：

```powershell
python -m vla_fail.server --llmd-model your_llmd.npz
```

默认地址 `http://127.0.0.1:8000`，OpenAPI 页面在 `/docs`。没有阈值的检测器可以提供分数，但触发判断为 `null`。

| 方法/路径 | 功能 |
|---|---|
| `GET /health` | 启用状态、LLMD 特征形状、阈值是否配置 |
| `POST /v1/llmd/score` | 请求 `{"tokens": [[...], ...]}`，返回 LLMDResult |
| `POST /v1/acc/score` | 传前后位置、实际执行步数、上次 EMA，返回 ACCResult |
| `POST /v1/intervention/evaluate` | 对所有已启用信号进行联合 OR 判断 |

LLMD 请求：

```python
import httpx
response = httpx.post("http://127.0.0.1:8000/v1/llmd/score",
                      json={"tokens": [[3, 12], [8, 9]]})
response.raise_for_status()
print(response.json())
```

ACC 请求：

```json
{
  "previous_positions": [[0.1,0.2,0.1],[0.2,0.2,0.1],[0.3,0.2,0.1]],
  "current_positions": [[0.1,0.24,0.1],[0.2,0.24,0.1],[0.3,0.24,0.1]],
  "executed_steps": 1,
  "previous_smoothed_score": 0.02
}
```

联合请求字段是 `tokens`、`positions`、`previous_positions`、`executed_steps`、`previous_smoothed_score`。首次没有历史时，只传 `tokens` 和 `positions`；只启用 LLMD 时仅传 `tokens`。后续需要原样传回上次响应中的 `acc.score`。

HTTP 服务的 ACC **不保留全局历史**，历史由客户端显式传入，避免并行环境串状态。请求维度/步数错误返回 422，未完成统计返回 409，单项检测器未配置返回 503。

实时控制优先使用 Python API。HTTP JSON 编解码、大 token 数组传输、GPU 到 CPU 的复制都增加开销；此服务主要方便跨进程接入与调试。

## 8. 开销与验证边界

- 完整 LLMD 统计量占用 `O(H*F^2)`，单次分数约 `O(H*F^2)`。FP64 Cholesky 因子在 `H=20,F=1024` 时约 160 MiB，`H=30` 时约 240 MiB；离线累计、分解时峰值更高。
- 分批统计避免保存所有训练特征，但不会消除完整协方差自身的开销。
- ACC 仅使用已有位置预测，额外复杂度 `O((H-R)*3)`。
- 固定噪声提取、VLA 推理、动作坐标转换和 MuJoCo 控制开销不包含在本库分数计算中。
- 不会自动降低特征维数、池化 token 或换成对角协方差，这些改变需要重新验证。

```powershell
python -m examples.benchmark --horizon 20 --features 1024 --samples 32 --iterations 20
```

该程序只测合成特征下的本地 CPU 数值计算，不代表端到端实时性或检测准确率。

自动化测试覆盖手算 LLMD/ACC、完整相关协方差、分批统计、低秩正则、保存/加载、错误输入、时间对齐、静止/单步重叠、平滑延迟、控制器状态重置、校准并列分数与样本不足、OR 的未知状态，以及 HTTP 请求隔离。真实 MuJoCo/VLA 集成和检测质量仍需由项目数据验证。

## 9. 源码入口与论文对应

| 文件 | 功能 |
|---|---|
| `vla_fail/llmd.py` | 固定噪声工具、逐 token 特征统计、LLMD、统计量持久化 |
| `vla_fail/acc.py` | XYZ 转换工具、无状态/有状态 ACC |
| `vla_fail/calibration.py` | 成功轨迹最大分数的恒定 conformal 阈值 |
| `vla_fail/detector.py` | 介入判定、OR 组合、episode reset |
| `vla_fail/results.py` | 可解释、可转 JSON 的返回结构 |
| `vla_fail/server.py` | 可选 FastAPI 服务 |
| `examples/quickstart.py` | 可运行的数值演示与保存/加载示例 |
| `examples/benchmark.py` | 本地 CPU 分数耗时测试 |
| `tests/` | 数值、状态、校准与 HTTP 回归测试 |

本地来源：`../main.tex:274` 固定噪声；`:290` token 统计与平方马氏距离；`:326` ACC；`:353` OR；`../appendix.tex:235` 校准设定、`:646` 参数。与论文对应的关键点是 `N` 分母、完整逐 token 协方差、最大平方距离、绝对 XYZ、位置极差归一化和 EMA。校准细化、零 EMA 初值、状态/API/持久化是本实现明确记录的工程选择。
