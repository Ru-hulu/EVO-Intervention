# ruff: noqa: RUF001
"""任务 prompt × 长期记忆联合采样，以及逐样本的文本/进度监督。"""

import random

from qwen_training.progress_text import PROGRESS_INSTRUCTION, append_progress

TASK_INTENT = "将螺丝插入螺栓中"
TASK_PROMPT = "将螺钉插入套筒中"
# (prompt 模式, 是否保留正确长期记忆, 百分比)。每次取样只抽一次组合。
INPUT_COMBINATIONS = (
    ("matched", True, 49),
    ("matched", False, 14),
    ("mismatched", True, 14),
    ("mismatched", False, 3),
    ("missing", True, 14),
    ("missing", False, 6),
)
# 当前数据均为装配任务；扩展任务时可由 CLI 或每条索引提供相应的负任务池。
MISMATCHED_PROMPTS = ("打开抽屉", "将毛巾折叠好", "把水倒入杯子中")
STAGE_NAMES = {
    1: "拿起套筒",
    2: "拿起螺钉",
    3: "将螺钉和套筒对齐并插入",
    4: "将装配好的组件放入盒子",
}
TASK_DESCRIPTION = "任务一共有四个阶段：" + "、".join(STAGE_NAMES.values()) + "。\n"
OBSERVATION_INSTRUCTION = (
    "现在获得了三个视角的当前以及历史视觉信息，判断当前任务状态。\n"
    "请输出任务意图，以及当前视觉观测下正在执行的阶段、已完成阶段、未完成阶段。"
)
VISUAL_ONLY_INSTRUCTION = (
    "现在获得了三个视角的当前以及历史视觉信息，判断当前任务意图、任务阶段和任务进度。"
)
STAGE_PREREQUISITES = {1: set(), 2: set(), 3: {1, 2}, 4: {3}}


def describe_state(state):
    """只转换名称，不推算或交换 nowid / finishid / restid 的内容。"""
    current = STAGE_NAMES[state["nowid"]]
    finished = "、".join(STAGE_NAMES[i] for i in state["finishid"]) or "无"
    remaining = "、".join(STAGE_NAMES[i] for i in state["restid"]) or "无"
    return f"正在执行的阶段：{current}；已完成阶段：{finished}；未完成阶段：{remaining}。"


def describe_target(target, expect_set=None, include_progress=True):
    """输出意图、状态、可用的 expect-set 和末尾进度；记忆只使用 describe_state。"""
    response = f"任务意图为{target.get('intent', TASK_INTENT)}，当前任务状态{describe_state(target)}"
    if expect_set is not None:
        names = [STAGE_NAMES[int(stage)] if str(stage).isdigit() else str(stage) for stage in expect_set]
        response += "expect-set：" + ("、".join(names) or "无") + "。"
    return append_progress(response, target.get("progress") if include_progress else None)


def expected_stages(target):
    """优先使用显式标签；装配任务按前置条件求集合，不复制实际 nowid。"""
    if "expect_set" in target:
        return target["expect_set"]  # None 表示未知，[] 表示已知没有合法阶段。
    if target.get("intent", TASK_INTENT) not in (TASK_INTENT, TASK_PROMPT):
        return None
    finished = set(target["finishid"])
    return [stage for stage, prerequisites in STAGE_PREREQUISITES.items()
            if stage not in finished and prerequisites <= finished]


def make_message(task_prompt, memory):
    """不把组合类别、真实意图或当前标签泄漏到输入中。"""
    parts = ["根据三个视角的历史与当前图像，判断实际执行的整项任务意图，"
             "并输出正在执行的阶段、已完成阶段和未完成阶段。"]
    if task_prompt is not None:
        parts.append("目标任务：" + task_prompt)
    if memory is not None:
        parts.append("先前长期记忆：" + describe_state(memory))
    if task_prompt is not None:
        parts.append("同时输出目标任务在此刻允许执行的阶段集合（expect-set）。")
        parts.append(PROGRESS_INSTRUCTION)
    return [{"role": "user", "content": "\n".join(parts)}]


def build_training_examples(states, targets, seed=0, mismatched_prompts=MISMATCHED_PROMPTS):
    """49/14/14/3/14/6 联合采样；视觉内容和实际意图/记忆标签不变。"""
    rng = random.Random(seed)
    combinations = rng.choices(range(len(INPUT_COMBINATIONS)),
                               weights=[row[2] for row in INPUT_COMBINATIONS], k=len(states))
    result = {key: [] for key in ("messages", "responses", "combination_ids")}
    for state, target, combination in zip(states, targets, combinations, strict=True):
        mode, keep_memory, _ = INPUT_COMBINATIONS[combination]
        task_prompt = target.get("task_prompt", target.get("intent", TASK_PROMPT))
        expect_set = None
        if mode == "matched":
            expect_set = expected_stages(target)
        elif mode == "mismatched":
            negative = rng.choice(target.get("mismatched_prompts") or mismatched_prompts)
            if isinstance(negative, str):
                task_prompt = negative
            else:
                task_prompt = negative["prompt"]
                expect_set = negative.get("expect_set")  # 必须是这条观测下负任务的合法集合。
            if task_prompt in (target.get("task_prompt", TASK_PROMPT), target.get("intent", TASK_INTENT)):
                raise ValueError("错配 prompt 不能与实际任务相同，请修改负任务池")
        else:
            task_prompt = None
        result["messages"].append(make_message(task_prompt, state if keep_memory else None))
        output_target = {**target, "progress": 0.0 if mode == "mismatched" else target["progress"]}
        result["responses"].append(describe_target(output_target, expect_set, include_progress=mode != "missing"))
        result["combination_ids"].append(combination)
    return result


def build_messages(states, state_drop_probability=0.0, seed=0, *, legacy=False):
    """评估装配任务；legacy 仅用于复核已保存的旧版报告提示词。"""
    rng = random.Random(seed)
    messages = []
    for state in states:
        memory = state if rng.random() >= state_drop_probability else None
        if legacy:
            prompt = (TASK_DESCRIPTION + "在获取当前视觉观测前，" + describe_state(state) + "\n"
                      + OBSERVATION_INSTRUCTION) if memory is not None else VISUAL_ONLY_INSTRUCTION
            messages.append([{"role": "user", "content": prompt}])
        else:
            messages.append(make_message(TASK_PROMPT, memory))
    return messages
