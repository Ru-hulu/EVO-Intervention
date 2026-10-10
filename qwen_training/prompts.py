# ruff: noqa: RUF001
"""构造两种输入提示，并把任务意图和阶段标签转成监督文本。"""

import random


TASK_INTENT = "将螺丝插入螺栓中"
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


def describe_state(state):
    """只转换名称，不推算或交换 nowid / finishid / restid 的内容。"""
    current = STAGE_NAMES[state["nowid"]]
    finished = "、".join(STAGE_NAMES[i] for i in state["finishid"]) or "无"
    remaining = "、".join(STAGE_NAMES[i] for i in state["restid"]) or "无"
    return f"正在执行的阶段：{current}；已完成阶段：{finished}；未完成阶段：{remaining}。"


def describe_target(target):
    """输出意图和当前状态；上一状态提示仍只使用 describe_state。"""
    return f"任务意图为{target['intent']}，当前任务状态{describe_state(target)}"


def build_messages(states, state_drop_probability=0.0, seed=0):
    """训练时以 0.3 概率仅给简短指令，同时省略四阶段说明与上一状态。"""
    rng = random.Random(seed)
    messages = []
    for state in states:
        if rng.random() >= state_drop_probability:
            prompt = TASK_DESCRIPTION
            prompt += "在获取当前视觉观测前，" + describe_state(state) + "\n"
            prompt += OBSERVATION_INSTRUCTION
        else:
            prompt = VISUAL_ONLY_INSTRUCTION
        messages.append([{"role": "user", "content": prompt}])
    return messages
