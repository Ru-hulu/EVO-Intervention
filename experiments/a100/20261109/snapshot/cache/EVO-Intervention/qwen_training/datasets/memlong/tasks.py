# ruff: noqa: RUF001, RUF002
"""MemLong 的通用指令、步骤及累计状态解析；模型输入不含本条轨迹的答案。"""
import re

TASKS = {
    '01-按数字按键': ('scene_count', '按数字牌点击对应红键，每组完成后按蓝键确认。',
                    ('按左侧红键一次。', '按中间红键一次。', '按蓝键确认一次。')),
    '02-交换积木': ('fixed', '利用空垫交换两块积木，每次搬运后按确认键。',
                   ('将第一块积木移至空垫。', '按确认键。', '将第二块积木移至腾出的垫子。',
                    '按确认键。', '将第一块积木移至目标垫子。', '按确认键。')),
    '03-输入数字序列': ('fixed', '按顺序在键盘上输入数字。', ('按下序列中的下一个数字。',)),
    '04-连续按按钮': ('fixed', '连续按蓝色按钮。', ('按蓝色按钮一次。',)),
    '05-依次开关抽屉': ('fixed', '从下到上依次打开三层抽屉，再从上到下依次关闭。',
                      ('打开下层抽屉。', '打开中层抽屉。', '打开上层抽屉。',
                       '关闭上层抽屉。', '关闭中层抽屉。', '关闭下层抽屉。')),
    '06-连续敲鼓': ('fixed', '拿起鼓槌，连续敲鼓，再将鼓槌放回桌面。',
                   ('拿起鼓槌。', '敲鼓一次。', '放回鼓槌。')),
    '07-重复挤洗洁精': ('fixed', '将清洁布移至出液口下，重复按压泵头，最后放回清洁布。',
                       ('拿起清洁布。', '移至出液口下。', '按压泵头一次。', '放回清洁布。')),
    '08-重复撒调料': ('fixed', '拿起调料瓶，向肉串重复撒调料，完成后放回桌面。',
                     ('拿起调料瓶。', '撒调料一次。', '放回调料瓶。')),
}


def number(text, pattern):
    match = re.search(pattern, text)
    return int(match[1]) if match else 0


def state_counts(task, text, order):
    """输入是完成事件之后的累计状态，不能把该状态段当作待执行动作。"""
    if text == '无阶段性事件':
        return [0] * len(TASKS[task][2])
    if task.startswith('01-'):
        return [number(text, rf'{key}已按(\d+)次') for key in ('左键', '中键', '蓝键')]
    if task.startswith(('02-', '05-')):
        return [int(i < order) for i in range(6)]
    if task.startswith('03-'):
        return [number(text, r'已完成按键(\d+)次')]
    if task.startswith('04-'):
        return [number(text, r'已按蓝色按钮(\d+)次')]
    if task.startswith('06-'):
        return [1, number(text, r'(?:已)?敲鼓(\d+)次'), int('放回' in text)]
    if task.startswith('07-'):
        returned = '放回' in text
        moved = '移至出液口下' in text or returned
        return [1, int(moved), number(text, r'已按压泵头(\d+)次'), int(returned)]
    return [1, number(text, r'已完成(\d+)次撒调料动作'), int('放回' in text)]
