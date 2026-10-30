"""Derive coarse action labels into NEW sidecar JSONs; never edit source annotations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


POLICIES = {
    '01': ('direct', '目标按键次数、分组确认要求和累计完成次数均明确。', []),
    '02': ('direct', '标注能还原具体搬运与确认顺序；首步存在多种合法路线，采用该示范已标注路线作为监督答案。', []),
    '03': ('missing_digit_sequence', '只有累计按键次数，没有目标数字序列，也没有每次实际按下的数字。',
           ['target_digit_sequence', 'per_press_digit_identity']),
    '04': ('needs_goal_count_confirmation', '示范记录按蓝键5次，但任务文本没有指定总次数或停止条件。', ['required_repeat_count_or_stop_condition']),
    '05': ('direct', '三层抽屉的开关顺序与各阶段完成状态明确。', []),
    '06': ('direct', '任务明确敲鼓5次，并标注了拿起、逐次敲击和放回。', []),
    '07': ('needs_goal_count_confirmation', '示范记录按压泵头5次，但任务只写重复按压，没有目标次数。', ['required_repeat_count_or_stop_condition']),
    '08': ('needs_goal_count_confirmation', '示范记录撒调料5次，但任务只写重复撒调料，没有目标次数。', ['required_repeat_count_or_stop_condition']),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match_states(states, expected):
    require(states == expected, '阶段状态与已识别任务规则不一致')


def count_in(pattern, text):
    match = re.search(pattern, text)
    return int(match[1]) if match else 0


def action_sequence(annotation, task_code):
    """One action per observed milestone; repetitions retain their own positions."""
    states = [s['state_description'] for s in annotation['stages']]
    prompt = annotation['task_prompt']
    require(states[0] == '无阶段性事件', '首阶段不是明确的初始状态')
    if task_code == '01':
        match = re.fullmatch(r'按数字牌点击红键：左键(\d+)次、中键(\d+)次，每组完成后按蓝键确认。', prompt)
        require(match is not None, '目标按键次数未明确')
        left, middle = map(int, match.groups())
        actions, expected = [], [(0, 0, 0)]
        for n in range(1, left + 1):
            actions.append('按左侧红键1次。')
            expected.append((n, 0, 0))
        actions.append('按蓝键1次，确认左键组完成。')
        expected.append((left, 0, 1))
        for n in range(1, middle + 1):
            actions.append('按中间红键1次。')
            expected.append((left, n, 1))
        actions.append('按蓝键1次，确认中键组完成。')
        expected.append((left, middle, 2))
        actual = [tuple(count_in(key + r'键已按(\d+)次', s) for key in ('左', '中', '蓝')) for s in states]
        require(actual == expected, '累计次数或确认顺序不符合任务要求')
        return actions, {'left_presses': left, 'middle_presses': middle, 'confirmations': 2, 'goal_source': 'task_prompt'}
    if task_code == '02':
        blank = next((p for word, p in [('中间', '中'), ('左侧', '左'), ('右侧', '右')] if f'利用{word}的空垫子' in prompt), None)
        require(blank is not None, '未找到空垫位置')
        objects = [p for p in '左中右' if p != blank]
        positions, confirmations, actions = {p: p for p in objects}, 0, []
        names = {'左': '左侧', '中': '中间', '右': '右侧'}
        for stage_index, state in enumerate(states[1:], 1):
            after = {p: p for p in objects}
            for origin, direct, via, destination in re.findall(r'([左中右])块(?:已移至([左中右])垫|经([左中右])垫移至([左中右])垫)', state):
                require(origin in objects, '标注出现原空垫上的积木')
                after[origin] = direct or destination
            pressed = count_in(r'已按键(\d+)次', state)
            moved = [p for p in objects if positions[p] != after[p]]
            require(len(set(after.values())) == 2, '两块积木被标为同一位置')
            if stage_index % 2:
                require(len(moved) == 1 and pressed == confirmations, '搬运阶段不明确')
                origin = moved[0]
                actions.append(f'将原本在{names[origin]}的积木从{names[positions[origin]]}垫子移到{names[after[origin]]}垫子。')
            else:
                require(not moved and pressed == confirmations + 1, '确认阶段不明确')
                actions.append('按前方确认按钮1次，确认本次搬运完成。')
            positions, confirmations = after, pressed
        require(len(actions) == 6 and confirmations == 3 and positions == dict(zip(objects, reversed(objects))), '最终交换结果不匹配')
        return actions, {'empty_pad': blank, 'goal_source': 'task_prompt', 'route_source': 'annotated_demonstration',
                         'route_is_one_valid_plan_not_unique': True}
    if task_code in ('03', '04'):
        pattern = r'已完成按键(\d+)次。' if task_code == '03' else r'已按蓝色按钮(\d+)次。'
        values = [count_in(pattern, s) for s in states]
        require(values == list(range(len(states))), '按键累计次数不连续')
        n = values[-1]
        text = '输入序列的下一位数字（具体数字未知）。' if task_code == '03' else '按蓝色按钮1次。'
        return [text] * n, {'observed_demonstration_presses': n, 'required_count_from_prompt': None,
                           'goal_source': 'not_explicit_in_task_prompt', 'target_digits': None} if task_code == '03' else (
                           {'observed_demonstration_repetitions': n, 'required_count_from_prompt': None,
                            'candidate_count_assumption': n, 'assumption_accepted': False})
    if task_code == '05':
        require(prompt == '从下到上依次打开三层抽屉，再从上到下依次关闭。', '抽屉目标顺序未识别')
        match_states(states, ['无阶段性事件', '已打开下层。', '已依次打开下、中层。', '已依次打开下、中、上层。',
            '已依次打开下、中、上层；随后已关闭上层。', '已依次打开下、中、上层；随后已依次关闭上、中层。',
            '已依次打开下、中、上层；随后已依次关闭上、中、下层。'])
        return [f'{verb}{layer}层抽屉。' for verb, layers in [('打开', '下中上'), ('关闭', '上中下')] for layer in layers], {'goal_source': 'task_prompt'}
    if task_code == '06':
        match = re.fullmatch(r'拿起鼓槌，连续敲鼓(\d+)次，再将鼓槌放回桌面。', prompt)
        require(match is not None, '敲鼓目标次数未识别')
        n = int(match[1])
        match_states(states, ['无阶段性事件', '已拿起鼓槌。'] + [f'已拿起鼓槌；已敲鼓{i}次。' for i in range(1, n + 1)] + [f'已拿起鼓槌、敲鼓{n}次并放回桌面。'])
        return ['拿起鼓槌。'] + ['敲鼓1次。'] * n + ['将鼓槌放回桌面。'], {'required_repetitions': n, 'goal_source': 'task_prompt'}
    if task_code == '07':
        require(prompt == '将清洁布移至洗洁精出液口下，重复按压泵头，最后将清洁布放回桌面。', '清洁布任务未识别')
        n = count_in(r'已按压泵头(\d+)次', states[-1])
        match_states(states, ['无阶段性事件', '已拿起清洁布。', '清洁布已移至出液口下。'] +
                     [f'清洁布已移至出液口下；已按压泵头{i}次。' for i in range(1, n + 1)] + [f'已按压泵头{n}次；清洁布已放回桌面。'])
        actions = ['拿起清洁布。', '将清洁布移至洗洁精出液口下。'] + ['按压泵头1次。'] * n + ['将清洁布放回桌面。']
    elif task_code == '08':
        require(prompt == '拿起调料瓶，向肉串重复撒调料，完成后放回桌面。', '撒调料任务未识别')
        n = count_in(r'已完成(\d+)次撒调料动作', states[-1])
        match_states(states, ['无阶段性事件', '已拿起调料瓶。'] + [f'已拿起调料瓶；已完成{i}次撒调料动作。' for i in range(1, n + 1)] + [f'已完成{n}次撒调料动作；调料瓶已放回桌面。'])
        actions = ['拿起调料瓶。'] + ['向肉串撒调料1次。'] * n + ['将调料瓶放回桌面。']
    else:
        raise ValueError('未知任务类型')
    return actions, {'observed_demonstration_repetitions': n, 'required_count_from_prompt': None,
                     'candidate_count_assumption': n, 'assumption_accepted': False}


def derive_episode(item):
    meta, annotation = item['metadata'], item['annotation']
    episode = meta['episode_id']
    code = episode[:2]
    category, reason, missing = POLICIES[code]
    stages = annotation['stages']
    require(annotation['num_stages'] == len(stages), '阶段数量不一致')
    require(annotation['frame_index_base'] == 0 and annotation['frame_range_convention'] == '[start_frame, end_frame)', '帧区间约定不符')
    cursor = 0
    for i, stage in enumerate(stages):
        require(stage['stage_id'] == i and stage['start_frame'] == cursor and stage['end_frame'] > cursor, '阶段区间有缺口或重叠')
        cursor = stage['end_frame']
    require(cursor == annotation['total_frames'], '阶段未覆盖完整视频')
    actions, goal = action_sequence(annotation, code)
    require(len(actions) == len(stages) - 1, '动作数与阶段转移数不一致')
    result = {'schema': 'procvlm-remaining-actions-audit-v1', 'episode_id': episode, 'split': meta['split'],
        'original_annotation': meta['annotation_source'], 'original_annotation_sha256': item['annotation_sha256'],
        'task_prompt': annotation['task_prompt'], 'classification': category, 'reason': reason, 'missing_information': missing,
        'goal_evidence': goal, 'granularity': 'completed task milestones; no inferred grasp/motion details',
        'full_demonstration_action_sequence': [{'action_index': i, 'action': text,
               'observed_completion_frame': stages[i + 1]['start_frame'], 'evidence_stage_id': i + 1} for i, text in enumerate(actions)],
        'annotation_usage': 'Full demonstration annotations may define supervised targets only; never include them in model inputs.',
        'scope_note': 'JSON semantic audit only; source video frames have not been visually re-annotated.', 'stages': [], 'samples_5fps': []}
    for i, stage in enumerate(stages):
        suffix = actions[i:]
        result['stages'].append({**stage, 'remaining_actions': suffix if category == 'direct' else None,
            'candidate_remaining_actions': suffix if category == 'needs_goal_count_confirmation' else None,
            'partial_remaining_actions_with_unknown_digits': suffix if category == 'missing_digit_sequence' else None,
            'remaining_demonstrated_event_count': len(suffix), 'label_is_directly_derivable': category == 'direct'})
    terminal = []
    stage_index = 0
    for sample in item['samples']:
        while sample['frame_index'] >= stages[stage_index]['end_frame']:
            stage_index += 1
        require(stages[stage_index]['start_frame'] <= sample['frame_index'] < stages[stage_index]['end_frame'], '采样帧未对应唯一阶段')
        suffix = actions[stage_index:]
        issues = list(missing)
        if not suffix and sample['pt'] < 1 - 1e-8:
            issues.append('annotated_events_finished_but_pt_below_one')
            terminal.append({'frame_index': sample['frame_index'], 'pt': sample['pt']})
        result['samples_5fps'].append({**sample, 'stage_id': stage_index,
            'remaining_actions': suffix if category == 'direct' else None,
            'candidate_remaining_actions': suffix if category == 'needs_goal_count_confirmation' else None,
            'partial_remaining_actions_with_unknown_digits': suffix if category == 'missing_digit_sequence' else None,
            'review_reasons': issues, 'eligible_without_open_semantic_issues': category == 'direct' and not issues})
    result['pt_terminal_review'] = {'count': len(terminal), 'frames': terminal,
        'interpretation': 'All annotated demonstration events ended while PT still increases. For underspecified goals this does not prove task success. Do not invent extra actions or rewrite PT.'}
    return result


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.snapshot.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks, episodes, failures, terminal_review = defaultdict(list), [], [], []
    for item in sorted(bundle['episodes'], key=lambda x: x['metadata']['episode_id']):
        try:
            result = derive_episode(item)
        except ValueError as error:
            failures.append({'episode_id': item['metadata']['episode_id'], 'error': str(error)})
            continue
        episode = result['episode_id']
        relative = Path('labels') / Path(episode).parent / (Path(episode).name + '.remaining_actions.json')
        write_new(args.output_dir / relative, result)
        entry = {'episode_id': episode, 'split': result['split'], 'classification': result['classification'],
            'samples_5fps': len(result['samples_5fps']),
            'eligible_without_open_semantic_issues': sum(s['eligible_without_open_semantic_issues'] for s in result['samples_5fps']),
            'pt_terminal_review_count': result['pt_terminal_review']['count'], 'sidecar': str(relative)}
        episodes.append(entry)
        tasks[episode.split('/')[0]].append(entry)
        if result['pt_terminal_review']['count']:
            terminal_review.append({'episode_id': episode, 'classification': result['classification'],
                                    **result['pt_terminal_review']})
    active = [e for e in episodes if e['split'] in ('train', 'validation')]
    by_task = []
    for task, entries in sorted(tasks.items()):
        selected = [e for e in entries if e['split'] != 'expert']
        category, reason, missing = POLICIES[task[:2]]
        by_task.append({'task': task, 'classification': category, 'reason': reason, 'missing_information': missing,
            'all_videos': len(entries), 'active_videos': len(selected), 'splits': dict(Counter(e['split'] for e in entries)),
            'active_samples': sum(e['samples_5fps'] for e in selected),
            'pt_terminal_review_samples': sum(e['pt_terminal_review_count'] for e in selected),
            'eligible_without_open_semantic_issues': sum(e['eligible_without_open_semantic_issues'] for e in selected)})
    summary = {'schema': 'procvlm-remaining-actions-audit-summary-v1',
        'annotation_files_audited': len(bundle['episodes']), 'successfully_parsed': len(episodes),
        'active_videos': len(active), 'expert_videos_held_out': sum(e['split'] == 'expert' for e in episodes),
        'active_samples_5fps': sum(e['samples_5fps'] for e in active),
        'classifications_all_videos': dict(Counter(e['classification'] for e in episodes)),
        'classifications_active_videos': dict(Counter(e['classification'] for e in active)),
        'classifications_active_samples': dict(sum((Counter({e['classification']: e['samples_5fps']}) for e in active), Counter())),
        'direct_action_labels_and_pt_without_open_issues': sum(e['eligible_without_open_semantic_issues'] for e in active),
        'pt_terminal_review_samples': sum(e['pt_terminal_review_count'] for e in active),
        'pt_terminal_review_videos': sum(e['pt_terminal_review_count'] > 0 for e in active),
        'source_files_unchanged': bundle['source_files_unchanged'], 'training_started': False,
        'limitations': ['基于JSON中的任务级动作里程碑，不等于逐帧视频复核。',
            '未来阶段标注只用于构造监督答案，不能进入模型输入；输入仍是当前及过去8帧与原任务文本。',
            '示范中的5次重复不自动视为任务规定的5次；候选标签未批准用于训练。',
            '保留原始PT。动作结束与PT未到1的帧单独复核，未生成正式SFT问答。'],
        'by_task': by_task, 'annotation_errors': failures, 'episodes': episodes}
    write_new(args.output_dir / 'audit-summary.json', summary)
    write_new(args.output_dir / 'review-required.json', {
        'goal_or_digit_review': [x for x in by_task if x['missing_information']],
        'pt_terminal_review': terminal_review, 'annotation_errors': failures})
    write_new(args.output_dir / 'source-integrity.json', {
        'source_snapshot': str(args.snapshot.resolve()),
        'source_snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        'original_files_unchanged_during_collection': bundle['source_files_unchanged'],
        'source_hashes_before': bundle['source_hashes_before'], 'source_hashes_after': bundle['source_hashes_after']})
    print(json.dumps({key: value for key, value in summary.items() if key not in ('episodes',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
