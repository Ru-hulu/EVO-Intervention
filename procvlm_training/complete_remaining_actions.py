"""Complete seven-task action labels in a new directory; preserve original annotations/PT."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .audit_remaining_actions import derive_episode, write_new


REPEAT_PROMPTS = {
    '04': '连续按蓝色按钮{count}次。',
    '07': '将清洁布移至洗洁精出液口下，按压泵头{count}次，最后将清洁布放回桌面。',
    '08': '拿起调料瓶，向肉串撒调料{count}次，完成后放回桌面。',
}


def complete_episode(item):
    audit = derive_episode(item)
    code = audit['episode_id'][:2]
    if code == '03':
        raise ValueError('输入数字序列已从本版本排除')
    prompt = audit['task_prompt']
    goal = dict(audit['goal_evidence'])
    if code in REPEAT_PROMPTS:
        count = goal['observed_demonstration_repetitions']
        prompt = REPEAT_PROMPTS[code].format(count=count)
        goal = {'required_repetitions': count, 'count_source': 'observed_demonstration_annotations',
                'policy': 'Use the annotated demonstration total as the target for this dataset version.',
                'count_was_explicit_in_original_prompt': False, 'count_is_explicit_in_new_prompt': True}
    actions = [row['action'] for row in audit['full_demonstration_action_sequence']]
    result = {'schema': 'procvlm-remaining-actions-v1', 'label_status': 'complete',
        'episode_id': audit['episode_id'], 'split': audit['split'], 'sample_fps': 5,
        'original_annotation': audit['original_annotation'],
        'original_annotation_sha256': audit['original_annotation_sha256'],
        'original_task_prompt': audit['task_prompt'], 'task_prompt': prompt,
        'task_prompt_adjusted': prompt != audit['task_prompt'], 'goal_evidence': goal,
        'full_demonstration_action_sequence': audit['full_demonstration_action_sequence'],
        'label_granularity': 'task milestones; repeated actions are preserved individually',
        'usage': 'Use task_prompt consistently for training and inference. Remaining actions are supervision only, never model inputs.',
        'annotation_scope': 'Derived from existing JSON annotations; no new video re-annotation.',
        'stages': [], 'samples_5fps': []}
    for stage in audit['stages']:
        remaining = actions[stage['stage_id']:]
        result['stages'].append({key: stage[key] for key in
            ('stage_id', 'start_frame', 'end_frame', 'state_description')} | {
            'remaining_actions': remaining, 'remaining_action_count': len(remaining)})
    terminal = []
    for sample in audit['samples_5fps']:
        remaining = actions[sample['stage_id']:]
        conflict = not remaining and sample['pt'] < 1 - 1e-8
        result['samples_5fps'].append({key: sample[key] for key in ('frame_index', 'pt', 'stage_id')} | {
            'remaining_actions': remaining, 'remaining_action_count': len(remaining),
            'needs_pt_terminal_review': conflict,
            'eligible_for_joint_training_without_pt_resolution': not conflict})
        if conflict:
            terminal.append({'frame_index': sample['frame_index'], 'pt': sample['pt']})
    result['pt_terminal_review'] = {'count': len(terminal), 'frames': terminal,
        'policy': 'Keep [] after all annotated goal actions are completed; preserve original PT and flag the discrepancy. Do not invent extra actions.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    bundle = json.loads(raw)
    if bundle['sample_fps'] != 5:
        raise ValueError('Expected the existing 5 FPS dataset')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    episodes, excluded, review = [], [], []
    by_task = defaultdict(list)
    for item in sorted(bundle['episodes'], key=lambda row: row['metadata']['episode_id']):
        meta = item['metadata']
        if meta['episode_id'].startswith('03-输入数字序列/'):
            excluded.append({'episode_id': meta['episode_id'], 'split': meta['split'],
                'samples_5fps': len(item['samples']), 'reason': 'user_excluded_task', 'original_data_deleted': False})
            continue
        result = complete_episode(item)
        episode = result['episode_id']
        path = Path('labels') / Path(episode).parent / (Path(episode).name + '.remaining_actions.json')
        write_new(args.output_dir / path, result)
        entry = {'episode_id': episode, 'split': result['split'], 'file': str(path),
            'samples_5fps': len(result['samples_5fps']), 'pt_terminal_review_samples': result['pt_terminal_review']['count'],
            'task_prompt_adjusted': result['task_prompt_adjusted']}
        episodes.append(entry)
        by_task[episode.split('/')[0]].append(entry)
        if result['pt_terminal_review']['count']:
            review.append({'episode_id': episode, 'split': result['split'], **result['pt_terminal_review']})
    tasks = []
    for task, rows in sorted(by_task.items()):
        tasks.append({'task': task, 'videos': len(rows), 'splits': dict(Counter(row['split'] for row in rows)),
            'samples_5fps': sum(row['samples_5fps'] for row in rows),
            'goal_count_policy': 'annotated demonstration total, made explicit in new task_prompt' if task[:2] in REPEAT_PROMPTS else 'original task requirements'})
    splits = {split: {'videos': sum(row['split'] == split for row in episodes),
                     'samples_5fps': sum(row['samples_5fps'] for row in episodes if row['split'] == split),
                     'pt_terminal_review_samples': sum(row['pt_terminal_review_samples'] for row in episodes if row['split'] == split)}
              for split in ('train', 'validation', 'expert')}
    manifest = {'schema': 'procvlm-remaining-actions-manifest-v1', 'status': 'labels_completed',
        'tasks': len(tasks), 'videos': len(episodes), 'splits': splits,
        'remaining_action_labels_missing': 0, 'sample_fps': 5,
        'task_scope': 'ProcVLM seven-task label dataset; original datasets and prior audit are unchanged.',
        'user_request': '放弃输入数字序列这个任务。将其他任务进行补齐。',
        'repeat_count_resolution': '三类未明示次数的重复任务按现有示范的总次数构建目标，并仅在新JSON的task_prompt明确该次数；original_task_prompt保留原文。',
        'source_snapshot': str(args.snapshot.resolve()), 'source_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
        'expert_policy': 'Retain expert split separately; do not include it in training or validation.',
        'pt_policy': 'Original PT and sampling are unchanged. Empty action lists mean annotated goal events ended; PT conflicts remain flagged.',
        'training_started': False, 'source_files_modified': False, 'previous_audit_modified': False,
        'by_task': tasks, 'excluded_episodes': excluded, 'episodes': episodes}
    write_new(args.output_dir / 'manifest.json', manifest)
    write_new(args.output_dir / 'pt-terminal-review.json', {
        'status': 'action_labels_complete_pt_semantics_unresolved',
        'samples': sum(row['count'] for row in review), 'episodes': review,
        'note': '这些帧已补齐remaining_actions=[]，不是缺标；原PT仍小于1，未修改也未伪造额外动作。'})
    print(json.dumps({key: value for key, value in manifest.items() if key not in ('episodes', 'excluded_episodes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
