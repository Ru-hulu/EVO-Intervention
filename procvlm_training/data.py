"""Shared causal prompt, visual preprocessing and assistant-only supervision."""
import json
from pathlib import Path
import re

import torch
from qwen_vl_utils import process_vision_info


def user_prompt(task):
    return (f'Given the recent observation and the task "{task}", first infer the remaining atomic '
            'actions required to complete the task. Then estimate the current completion percentage '
            'and output it as a float wrapped by <progress> tags.')


def answer_text(actions, pt):
    reasoning = ('The following actions are required:\n' +
                 '\n'.join(f'{i + 1}. {action}' for i, action in enumerate(actions))) if actions else 'No remaining task actions.'
    return f'{reasoning}\nTherefore, the estimated progress is <progress>{100 * pt:.2f}%</progress>.'


def parse_progress(text):
    match = re.search(r'<progress>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</progress>', text, re.I)
    value = float(match[1]) / 100 if match else None
    return value if value is not None and 0 <= value <= 1 else None


def messages(row):
    if len(row['image']) != 8:
        raise ValueError('ProcVLM requires the current sampled frame and seven preceding frames')
    return [{'role': 'user', 'content': [{'type': 'image', 'image': image} for image in row['image']] +
             [{'type': 'text', 'text': user_prompt(row['task'])}]}]


class ProgressDataset:
    def __init__(self, path, *, supervised=False):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if supervised:
            self.rows = [row for row in self.rows if row['supervision_eligible']]
        if not self.rows:
            raise ValueError(f'Empty dataset: {path}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ProgressCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def encode(self, rows, *, supervised=False):
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = 'right' if supervised else 'left'
        conversations = [messages(row) for row in rows]
        prompts = [self.processor.apply_chat_template(value, tokenize=False, add_generation_prompt=True)
                   for value in conversations]
        texts, answer_ids = prompts, []
        if supervised:
            if any(not row['supervision_eligible'] for row in rows):
                raise ValueError('Terminal PT conflicts must be filtered before loss supervision')
            answers = [answer_text(row['remaining_actions'], row['pt']) + tokenizer.eos_token for row in rows]
            texts = [prompt + answer for prompt, answer in zip(prompts, answers)]
            answer_ids = [tokenizer.encode(answer, add_special_tokens=False) for answer in answers]
        images, videos = process_vision_info(conversations)
        batch = dict(self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt'))
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError('Sample exceeds max_length; do not truncate history frames or supervision')
        if supervised:
            labels = torch.full_like(batch['input_ids'], -100)
            for i, answer in enumerate(answer_ids):
                end = int(batch['attention_mask'][i].sum())
                start = end - len(answer)
                if batch['input_ids'][i, start:end].tolist() != answer:
                    raise ValueError('Assistant token boundary mismatch')
                labels[i, start:end] = batch['input_ids'][i, start:end]
            batch['labels'] = labels
            # Upstream expects percentages. Keep full PT precision independently of rounded answer text.
            batch['progress_gt'] = torch.tensor([100 * row['pt'] for row in rows], dtype=torch.float32)
        return batch

    def __call__(self, rows):
        return self.encode(rows, supervised=True)
