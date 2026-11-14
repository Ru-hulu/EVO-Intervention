"""Progress-only ProcVLM: identical cached/online inputs, assistant-only loss.

This is a separate contract from the historical remaining-actions SFT dataset.
Only past/current main-view images and the original task prompt enter inference.
"""
from functools import lru_cache
import json
from pathlib import Path

from PIL import Image
import torch
from safetensors.torch import load_file

POLICY = 'procvlm-progress-only-recent8-leftpad-v1'


def user_prompt(task):
    return (f'Given the recent observation and the task "{task}", estimate the current completion '
            'percentage. Output only a float percentage wrapped by <progress> tags, '
            'for example <progress>25.00%</progress>.')


def answer_text(pt):
    return f'<progress>{100 * pt:.2f}%</progress>'


def chat_prompt(processor, task):
    content = [{'type': 'image'} for _ in range(8)] + [{'type': 'text', 'text': user_prompt(task)}]
    return processor.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)


def online_prompt(processor, task, paths):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return dict(processor(text=[chat_prompt(processor, task)], images=images,
                          return_tensors='pt', min_pixels=4096, max_pixels=65536))


@lru_cache(maxsize=4)
def episode_cache(path):
    return load_file(path, device='cpu')


class ProgressDataset:
    def __init__(self, path, *, supervised=False):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError('Empty progress-only cache')
        for row in self.rows:
            i = row['sample_index']
            assert row['supervision_policy'] == POLICY
            assert row['window_indices'] == [max(0, j) for j in range(i - 7, i + 1)]
            assert row['assistant_target'] == answer_text(row['pt'])
            assert row['supervision_eligible'] is True

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


class ProgressCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def encode(self, rows, *, supervised=False):
        tok = self.processor.tokenizer
        tok.padding_side = 'right' if supervised else 'left'
        features, all_pixels, all_grids, label_rows = [], [], [], []
        for row in rows:
            if 'pixel_cache' in row:
                stored = episode_cache(row['pixel_cache'])
                feature = {key[6:]: value.tolist() for key, value in stored.items() if key.startswith('input_')}
                pixels = torch.cat([stored['pixels_cam_high'][i] for i in row['window_indices']])
                grids = torch.stack([stored['grids_cam_high'][i] for i in row['window_indices']])
            else:
                batch = online_prompt(self.processor, row['task'], row['image'])
                feature = {key: val[0].tolist() for key, val in batch.items()
                           if key not in ('pixel_values', 'image_grid_thw')}
                pixels, grids = batch['pixel_values'], batch['image_grid_thw']
            assert set(feature) == {'input_ids', 'attention_mask'}, set(feature)
            prompt_len = len(feature['input_ids'])
            if supervised:
                assert row['supervision_eligible']
                answer = answer_text(row['pt']) + tok.eos_token
                ids = tok.encode(answer, add_special_tokens=False)
                feature['input_ids'] += ids
                feature['attention_mask'] += [1] * len(ids)
                label_rows.append([-100] * prompt_len + ids)
            features.append(feature); all_pixels.append(pixels); all_grids.append(grids)
        result = dict(tok.pad(features, padding=True, return_tensors='pt'))
        if result['input_ids'].shape[1] > self.max_length:
            raise ValueError('Refuse truncation of image windows or targets')
        result.update(pixel_values=torch.cat(all_pixels), image_grid_thw=torch.cat(all_grids))
        if supervised:
            labels = torch.full_like(result['input_ids'], -100)
            for i, values in enumerate(label_rows):
                labels[i, :len(values)] = torch.tensor(values, dtype=labels.dtype)
            result['labels'] = labels
            result['progress_gt'] = torch.tensor([100 * r['pt'] for r in rows], dtype=torch.float32)
        return result

    def __call__(self, rows): return self.encode(rows, supervised=True)
