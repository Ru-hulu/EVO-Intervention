"""Causal observation windows and shared training/inference preprocessing."""
from collections import deque
from functools import lru_cache
import json
from pathlib import Path

from PIL import Image
import torch
from safetensors.torch import load_file

from .model import progress_batch


class ObservationWindow:
    def __init__(self, size=8):
        self.frames = deque(maxlen=size)
        self.episode_id = None
        self.size = size

    def push(self, episode_id, frame):
        if episode_id != self.episode_id:
            self.frames.clear()
            self.episode_id = episode_id
        self.frames.append(frame)
        return [self.frames[0]] * (self.size - len(self.frames)) + list(self.frames)


class ProgressDataset:
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ProgressCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def encode(self, rows, supervised=False):
        if all('pixel_cache' in row for row in rows):
            return self._cached_encode(rows, supervised)
        opened, examples = {}, []
        for row in rows:
            for path in row['image']:
                if path not in opened:
                    with Image.open(path) as image:
                        opened[path] = image.convert('RGB')
            examples.append((row['task'], [opened[path] for path in row['image']]))
        batch = dict(progress_batch(self.processor, examples))
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError('Window exceeds max_length; lower image resolution rather than truncate frames')
        if supervised:
            batch['labels'] = torch.tensor([row['pt'] for row in rows], dtype=torch.float32)
        return batch

    def _cached_encode(self, rows, supervised):
        features, pixels, grids = [], [], []
        expected_views = ['cam_high'] * 8 + ['cam_left_wrist', 'cam_right_wrist']
        for row in rows:
            if row.get('input_views') != expected_views or len(row['window_indices']) != 10:
                raise ValueError('Cache does not match history7-main-current3-v1')
            cached = _episode_cache(row['pixel_cache'])
            for view, index in zip(row['input_views'], row['window_indices'], strict=True):
                pixels.append(cached['pixels_' + view][index])
                grids.append(cached['grids_' + view][index].unsqueeze(0))
            features.append({key[6:]: value.tolist() for key, value in cached.items()
                             if key.startswith('input_')})
        batch = dict(self.processor.tokenizer.pad(features, padding=True, return_tensors='pt'))
        batch.update(pixel_values=torch.cat(pixels), image_grid_thw=torch.cat(grids))
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError('Cached window exceeds max_length')
        if supervised:
            batch['labels'] = torch.tensor([row['pt'] for row in rows], dtype=torch.float32)
        return batch

    def __call__(self, rows):
        return self.encode(rows, supervised=True)


@lru_cache(maxsize=8)
def _episode_cache(path):
    # Safetensors maps all views; only seven past main images and three current images are copied.
    return load_file(path)
