"""Causal observation windows and shared training/inference preprocessing."""
from collections import deque
import json
from pathlib import Path

from PIL import Image
import torch

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

    def __call__(self, rows):
        return self.encode(rows, supervised=True)
