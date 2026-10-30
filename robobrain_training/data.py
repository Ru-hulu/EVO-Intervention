"""训练和推理共用的官方八图布局、提示词及答案编码。"""

import json
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def score_text(pt):
    return f"<score>{100 * pt:+.1f}%</score>"


def parse_score(text):
    match = re.fullmatch(r"\s*<score>\s*([+-]?\d+(?:\.\d+)?)%\s*</score>\s*", text)
    if not match:
        return None
    value = float(match[1]) / 100
    return value if 0 <= value <= 1 else None


def user_prompt(task):
    return f"""You are a vision evaluator for robot task progress.
Task: {task}
REFERENCE START — front view, task beginning: <image>
REFERENCE END — front view, task completed: <image>
BEFORE — front view: <image>
BEFORE — left wrist view: <image>
BEFORE — right wrist view: <image>
AFTER — front view: <image>
AFTER — left wrist view: <image>
AFTER — right wrist view: <image>
BEFORE is the beginning of this video. AFTER is the current observation.
Using the reference images as the 0% and 100% anchors, estimate the progress
from BEFORE to AFTER. Consider all three views. Unchanged task state means
unchanged progress. Return only <score>+X.X%</score>, with one decimal place,
between +0.0% and +100.0%."""


def make_sample(episode_id, frame_index, task, pt, start, current, end):
    return {
        "id": f"{episode_id}/{frame_index:08d}", "episode_id": episode_id,
        "frame_index": frame_index, "pt": pt,
        "image": [start[0], end, *start, *current],
        "conversations": [{"from": "human", "value": user_prompt(task)},
                          {"from": "gpt", "value": score_text(pt)}],
    }


class RewardDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class RewardCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def encode(self, rows, *, supervised):
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "right" if supervised else "left"
        texts, images, answers = [], [], []
        for row in rows:
            parts = row["conversations"][0]["value"].split("<image>")
            if len(row["image"]) != 8 or len(parts) != 9:
                raise ValueError("RoboBrain reward samples require eight ordered images")
            content = []
            for index, part in enumerate(parts):
                content.append({"type": "text", "text": part})
                if index < 8:
                    content.append({"type": "image"})
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            answer = row["conversations"][1]["value"] + tokenizer.eos_token
            texts.append(prompt + answer if supervised else prompt)
            answers.append(tokenizer.encode(answer, add_special_tokens=False))
            opened = {}
            for path in row["image"]:
                if path not in opened:
                    with Image.open(path) as image:
                        opened[path] = image.convert("RGB")
                images.append(opened[path])
        batch = dict(self.processor(text=texts, images=images, padding=True, return_tensors="pt"))
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError("Sample exceeds max_length; do not truncate images or answers")
        if supervised:
            labels = torch.full_like(batch["input_ids"], -100)
            for index, answer in enumerate(answers):
                end = int(batch["attention_mask"][index].sum())
                start = end - len(answer)
                if batch["input_ids"][index, start:end].tolist() != answer:
                    raise ValueError("Assistant answer token boundary mismatch")
                labels[index, start:end] = batch["input_ids"][index, start:end]
            batch["labels"] = labels
        return batch

    def __call__(self, rows):
        return self.encode(rows, supervised=True)
