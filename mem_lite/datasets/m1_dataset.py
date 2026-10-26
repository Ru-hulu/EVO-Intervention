"""读取已构造的 M1 样本，组织原生多图输入与仅答案部分的 CE 标签。"""

import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from mem_lite.datasets.m1_samples import read_training_sample


class M1Dataset(Dataset):
    def __init__(self, sample_files):
        self.samples = []
        for source in sample_files:
            path = Path(source)
            if path.is_dir():
                path = path / "samples.jsonl"
            with path.open(encoding="utf-8") as stream:
                self.samples.extend((path.parent, json.loads(line)) for line in stream if line.strip())
        if not self.samples:
            raise ValueError("M1 样本集为空，请先构造 samples.jsonl")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        directory, sample = self.samples[index]
        return read_training_sample(sample, directory)


class M1Collator:
    def __init__(self, processor, max_sequence_length=8192):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self.max_sequence_length = max_sequence_length

    def build_prompt(self, samples):
        """只构造观测与判断指令，供自回归评测使用；不读取response。"""
        prompts, images = [], []
        for sample in samples:
            content = [{"type": "text", "text": sample["prompt"]}]
            for index, image in enumerate(sample["images"]):
                content.extend([
                    {"type": "text", "text": f"\n观测 {index}："},
                    {"type": "image"},
                ])
                images.append(image)
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
        batch = dict(self.processor(text=prompts, images=images, return_tensors="pt", padding=True))
        if batch["input_ids"].shape[1] > self.max_sequence_length:
            raise ValueError("观测队列超过max_sequence_length，不能截断队列后沿用边界编号")
        return batch

    def __call__(self, samples):
        batch = self.build_prompt(samples)
        tokenizer = self.processor.tokenizer
        sequences, labels, token_types = [], [], []
        for row, sample in enumerate(samples):
            keep = batch["attention_mask"][row].bool()
            prefix = batch["input_ids"][row][keep]
            answer = prefix.new_tensor(
                [*tokenizer.encode(sample["response"], add_special_tokens=False), tokenizer.eos_token_id]
            )
            sequences.append(torch.cat((prefix, answer)))
            labels.append(torch.cat((torch.full_like(prefix, -100), answer)))
            if "mm_token_type_ids" in batch:
                token_types.append(torch.cat((batch["mm_token_type_ids"][row][keep], torch.zeros_like(answer))))

        width = max(len(sequence) for sequence in sequences)
        if width > self.max_sequence_length:
            raise ValueError(
                f"输入及答案共 {width} tokens，超过 max_sequence_length={self.max_sequence_length}；"
                "请降低 max_pixels 或调整长度配置，不能截断队列后继续使用原边界标签"
            )
        batch["input_ids"] = torch.stack([
            F.pad(ids, (width - len(ids), 0), value=tokenizer.pad_token_id) for ids in sequences
        ])
        batch["attention_mask"] = torch.stack([
            F.pad(torch.ones_like(ids), (width - len(ids), 0)) for ids in sequences
        ])
        batch["labels"] = torch.stack([F.pad(ids, (width - len(ids), 0), value=-100) for ids in labels])
        if token_types:
            batch["mm_token_type_ids"] = torch.stack([F.pad(ids, (width - len(ids), 0)) for ids in token_types])
        # expert_tokens_path 只保留在 Dataset 返回值中；本阶段不读取，也不加入提示或模型输入。
        return batch
