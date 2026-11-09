"""当前三视角主序列与独立历史 KV 张量，训练和在线推理共用。"""

import torch
from torch.nn import functional as F

from mem_lite.datasets.views import fill_views


class M3Collator:
    """samples: task_prompt、同一时刻的3张images、可选[N, history_dim]历史。"""

    def __init__(self, processor, history_dim, max_sequence_length=8192, history_capacity=1008):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self.history_dim = history_dim
        self.max_sequence_length = max_sequence_length
        self.history_capacity = history_capacity

    def __call__(self, samples):
        prompts, images, histories = [], [], []
        for sample in samples:
            views = fill_views(sample["images"])
            history = sample.get("history_tokens")
            if history is not None and (history.ndim != 2 or history.shape[1] != self.history_dim
                                        or len(history) > self.history_capacity):
                raise ValueError(f"history_tokens 应为 [N, {self.history_dim}]，N 不超过 {self.history_capacity}")
            histories.append(history)
            # 三路图像保持调用方提供的固定视角顺序，不拼成一张图。
            content = [{"type": "text", "text": f"任务指令：{sample['task_prompt']}\n"}]
            for index, image in enumerate(views):
                content.extend([{"type": "image"}, {"type": "text", "text": f"（视角{index}）\n"}])
                images.append(image)
            content.append({"type": "text", "text": "根据当前观测与已有历史信息，判断任务完成进度。"})
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
        batch = dict(self.processor(text=prompts, images=images, return_tensors="pt", padding=True))
        tokenizer = self.processor.tokenizer
        sequences, types = [], []
        for row in range(len(samples)):
            keep = batch["attention_mask"][row].bool()
            ids, token_types = batch["input_ids"][row][keep], batch["mm_token_type_ids"][row][keep]
            sequences.append(ids)
            types.append(token_types)

        width = max(len(ids) for ids in sequences)
        if width > self.max_sequence_length:
            raise ValueError("M3 输入超过 max_sequence_length，不能静默截断任务或三视角观测")
        batch["input_ids"] = torch.stack([
            F.pad(ids, (width - len(ids), 0), value=tokenizer.pad_token_id) for ids in sequences
        ])
        batch["attention_mask"] = torch.stack([
            F.pad(torch.ones_like(ids), (width - len(ids), 0)) for ids in sequences
        ])
        batch["mm_token_type_ids"] = torch.stack([F.pad(ids, (width - len(ids), 0)) for ids in types])
        # 末尾补齐形状以减少重复编译，不移动有效token，避免改变线性注意力的分块位置。
        extra = min(self.max_sequence_length, (width + 63) // 64 * 64) - width
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            batch[key] = F.pad(batch[key], (0, extra), value=tokenizer.pad_token_id if key == "input_ids" else 0)
        # 在线历史留在原设备，避免 GPU→CPU→GPU 往返；入口统一搬运其余张量。
        reference = next((history for history in histories if history is not None), torch.empty(0))
        batch["history_tokens"] = reference.new_zeros(len(samples), self.history_capacity, self.history_dim)
        batch["history_valid_mask"] = torch.zeros(len(samples), self.history_capacity, dtype=torch.bool)
        for row, history in enumerate(histories):
            if history is not None:
                count = len(history)
                batch["history_tokens"][row, :count] = history.detach()
                batch["history_valid_mask"][row, :count] = True
        return batch
