"""M3 三视角与历史输入组织；不读取 PT 标签文件，不构造第三阶段数据集。"""

import torch
from torch.nn import functional as F


class M3Collator:
    """samples: task_prompt、同一时刻的3张images、可选[128, history_dim]历史。"""

    def __init__(self, processor, history_dim=512, max_sequence_length=8192):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self.history_dim = history_dim
        self.max_sequence_length = max_sequence_length

    def __call__(self, samples):
        prompts, images, histories = [], [], []
        for sample in samples:
            if len(sample["images"]) != 3:
                raise ValueError("M3 每条样本必须包含同一时刻的3个视角，不能用补图代替缺失视角")
            history = sample.get("history_tokens")
            if history is not None and history.shape != (128, self.history_dim):
                raise ValueError(f"history_tokens 应为 [128, {self.history_dim}]；尚无事件时传 None")
            histories.append(history)
            # 三路图像保持调用方提供的固定视角顺序，不拼成一张图。
            content = [{"type": "text", "text": f"任务指令：{sample['task_prompt']}\n"}]
            for index, image in enumerate(sample["images"]):
                content.extend([{"type": "image"}, {"type": "text", "text": f"（视角{index}）\n"}])
                images.append(image)
            content.append({"type": "text", "text": "根据当前观测与已有历史信息，判断任务完成进度。"})
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
        batch = dict(self.processor(text=prompts, images=images, return_tensors="pt", padding=True))
        tokenizer = self.processor.tokenizer
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        sequences, types, history_masks = [], [], []
        for row, history in enumerate(histories):
            keep = batch["attention_mask"][row].bool()
            ids, token_types = batch["input_ids"][row][keep], batch["mm_token_type_ids"][row][keep]
            history_mask = torch.zeros_like(ids, dtype=torch.bool)
            if history is not None:
                # 历史插在任务文本之后、第一个原生图像块之前。占位ID不扩充词表，
                # 在模型入口会全部替换为连续历史embedding，attention_mask保持为1。
                start = (ids == vision_start_id).nonzero()[0, 0].item()
                ids = torch.cat((ids[:start], ids.new_full((128,), tokenizer.pad_token_id), ids[start:]))
                token_types = torch.cat((token_types[:start], token_types.new_zeros(128), token_types[start:]))
                history_mask = torch.zeros_like(ids, dtype=torch.bool)
                history_mask[start:start + 128] = True
            sequences.append(ids)
            types.append(token_types)
            history_masks.append(history_mask)

        width = max(len(ids) for ids in sequences)
        if width > self.max_sequence_length:
            raise ValueError("M3 输入超过 max_sequence_length，不能静默截断历史或三视角观测")
        batch["input_ids"] = torch.stack([
            F.pad(ids, (width - len(ids), 0), value=tokenizer.pad_token_id) for ids in sequences
        ])
        batch["attention_mask"] = torch.stack([
            F.pad(torch.ones_like(ids), (width - len(ids), 0)) for ids in sequences
        ])
        batch["mm_token_type_ids"] = torch.stack([F.pad(ids, (width - len(ids), 0)) for ids in types])
        batch["history_mask"] = torch.stack([F.pad(mask, (width - len(mask), 0)) for mask in history_masks])
        reference = next((history for history in histories if history is not None), None)
        batch["history_tokens"] = None if reference is None else torch.stack([
            history if history is not None else torch.zeros_like(reference) for history in histories
        ])
        return batch
