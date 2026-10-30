"""Qwen 原生多图输入，六帧状态全部通过语言头做统一 token CE。"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from qwen_training.reference_io import FRAME_COUNT, IMAGE_COUNT, parse_frame_predictions


class ReferenceProgressModel(nn.Module):
    def __init__(self, model_path, dtype=torch.bfloat16, freeze_backbone=True,
                 max_pixels=262144, lora_rank=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa",
        )
        if freeze_backbone:
            self.qwen.requires_grad_(False)
        self.qwen.model.visual.requires_grad_(False)
        if lora_rank:
            from peft import LoraConfig, inject_adapter_in_model

            targets = [name for name, module in self.qwen.named_modules()
                       if name.startswith("model.language_model.layers.") and isinstance(module, nn.Linear)]
            inject_adapter_in_model(LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=targets, bias="none",
            ), self.qwen)
        self.qwen.model.visual.eval()

    def train(self, mode=True):
        super().train(mode)
        self.qwen.model.visual.eval()
        return self

    def build_inputs(self, messages, images, responses=None):
        """处理13张独立图片，保留原生图像网格和多模态位置编码。

        只预处理图片一次；在已展开的图文 Prompt 后追加答案 token 和 EOS。
        不截断图片、历史或答案；padding 和整个输入均不计算损失。
        """
        if len(messages) != len(images) or any(len(row) != IMAGE_COUNT for row in images):
            raise ValueError("每条消息需要13张图片：5参考、六个时刻的8张观测图")
        if responses is not None and len(responses) != len(messages):
            raise ValueError("responses 与 messages 的样本数必须一致")
        prompts = [
            self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for chat in messages
        ]
        batch = dict(self.processor(
            text=prompts, images=[image for row in images for image in row],
            return_tensors="pt", padding=True,
        ))
        if responses is not None:
            sequences, labels, types = [], [], []
            for i, response in enumerate(responses):
                valid = batch["attention_mask"][i].bool()
                prefix = batch["input_ids"][i][valid]
                answer = prefix.new_tensor(
                    self.tokenizer.encode(response, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                )
                sequences.append(torch.cat((prefix, answer)))
                labels.append(torch.cat((prefix.new_full(prefix.shape, -100), answer)))
                if "mm_token_type_ids" in batch:
                    types.append(torch.cat((
                        batch["mm_token_type_ids"][i][valid], torch.zeros_like(answer),
                    )))
            width = max(len(ids) for ids in sequences)
            batch["input_ids"] = torch.stack([
                F.pad(ids, (width - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in sequences
            ])
            batch["attention_mask"] = torch.stack([
                F.pad(torch.ones_like(ids), (width - len(ids), 0)) for ids in sequences
            ])
            batch["labels"] = torch.stack([
                F.pad(ids, (width - len(ids), 0), value=-100) for ids in labels
            ])
            if types:
                batch["mm_token_type_ids"] = torch.stack([
                    F.pad(ids, (width - len(ids), 0)) for ids in types
                ])
        device = self.qwen.get_input_embeddings().weight.device
        return {key: value.to(device) for key, value in batch.items()}

    def forward(self, messages, images, responses):
        batch = self.build_inputs(messages, images, responses)
        labels = batch.pop("labels")
        # 不复用上一窗口的旋转位置偏移；每个窗口包含完整的图文上下文。
        self.qwen.model.rope_deltas = None
        hidden = self.qwen.model(
            **batch, use_cache=False, return_dict=True, output_hidden_states=False,
        ).last_hidden_state
        answer_mask = labels[:, 1:].ne(-100)
        logits = self.qwen.lm_head(hidden[:, :-1][answer_mask])
        loss = F.cross_entropy(logits.float(), labels[:, 1:][answer_mask])
        return SimpleNamespace(loss=loss, text_loss=loss, text_tokens=answer_mask.sum())

    @torch.inference_mode()
    def generate(self, messages, images, max_new_tokens=256):
        was_training = self.training
        self.eval()
        try:
            batch = self.build_inputs(messages, images)
            self.qwen.model.rope_deltas = None
            ids = self.qwen.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            )
            # 原生 input_ids 生成包含输入前缀，只解码新增答案。
            responses = self.tokenizer.batch_decode(
                ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            device = batch["input_ids"].device
            shape = (len(responses), FRAME_COUNT)
            result = {
                "responses": responses,
                "progress": torch.full(shape, float("nan"), device=device),
                "done": torch.full(shape, -1, dtype=torch.long, device=device),
                "valid": torch.zeros(shape, dtype=torch.bool, device=device),
                "parse_valid": torch.zeros(shape, dtype=torch.bool, device=device),
            }
            for i, text in enumerate(responses):
                for j, frame in enumerate(parse_frame_predictions(text)):
                    if frame is not None:
                        result["parse_valid"][i, j] = True
                        if frame["valid"]:
                            for key in ("progress", "done"):
                                result[key][i, j] = frame[key]
                            result["valid"][i, j] = True
            return result
        finally:
            self.train(was_training)
