"""Qwen3.5 原生多图事件切分：冻结基座，仅训练语言层 LoRA。"""

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.nn import functional as F
from transformers import GenerationConfig, Qwen3_5ForConditionalGeneration


class M1Segmenter(nn.Module):
    def __init__(
        self, model_path, dtype=torch.bfloat16, lora_rank=16, lora_alpha=32,
        lora_dropout=0.05, gradient_checkpointing=True,
    ):
        super().__init__()
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
        )
        self.qwen.requires_grad_(False)
        # 精确匹配语言层；不会把视觉编码器、视觉合并层或输出头纳入 LoRA。
        targets = [name for name, module in self.qwen.named_modules()
                   if name.startswith("model.language_model.layers.") and isinstance(module, nn.Linear)]
        inject_adapter_in_model(LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias="none",
        ), self.qwen)
        for parameter in self.qwen.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if gradient_checkpointing:
            self.qwen.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        self.qwen.model.visual.eval()

    def train(self, mode=True):
        super().train(mode)
        self.qwen.model.visual.eval()
        return self

    @property
    def visual_feature_dim(self):
        return self.qwen.config.vision_config.out_hidden_size

    @torch.no_grad()
    def extract_frame_features(self, pixel_values, image_grid_thw):
        """复用冻结 ViT，返回按输入图像顺序排列的逐帧空间 tokens。

        取视觉合并层之后、语言模型之前的原生图像表征；不是每帧池化为一个向量。
        使用 no_grad 而非 inference_mode，输出可参与后续可训练事件编码器的反向计算。
        """
        self.qwen.model.visual.eval()
        output = self.qwen.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        return list(output.pooler_output)

    def forward(self, batch, expert_tokens=None):
        """专家接口预留但禁用；返回答案 token 的损失和，便于跨微批次正确归一化。"""
        if expert_tokens is not None:
            raise NotImplementedError("专家 tokens 尚未接入；当前阶段须传 None")
        labels = batch["labels"]
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        # 每个窗口独立计算多模态位置，不复用上一窗口的位置偏移。
        self.qwen.model.rope_deltas = None
        hidden = self.qwen.model(
            **inputs, use_cache=False, return_dict=True, output_hidden_states=False,
        ).last_hidden_state
        supervised = labels[:, 1:].ne(-100)
        # 仅答案位置经过词表头，避免为全部图像/提示 tokens 分配词表 logits。
        logits = self.qwen.lm_head(hidden[:, :-1][supervised])
        loss_sum = F.cross_entropy(logits.float(), labels[:, 1:][supervised], reduction="sum")
        return {"loss_sum": loss_sum, "text_tokens": supervised.sum()}

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens=96):
        """从无答案的观测提示生成JSON，仅返回新增token；各窗口独立建立KV缓存。"""
        if "labels" in batch:
            raise ValueError("生成评测只能输入提示，不能输入带答案的训练批次")
        self.qwen.model.rope_deltas = None
        config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=self.qwen.generation_config.eos_token_id,
            pad_token_id=self.qwen.generation_config.pad_token_id,
        )
        output = self.qwen.generate(**batch, generation_config=config, logits_to_keep=1)
        return output[:, batch["input_ids"].shape[1]:]
