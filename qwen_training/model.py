# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License; see LICENSE.
# Original interface implemented by Shijie LIAN / Huazhong University of Science & Technology (2026).
# Original design and merge by Jinhui YE / HKUST University (2026).
"""上次状态的 JSON Prompt + 当前历史视觉表征 → 当前状态的 JSON 答案。"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from qwen_training.vision_encoder import encode_history, encode_three_views


class QwenProgressModel(nn.Module):
    def __init__(self, model_path, feature_dim=None, dtype=torch.bfloat16, freeze_backbone=False,
                 max_pixels=262144):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, use_fast=True, max_pixels=max_pixels,
        )
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa"
        )
        # encode_images 复用这个编码器；传入 visual_features 时不重复编码。
        self.qwen.model.visual.requires_grad_(False)
        if freeze_backbone:
            self.qwen.requires_grad_(False)

        embedding = self.qwen.get_input_embeddings().weight
        if feature_dim is None:
            feature_dim = self.qwen.config.vision_config.out_hidden_size
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.qwen.config.text_config.hidden_size),
        ).to(device=embedding.device, dtype=embedding.dtype)

    def encode_images(self, images):
        """三视角单帧 → 现有 visual_features 接口；图像模式使用默认 feature_dim。"""
        return encode_three_views(self.qwen.model.visual, self.image_processor, images)

    def encode_history(self, images, frame_mask=None):
        """三视角历史帧 → 含时间编码的 MEM 特征；本阶段仍输出所有帧的 token。"""
        return encode_history(
            self.qwen.model.visual, self.image_processor, images,
            frame_mask=frame_mask,
        )

    def build_inputs(self, messages, visual_features, visual_mask=None, responses=None):
        """拼成 [有效视觉前缀, 聊天 Prompt, 答案]，仅监督答案及结束标记。

        messages 是批量 system/user 对话；responses 是对应的 JSON 答案字符串。
        visual_features 为 [B,N,D]，visual_mask 为 [B,N]；推理时不传 responses。
        融合后的 E_t 使用连续序列位置，不冒充原始图片的 patch 网格。
        """
        if responses is not None and len(responses) != len(messages):
            raise ValueError("responses 和 messages 的样本数必须一致")

        embedding = self.qwen.get_input_embeddings()
        device = embedding.weight.device
        visual = self.visual_adapter(visual_features.to(self.visual_adapter[1].weight))
        # 统一设备和数据类型，再把视觉特征投影到 Qwen 的隐藏维度。
        if visual_mask is None:
            visual_mask = torch.ones(visual.shape[:2], device=device, dtype=torch.bool)
        else:
            visual_mask = visual_mask.to(device=device, dtype=torch.bool)

        sequences, labels = [], []
        for i, (chat, features, mask) in enumerate(zip(messages, visual, visual_mask, strict=True)):
            # Qwen3.5 统一关闭 thinking，空思考前缀属于 Prompt，不参与监督。
            prompt_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_dict=False, add_generation_prompt=True,
                enable_thinking=False,
            ) # 转换为 token ID 序列
            token_ids = prompt_ids
            if responses is not None:
                full_chat = chat + [{"role": "assistant", "content": responses[i]}]
                token_ids = self.tokenizer.apply_chat_template(
                    full_chat, tokenize=True, return_dict=False, add_generation_prompt=False,
                    enable_thinking=False,
                )
                if token_ids[:len(prompt_ids)] != prompt_ids:
                    raise ValueError("聊天模板的训练与推理前缀不一致，无法确定答案位置")
                # 保留 assistant 的结束标记，不监督其后的模板换行。
                stop = token_ids.index(self.tokenizer.eos_token_id, len(prompt_ids)) + 1
                token_ids = token_ids[:stop]

            token_ids = torch.tensor(token_ids, device=device, dtype=torch.long)
            features = features[mask] # 图的token
            sequences.append(torch.cat((features, embedding(token_ids)), dim=0))
            # 视觉embedding 输入的prompt 输出的答案
            if responses is not None:
                text_labels = token_ids.clone() # 输入的prompt 输出的答案
                text_labels[:len(prompt_ids)] = -100 # 把输入的prompt给掩码掉
                visual_labels = token_ids.new_full((features.shape[0],), -100) # 把输入的image的token给掩码掉
                labels.append(torch.cat((visual_labels, text_labels))) # 输入image token 被掩码 输入 prompt 被掩码，答案不掩码

        # 统一左侧补齐，保证批量生成时最后一个位置始终是有效 Prompt。
        max_length = max(sequence.shape[0] for sequence in sequences)
        batch = {
            "inputs_embeds": torch.stack([
                F.pad(sequence, (0, 0, max_length - sequence.shape[0], 0))
                for sequence in sequences
            ]),
            "attention_mask": torch.stack([
                F.pad(torch.ones(sequence.shape[0], device=device, dtype=torch.long),
                      (max_length - sequence.shape[0], 0))
                for sequence in sequences
            ]),
        }
        if responses is not None:
            batch["labels"] = torch.stack([
                F.pad(label, (max_length - label.shape[0], 0), value=-100)
                for label in labels
            ])
        return batch

    def forward(self, messages, visual_features, visual_mask=None, responses=None):
        """训练：model(**inputs, **targets)，返回原生输出，通过 outputs.loss 取损失。"""
        batch = self.build_inputs(messages, visual_features, visual_mask, responses)
        # 即使冻结 Qwen，也要保留这段计算图，让梯度传回视觉适配层。
        return self.qwen(**batch, use_cache=False, return_dict=True)

    @torch.inference_mode()
    def generate(self, messages, visual_features, visual_mask=None, max_new_tokens=128):
        """推理：仅输入 Prompt 和视觉表征，返回生成的答案字符串列表。"""
        was_training = self.training
        self.eval()
        try:
            batch = self.build_inputs(messages, visual_features, visual_mask)
            generated = self.qwen.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            # 只传 inputs_embeds 时，返回的 token ID 序列不含输入前缀。
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        finally:
            self.train(was_training)
