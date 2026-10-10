# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License; see LICENSE.
# Original interface implemented by Shijie LIAN / Huazhong University of Science & Technology (2026).
# Original design and merge by Jinhui YE / HKUST University (2026).
"""任务提示、可选的上次状态与历史视觉表征 → 状态文本及独立回归的任务进度。"""

from types import SimpleNamespace

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from qwen_training.vision_encoder import encode_history, encode_three_views


class ProgressHead(nn.Module):
    """最后一层全部有效输入 token → 注意力池化 → 三层 MLP → [B] 进度。"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, hidden_states, input_mask):
        hidden_states = hidden_states.to(self.attention_score.weight)  # [B,L,D]。
        scores = self.attention_score(hidden_states).squeeze(-1).float()  # [B,L]。
        weights = scores.masked_fill(~input_mask, float("-inf")).softmax(dim=-1)
        pooled = torch.bmm(weights.to(hidden_states.dtype).unsqueeze(1), hidden_states).squeeze(1)
        # pooled [B,D]；Sigmoid 在 FP32 下计算，输出范围 0～1。
        return self.mlp(pooled).squeeze(-1).float().sigmoid()


class QwenProgressModel(nn.Module):
    def __init__(self, model_path, feature_dim=None, dtype=torch.bfloat16, freeze_backbone=False,
                 max_pixels=262144, lora_rank=0, lora_alpha=32, lora_dropout=0.05,
                 progress_loss_weight=0.3, text_loss_weight=0.7):
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
        if lora_rank:
            # 只给语言层的 Linear 加 LoRA，覆盖全注意力、线性注意力和 MLP。
            # 不改视觉编码器、词嵌入和原语言输出头；原模型接口保持不变。
            targets = [name for name, module in self.qwen.named_modules()
                       if name.startswith("model.language_model.layers.")
                       and isinstance(module, nn.Linear)]
            inject_adapter_in_model(LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=targets, bias="none",
            ), self.qwen)

        embedding = self.qwen.get_input_embeddings().weight
        if feature_dim is None:
            feature_dim = self.qwen.config.vision_config.out_hidden_size
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.qwen.config.text_config.hidden_size),
        ).to(device=embedding.device, dtype=embedding.dtype)
        self.progress_head = ProgressHead(self.qwen.config.text_config.hidden_size).to(
            device=embedding.device, dtype=embedding.dtype,
        )
        self.text_loss_weight = text_loss_weight
        self.progress_loss_weight = progress_loss_weight

    def encode_images(self, images):
        """三视角单帧 → 现有 visual_features 接口；图像模式使用默认 feature_dim。"""
        return encode_three_views(self.qwen.model.visual, self.image_processor, images)

    def encode_history(self, images, frame_mask=None):
        """三视角历史帧 → 融合历史信息的三个当前视角 token。"""
        return encode_history(
            self.qwen.model.visual, self.image_processor, images,
            frame_mask=frame_mask,
        )

    def build_inputs(self, messages, visual_features, visual_mask=None, responses=None):
        """拼成 [有效视觉前缀, 聊天 Prompt, 答案]，仅监督答案及结束标记。

        messages 是批量对话；responses 是包含任务意图和阶段状态的答案字符串。
        visual_features 为 [B,N,D]，visual_mask 为 [B,N]；推理时不传 responses。
        融合后的 E_t 使用连续序列位置，不冒充原始图片的 patch 网格。
        """
        if responses is not None and len(responses) != len(messages):
            raise ValueError("responses 和 messages 的样本数必须一致")

        embedding = self.qwen.get_input_embeddings()
        device = embedding.weight.device
        visual = self.visual_adapter(visual_features.to(self.visual_adapter[1].weight))
        # 适配层可保留 FP32 参数，送入主干的 embedding 与 Qwen 类型一致。
        visual = visual.to(embedding.weight.dtype)
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

    def forward(self, messages, visual_features, visual_mask=None, responses=None, progress=None):
        """responses 为意图和状态文本；progress 为 [B] 浮点标签。都不传时只预测进度。"""
        batch = self.build_inputs(messages, visual_features, visual_mask, responses)
        labels = batch.pop("labels", None)
        # 直接取主干最终输出，不收集前面各层；保留 LoRA 与适配层的梯度。
        hidden_states = self.qwen.model(
            **batch, use_cache=False, return_dict=True, output_hidden_states=False,
        ).last_hidden_state  # [B,L,D]。
        input_mask = batch["attention_mask"].bool()
        if labels is not None:
            input_mask = input_mask & labels.eq(-100)  # 排除 teacher forcing 的答案 token。
        prediction = self.progress_head(hidden_states, input_mask)

        text_loss = progress_loss = loss = None
        text_tokens = None
        if labels is not None:
            # 用前一个位置预测下一个答案 token；只计算受监督位置的词表 logits。
            answer_mask = labels[:, 1:].ne(-100)
            text_tokens = answer_mask.sum()
            logits = self.qwen.lm_head(hidden_states[:, :-1][answer_mask])
            text_loss = F.cross_entropy(logits.float(), labels[:, 1:][answer_mask])
            loss = self.text_loss_weight * text_loss
        if progress is not None:
            progress_loss = F.mse_loss(prediction, progress.to(prediction))
            weighted_loss = self.progress_loss_weight * progress_loss
            loss = weighted_loss if loss is None else loss + weighted_loss
        return SimpleNamespace(
            loss=loss, text_loss=text_loss, progress_loss=progress_loss, progress=prediction,
            text_tokens=text_tokens,
        )

    @torch.inference_mode()
    def generate(self, messages, visual_features, visual_mask=None, max_new_tokens=128):
        """返回 {responses: 意图和阶段文本字符串列表, progress: [B] 回归值}。"""
        was_training = self.training
        visual_was_training = self.qwen.model.visual.training
        self.eval()
        handle = None
        try:
            batch = self.build_inputs(messages, visual_features, visual_mask)
            predictions = []

            def pool_prompt(module, args, output):
                # 只在生成的首轮（完整输入）池化，后续生成 token 不参与进度预测。
                if not predictions:
                    predictions.append(self.progress_head(
                        output.last_hidden_state, batch["attention_mask"].bool(),
                    ))

            # 复用 generate 的首次主干前向，避免为了进度再运行一次 VLM。
            handle = self.qwen.model.register_forward_hook(pool_prompt)
            generated = self.qwen.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                output_hidden_states=False,
            )
            # 只传 inputs_embeds 时，返回的 token ID 序列不含输入前缀。
            return {
                "responses": self.tokenizer.batch_decode(generated, skip_special_tokens=True),
                "progress": predictions[0],
            }
        finally:
            if handle is not None:
                handle.remove()
            self.train(was_training)
            self.qwen.model.visual.train(visual_was_training)
