"""原生 Qwen3.5 当前三视角 + 层间门控历史 cross-attention + 进度分箱头。"""

from functools import partial

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5ForConditionalGeneration

from .cross_attention import GatedHistoryAttention, HistoryDecoderLayer
from .linear_memory import LinearMemory, linear_decoder_forward


def adjacent_bin_targets(target_pt, num_bins=101):
    """把[0,1]连续PT分配到相邻两个取值点，端点0/1退化为单点标签。"""
    if target_pt.ndim != 1 or not (torch.isfinite(target_pt) & (target_pt >= 0) & (target_pt <= 1)).all():
        raise ValueError("target_pt 必须是 [batch_size] 的有限数值，范围为[0,1]")
    if num_bins < 2:
        raise ValueError("进度取值点至少需要2个")
    position = target_pt.float() * (num_bins - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(num_bins - 1)
    fraction = position - left
    targets = position.new_zeros((len(position), num_bins))
    targets.scatter_add_(1, left[:, None], (1 - fraction)[:, None])
    targets.scatter_add_(1, right[:, None], fraction[:, None])
    return targets


class M3ProgressPredictor(nn.Module):
    """只训练语言 LoRA、历史 cross-attention 和进度头，视觉编码器冻结。"""

    def __init__(
        self, model_path, history_dim=None, num_bins=101, dtype=torch.bfloat16,
        lora_rank=16, lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True,
        cross_attention_layers=(8, 16, 24), cross_attention_dim=512, cross_attention_heads=8,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("进度取值点至少需要2个")
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
        )
        self.qwen.requires_grad_(False)
        layers = self.qwen.model.language_model.layers
        self.cross_attention_layers = tuple(cross_attention_layers)
        if (not self.cross_attention_layers or len(set(self.cross_attention_layers)) != len(self.cross_attention_layers)
                or any(not isinstance(i, int) or not 1 <= i < len(layers) for i in self.cross_attention_layers)):
            raise ValueError("cross_attention_layers 使用从1开始的层号，必须位于模型层之间且不重复")
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
        hidden_size = self.qwen.config.text_config.hidden_size
        self.history_dim = history_dim or self.qwen.config.vision_config.out_hidden_size
        for number in self.cross_attention_layers:
            layers[number - 1] = HistoryDecoderLayer(layers[number - 1], GatedHistoryAttention(
                hidden_size, self.history_dim, cross_attention_dim, cross_attention_heads,
            ))
        self.progress_head = nn.Linear(hidden_size, num_bins)
        self.register_buffer("bin_centers", torch.linspace(0, 1, num_bins))

    @property
    def memory_adapters(self):
        return {str(i): self.qwen.model.language_model.layers[i - 1].memory_attention
                for i in self.cross_attention_layers}

    def train(self, mode=True):
        super().train(mode)
        self.qwen.model.visual.eval()
        return self

    def forward(self, batch, target_pt=None, state=None):
        """batch 为一帧批次或按时间排列的批次列表；state 按 batch 槽位对应视频。

        不传 state 表示新视频；返回的新 state 显式交给下一次调用，模型不持有会话状态。
        """
        # 训练、缓存评估和在线推理共用精度策略，不依赖调用方的 autocast。
        with torch.autocast(self.qwen.device.type, dtype=self.qwen.dtype, enabled=self.qwen.dtype != torch.float32):
            return self._forward(batch, target_pt, state)

    def _frame(self, batch, state):
        """接收M3Collator的batch；target_pt只计算软标签CE，不进入模型输入。

        历史视为冻结的外部 KV 输入，不占主序列位置。
        不接状态文本解码器，不添加专家参考，不调用generate或词表输出头。
        """
        ids, attention = batch["input_ids"], batch["attention_mask"]
        # 主序列只含任务和当前图像，原生多模态 RoPE 不随历史长度平移。
        position_ids, _ = self.qwen.model.get_rope_index(
            ids, batch["mm_token_type_ids"], image_grid_thw=batch["image_grid_thw"], attention_mask=attention,
        )
        language = self.qwen.model.language_model
        hidden = language.embed_tokens(ids)
        image_features = self.qwen.model.get_image_features(
            batch["pixel_values"], batch["image_grid_thw"], return_dict=True,
        ).pooler_output
        image_features = torch.cat(image_features, dim=0).to(hidden.device, hidden.dtype)
        image_mask, _ = self.qwen.model.get_placeholder_mask(ids, inputs_embeds=hidden, image_features=image_features)
        hidden = hidden.masked_scatter(image_mask, image_features)
        valid = attention.bool()
        if "sample_valid" in batch:
            valid = valid & batch["sample_valid"][:, None]
        hidden = hidden.masked_fill(~valid[..., None], 0)
        position_embeddings = language.rotary_emb(hidden, position_ids)
        length = ids.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()
        # 仅当前帧内部因果注意力，Full Attention 从不收到跨帧 KV。
        mask = causal[None, None] & valid[:, None, None, :]
        updated = LinearMemory()
        for index, layer in enumerate(language.layers):
            decoder = layer.decoder if isinstance(layer, HistoryDecoderLayer) else layer
            if decoder.layer_type == "linear_attention":
                old = state.layers.get(index, (None, None))
                function = partial(linear_decoder_forward, decoder)
                if language.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    hidden, conv, recurrent = checkpoint(function, hidden, valid, *old, use_reentrant=False)
                else:
                    hidden, conv, recurrent = function(hidden, valid, *old)
                updated.layers[index] = (conv, recurrent)
            else:
                hidden = decoder(hidden, position_embeddings=position_embeddings, attention_mask=mask,
                                 position_ids=position_ids, past_key_values=None, use_cache=False)
            if isinstance(layer, HistoryDecoderLayer):
                hidden = layer.memory_attention(hidden, batch["history_tokens"], batch["history_valid_mask"], valid)
        hidden = language.norm(hidden)
        indices = torch.arange(ids.shape[1], device=ids.device).expand_as(attention)
        last = indices.masked_fill(attention == 0, -1).max(1).values
        pooled = hidden[torch.arange(len(ids), device=ids.device), last]
        return self.progress_head(pooled.to(self.progress_head.weight.dtype)).float(), updated

    def _forward(self, batch, target_pt, state):
        batches = batch if isinstance(batch, (list, tuple)) else [batch]
        state = LinearMemory() if state is None else state
        outputs, validity = [], []
        for inputs in batches:
            logits, state = self._frame(inputs, state)
            outputs.append(logits)
            validity.append(inputs.get("sample_valid", torch.ones(len(logits), dtype=torch.bool, device=logits.device)))
        logits, valid = torch.cat(outputs), torch.cat(validity)
        probabilities = logits.softmax(dim=-1)
        # 逐元素乘加保持FP32；autocast下的矩阵乘法会把最终连续PT降为BF16。
        progress = (probabilities * self.bin_centers.float()).sum(-1)
        result = {"pt": progress, "bin_logits": logits, "bin_probs": probabilities, "state": state,
                  "sample_valid": valid}
        if target_pt is not None:
            if target_pt.shape != (len(logits),):
                raise ValueError("target_pt 必须与输入批次逐条对应，形状为 [batch_size]")
            targets = adjacent_bin_targets(target_pt.to(logits.device), len(self.bin_centers))
            loss_sum = (-(targets * logits.log_softmax(dim=-1)).sum(-1) * valid).sum()
            count = valid.sum()
            result.update(loss=loss_sum / count.clamp_min(1), loss_sum=loss_sum, num_samples=count)
        return result
