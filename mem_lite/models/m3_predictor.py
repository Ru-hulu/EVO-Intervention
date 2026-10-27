"""原生 Qwen3.5 三视角 ViT + 连续历史输入 + 进度分箱头；不生成文本。"""

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration


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
    """只训练语言LoRA、独立历史投影和线性进度头；M3持有自己的冻结ViT。"""

    def __init__(
        self, model_path, history_dim=None, num_bins=101, dtype=torch.bfloat16,
        lora_rank=16, lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("进度取值点至少需要2个")
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
        )
        self.qwen.requires_grad_(False)
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
        history_dim = history_dim or self.qwen.config.vision_config.out_hidden_size
        self.history_projection = nn.Linear(history_dim, hidden_size)
        self.progress_head = nn.Linear(hidden_size, num_bins)
        self.register_buffer("bin_centers", torch.linspace(0, 1, num_bins))

    def train(self, mode=True):
        super().train(mode)
        self.qwen.model.visual.eval()
        return self

    def forward(self, batch, target_pt=None):
        # 训练、缓存评估和在线推理共用精度策略，不依赖调用方的 autocast。
        with torch.autocast(self.qwen.device.type, dtype=self.qwen.dtype, enabled=self.qwen.dtype != torch.float32):
            return self._forward(batch, target_pt)

    def _forward(self, batch, target_pt=None):
        """接收M3Collator的batch；target_pt只计算软标签CE，不进入模型输入。

        历史视为冻结输入；只投影固定容量张量内的有效 tokens。
        不接状态文本解码器，不添加专家参考，不调用generate或词表输出头。
        """
        ids, attention = batch["input_ids"], batch["attention_mask"]
        targets = None
        if target_pt is not None:
            if target_pt.shape != (len(ids),):
                raise ValueError("target_pt 必须与输入批次逐条对应，形状为 [batch_size]")
            targets = adjacent_bin_targets(target_pt.to(ids.device), len(self.bin_centers))
        embeds = self.qwen.get_input_embeddings()(ids)
        history_mask = batch["history_mask"]
        valid = batch["history_valid_mask"]
        if valid.any():
            history = batch["history_tokens"][valid].detach()
            projected = self.history_projection(history.to(self.history_projection.weight.dtype))
            embeds = embeds.masked_scatter(history_mask[..., None], projected.to(embeds.dtype))

        # inputs_embeds会使原模型无法从input_ids自动推导多模态位置。
        # 在替换后的完整序列上重算原生3D RoPE，历史按文本位置、图像按空间位置编码。
        position_ids, _ = self.qwen.model.get_rope_index(
            ids, batch["mm_token_type_ids"], image_grid_thw=batch["image_grid_thw"], attention_mask=attention,
        )
        hidden = self.qwen.model(
            inputs_embeds=embeds, attention_mask=attention, position_ids=position_ids,
            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
            use_cache=False, return_dict=True,
        ).last_hidden_state
        indices = torch.arange(ids.shape[1], device=ids.device).expand_as(attention)
        last = indices.masked_fill(attention == 0, -1).max(1).values
        pooled = hidden[torch.arange(len(ids), device=ids.device), last]
        logits = self.progress_head(pooled.to(self.progress_head.weight.dtype)).float()
        probabilities = logits.softmax(dim=-1)
        # 逐元素乘加保持FP32；autocast下的矩阵乘法会把最终连续PT降为BF16。
        progress = (probabilities * self.bin_centers.float()).sum(-1)
        result = {"pt": progress, "bin_logits": logits, "bin_probs": probabilities}
        if targets is not None:
            loss_sum = -(targets * logits.log_softmax(dim=-1)).sum()
            result.update(loss=loss_sum / len(ids), loss_sum=loss_sum, num_samples=len(ids))
        return result
