"""独立冻结的 Qwen3.5 文本解码器：历史 tokens → 累计状态文本监督。"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, Qwen3_5ForCausalLM


class StateTextDecoder(nn.Module):
    """只训练历史投影层；语言基座、词嵌入和语言头始终冻结，不接入 M1 LoRA。"""

    def __init__(self, model_path, history_dim=512, dtype=torch.bfloat16, max_sequence_length=2048):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        text_config = config.get_text_config()
        text_config.tie_word_embeddings = config.tie_word_embeddings
        # Transformers 原生将 model.language_model.* 映射到 model.*，忽略视觉权重。
        self.language_model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, config=text_config, dtype=dtype,
            attn_implementation="sdpa", local_files_only=True,
        )
        self.language_model.requires_grad_(False)
        self.language_model.eval()
        self.history_projection = nn.Linear(history_dim, text_config.hidden_size)
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("状态文本解码器需要 tokenizer 的结束符")

        # 占位字符串只用于定位连续 embedding 的插入处，不添加词表项或占位 token。
        marker = "{history_embeddings}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"描述截至目前已完成的事件状态，仅输出状态描述。\n历史信息：\n{marker}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        before, after = prompt.split(marker)
        for name, text in (("prefix_ids", before), ("suffix_ids", after)):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self.register_buffer(name, torch.tensor(ids, dtype=torch.long), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.language_model.eval()
        return self

    def build_prompt(self, history_tokens):
        """固定指令与连续历史embeddings；不接收真值状态文本。"""
        if history_tokens.ndim != 3 or history_tokens.shape[0] < 1 or history_tokens.shape[1] != 128:
            raise ValueError("history_tokens 应为 [状态文本数, 128, history_dim]")
        embedding = self.language_model.get_input_embeddings()
        history = self.history_projection(history_tokens.to(self.history_projection.weight.dtype))
        history = history.to(embedding.weight.dtype)
        prefix, suffix = embedding(self.prefix_ids), embedding(self.suffix_ids)
        batch_size = len(history)
        prompt = torch.cat((prefix.expand(batch_size, -1, -1), history, suffix.expand(batch_size, -1, -1)), dim=1)
        if prompt.shape[1] > self.max_sequence_length:
            raise ValueError("历史提示超过max_sequence_length，不能静默截断")
        return {"inputs_embeds": prompt,
                "attention_mask": torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device)}

    def build_inputs(self, history_tokens, state_descriptions):
        """拼接固定提示、128个连续历史 embeddings、答案及 EOS；只监督答案和 EOS。"""
        if len(history_tokens) != len(state_descriptions):
            raise ValueError("历史数量与状态文本数量不一致")
        prompts = self.build_prompt(history_tokens)["inputs_embeds"]
        embedding = self.language_model.get_input_embeddings()
        prompt_length = prompts.shape[1]
        sequences, labels = [], []
        for prompt, description in zip(prompts, state_descriptions, strict=True):
            answer = self.tokenizer.encode(description, add_special_tokens=False)
            answer.append(self.tokenizer.eos_token_id)
            if prompt_length + len(answer) > self.max_sequence_length:
                raise ValueError("状态文本序列超过 max_sequence_length，不能静默截断历史或监督文本")
            answer_ids = torch.tensor(answer, dtype=torch.long, device=prompts.device)
            sequences.append(torch.cat((prompt, embedding(answer_ids))))
            labels.append(F.pad(answer_ids, (prompt_length, 0), value=-100))

        # 训练使用右侧 padding；答案只依赖其前文，不受末尾补齐位置影响。
        width = max(len(sequence) for sequence in sequences)
        inputs_embeds = torch.stack([F.pad(sequence, (0, 0, 0, width - len(sequence))) for sequence in sequences])
        labels = torch.stack([F.pad(label, (0, width - len(label)), value=-100) for label in labels])
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=prompts.device)
        attention_mask = (torch.arange(width, device=prompts.device)[None, :] < lengths[:, None]).long()
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
        return {
            "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
            "position_ids": position_ids, "labels": labels,
        }

    def forward(self, history_tokens, state_descriptions):
        """返回 token CE 总和与监督 token 数；调用方跨前缀/批次汇总后归一化。"""
        batch = self.build_inputs(history_tokens, state_descriptions)
        labels = batch.pop("labels")[:, 1:]
        # 冻结参数不等于关闭 autograd：文本损失必须回传至投影层、M2、事件编码器。
        hidden = self.language_model.model(**batch, use_cache=False).last_hidden_state
        supervised = labels != -100
        logits = self.language_model.lm_head(hidden[:, :-1][supervised])
        loss_sum = F.cross_entropy(logits.float(), labels[supervised], reduction="sum")
        return {"loss_sum": loss_sum, "text_tokens": supervised.sum()}

    @torch.inference_mode()
    def generate(self, history_tokens, max_new_tokens=256):
        prompt = self.build_prompt(history_tokens)
        if prompt["inputs_embeds"].shape[1] + max_new_tokens > self.max_sequence_length:
            raise ValueError("提示与生成预算超过max_sequence_length")
        config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
                                  eos_token_id=self.tokenizer.eos_token_id,
                                  pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                                else self.tokenizer.eos_token_id))
        # 只传inputs_embeds时，原生generate返回新增token，不包含提示前缀。
        return self.language_model.generate(**prompt, generation_config=config, logits_to_keep=1)
