"""显式加载 Qwen3-VL；只向语言注意力层注入 LoRA。"""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_model(config, *, trainable=False, adapter=None, device="cuda"):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_path"], dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa", local_files_only=True,
    ).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        targets = [name for name, layer in model.named_modules()
                   if name.startswith("model.language_model.layers.") and isinstance(layer, torch.nn.Linear)
                   and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        if not targets:
            raise ValueError("No Qwen3-VL language attention targets found")
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules=targets, bias="none", **config["lora"],
        ))
    processor = AutoProcessor.from_pretrained(
        config["model_path"], max_pixels=config["max_pixels"], min_pixels=config["min_pixels"],
        local_files_only=True,
    )
    return model, processor


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
