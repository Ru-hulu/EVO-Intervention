"""Apply the small compatibility fixes to our pinned, isolated upstream checkout."""
from pathlib import Path
import sys


def replace(path, old, new):
    text = path.read_text()
    if old not in text:
        if new in text:
            return
        raise ValueError(f'Unexpected upstream source: {path}')
    path.write_text(text.replace(old, new))


def main(root):
    model = root / 'evqa/model.py'
    # Keep the existing public inference mapping, and train against that same mapping.
    replace(model, 'torch.sigmoid(value_logits.float() * 1.2 - 0.1).clamp(0.0, 1.0)',
            'torch.sigmoid(value_logits.float())')
    # The sidecar keys belong to ProcVLM, not PEFT's prefixed wrapper state dict.
    replace(model, '                model.load_state_dict(head_state, strict=False)\n\n            # LoRA adapter',
            '                base_procvlm_model.load_state_dict(head_state, strict=False)\n\n            # LoRA adapter')
    replace(model, 'attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
    trainer = root / 'evqa/train/trainer.py'
    replace(trainer, 'from flash_attn.flash_attn_interface import flash_attn_varlen_func',
            'try:\n    from flash_attn.flash_attn_interface import flash_attn_varlen_func\n'
            'except ImportError:\n    flash_attn_varlen_func = None  # SDPA does not use this packed-attention path.')
    replace(root / 'evqa/train/train_qwen.py', 'def train(attn_implementation="flash_attention_2"):',
            'def train(attn_implementation="sdpa"):')
    print('Applied: consistent value mapping, LoRA head reload, SDPA defaults.')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
