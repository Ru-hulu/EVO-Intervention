"""CPU-only model/interface checks. No training loop or optimizer updates."""
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')

from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import Qwen3VLConfig

from robometer_training.model import (MODEL_PATH, attach_lora, load_adapter, load_model,
                                     load_processor, progress_inputs, progress_values, save_adapter)
from robometer.models.rbm import RBM


def main():
    torch.set_num_threads(2)
    assert sys.prefix == '/root/envs/robometer' and not torch.cuda.is_available()
    assert not any('/autodl-tmp/envs/' in p for p in sys.path)
    for other in ('mem-lite', 'procvlm'):
        assert not os.path.samefile(torch.__file__,
            f'/root/autodl-tmp/envs/{other}/lib/python3.12/site-packages/torch/__init__.py')
    report = {'python': sys.executable, 'gpu_used': False, 'optimizer_steps': 0,
              'offline_load': True, 'isolated_package_files': True,
              'packages': {p: importlib.metadata.version(p) for p in
                           ('torch', 'torchvision', 'transformers', 'peft', 'accelerate')}}
    processor = load_processor(min_pixels=4096, max_pixels=4096)
    real, _ = load_model(processor=processor)
    report['full_checkpoint_parameters'] = sum(p.numel() for p in real.parameters())
    report['checkpoint_all_keys_restored'] = True  # strict load above
    report['special_token_ids'] = {t: processor.tokenizer.convert_tokens_to_ids(t) for t in
                                   ('<|split_token|>', '<|reward_token|>', '<|pref_token|>', '<|sim_token|>', '<|prog_token|>')}
    settings = copy.copy(real.model_config)
    attach_lora(real)
    trainable = {k: p for k, p in real.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in k or k.startswith('progress_head.') for k in trainable)
    report['lora_modules'] = sum(1 for name, _ in real.named_modules() if name.endswith('lora_A.default'))
    report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
    report['progress_bins'] = real.progress_head[-1].out_features
    del real, trainable
    gc.collect()
    print('Full 4B checkpoint loaded offline; LoRA and progress head ready.', flush=True)

    tiny_config = Qwen3VLConfig(
        text_config={'vocab_size': len(processor.tokenizer), 'hidden_size': 64, 'intermediate_size': 128,
                     'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 16, 'max_position_embeddings': 2048,
                     'rope_scaling': {'rope_type': 'default', 'mrope_section': [2, 3, 3], 'mrope_interleaved': True}},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
                       'out_hidden_size': 64, 'patch_size': 16, 'spatial_merge_size': 2,
                       'temporal_patch_size': 2, 'num_position_embeddings': 16, 'deepstack_visual_indexes': []})
    tiny_config._attn_implementation = 'sdpa'
    tiny_config.text_config.use_cache = False
    tiny = RBM(tiny_config, processor, processor.tokenizer,
               base_model_id=settings.base_model_id, model_config=settings)
    reload_base = copy.deepcopy(tiny)
    attach_lora(tiny, rank=2, alpha=4, dropout=0)
    images = [Image.new('RGB', (64, 64), (i * 25, 30, 40)) for i in range(8)]
    batch = progress_inputs(processor, 'Press the button.', images)
    tiny.train()
    output, _ = tiny(**batch, sample_type='progress')
    logits = output.progress_logits['A']
    assert logits.shape == (1, 8, 10)
    loss = torch.nn.functional.cross_entropy(logits[:, -1].float(), torch.tensor([5]))
    assert torch.isfinite(loss)
    loss.backward()  # Small random CPU model only; no optimizer exists.
    gradients = {k: p.grad for k, p in tiny.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    assert all('lora_' in k or k.startswith('progress_head.') for k in gradients)
    assert any('lora_' in k and g.abs().max() > 0 for k, g in gradients.items())
    assert any(k.startswith('progress_head.') and g.abs().max() > 0 for k, g in gradients.items())
    report['tiny_cpu_backward'] = {'loss': loss.item(), 'gradient_tensors': len(gradients),
                                  'shape': list(logits.shape)}
    with torch.no_grad():
        tiny.progress_head[-1].bias.add_(torch.arange(10) / 100)
        # A nonzero synthetic adapter makes the roundtrip detect ignored LoRA files.
        for name, parameter in tiny.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0, 0.01)
    with tempfile.TemporaryDirectory(prefix='robometer-cpu-check-') as directory:
        save_adapter(tiny, directory)
        adapter_keys = load_file(Path(directory) / 'adapter_model.safetensors')
        assert adapter_keys and all('lora_' in key for key in adapter_keys)
        report['adapter_contains_base_weights'] = False
        report['tiny_adapter_bytes'] = sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
        restored = load_adapter(reload_base, directory)
        tiny.eval()
        restored.eval()
        with torch.no_grad():
            expected = progress_values(tiny(**batch, sample_type='progress')[0].progress_logits['A'])
            actual = progress_values(restored(**batch, sample_type='progress')[0].progress_logits['A'])
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
    report['adapter_and_head_roundtrip_exact'] = True
    report['status'] = 'cpu_checks_passed_gpu_preflight_pending'
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
