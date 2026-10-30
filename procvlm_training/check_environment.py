"""CPU-only readiness checks; never run an optimizer or touch the shared GPU."""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig

from procvlm_training.window import ObservationWindow, history_window


def attach_lora(model, rank):
    for parameter in model.parameters():
        parameter.requires_grad = False
    wrapped = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none', task_type='CAUSAL_LM',
    ))
    for module in (model.pooler, model.value_head):
        module.requires_grad_(True)
    return wrapped


def main():
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    sys.path.insert(0, config['upstream_path'])
    from evqa.model import ProcVLMWithValueHead

    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    assert sys.prefix == '/root/autodl-tmp/envs/procvlm'
    assert not any('/envs/mem-lite' in entry for entry in sys.path)
    old_torch = Path('/root/autodl-tmp/envs/mem-lite/lib/python3.12/site-packages/torch/__init__.py')
    assert not os.path.samefile(torch.__file__, old_torch)
    report = {'python': sys.executable, 'gpu_used': False, 'optimizer_steps': 0,
              'isolated_from_mem_lite': True,
              'packages': {p: importlib.metadata.version(p) for p in
                           ('torch', 'torchvision', 'transformers', 'peft', 'accelerate')}}

    source = Path(config['source_dataset'])
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['sample_fps'] == 5
    checked = 0
    for split in ('train', 'validation'):
        rows = [json.loads(line) for line in (source / f'{split}.jsonl').read_text().splitlines()]
        episode = None
        group = []
        online = ObservationWindow(8)
        for row in rows:
            if row['episode_id'] != episode:
                episode, group = row['episode_id'], []
            # The fifth image offset in the existing eight-image dataset is current cam_high.
            item = (row['frame_index'], row['image'][5])
            group.append(item)
            offline = history_window(group, len(group) - 1)
            assert online.push(episode, item) == offline
            assert all(frame <= row['frame_index'] and Path(path).is_file() for frame, path in offline)
            checked += 1
    report['causal_windows_checked'] = checked
    report['sample_fps'], report['window_size'] = 5, 8

    processor = AutoProcessor.from_pretrained(config['model_path'], min_pixels=4096, max_pixels=4096)
    tokenizer = processor.tokenizer
    head_file = Path(config['model_path']) / 'procvlm_extra/procvlm_head.pt'
    head = torch.load(head_file, map_location='cpu', weights_only=True)
    real = ProcVLMWithValueHead.from_pretrained(
        config['model_path'], tokenizer, device_map='cpu', torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    restored = real.state_dict()
    assert head and all(k in restored and torch.equal(restored[k], v.to(restored[k].dtype)) for k, v in head.items())
    report['published_head_tensors_loaded'] = len(head)
    adapted = attach_lora(real, 64)
    report['lora_and_head_trainable_parameters'] = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    report['lora_modules'] = sum(1 for name, _ in adapted.named_modules() if name.endswith('lora_A.default'))
    del adapted, real, restored, head
    gc.collect()

    # Exercise the real ProcVLM/PEFT/processor interfaces on a small random CPU model.
    tiny_config = Qwen3VLConfig(
        text_config={'vocab_size': 151936, 'hidden_size': 64, 'intermediate_size': 128,
                     'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 16, 'max_position_embeddings': 2048,
                     'rope_scaling': {'rope_type': 'default', 'mrope_section': [2, 3, 3], 'mrope_interleaved': True}},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
                       'out_hidden_size': 64, 'patch_size': 16, 'spatial_merge_size': 2,
                       'temporal_patch_size': 2, 'num_position_embeddings': 16, 'deepstack_visual_indexes': []},
    )
    tiny_config._attn_implementation = 'sdpa'
    tiny = ProcVLMWithValueHead(tiny_config, tokenizer, value_dropout=0, value_noise_std=0)
    messages = [{'role': 'user', 'content': [{'type': 'image'} for _ in range(8)] +
                [{'type': 'text', 'text': 'Estimate current task completion.'}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer = 'Remaining action: place the object. <progress>25.00%</progress>' + tokenizer.eos_token
    batch = processor(text=[prompt + answer], images=[Image.new('RGB', (64, 64)) for _ in range(8)],
                      return_tensors='pt', padding=True)
    labels = torch.full_like(batch.input_ids, -100)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    assert batch.input_ids[0, -len(answer_ids):].tolist() == answer_ids
    labels[0, -len(answer_ids):] = torch.tensor(answer_ids)
    with tempfile.TemporaryDirectory(prefix='procvlm-cpu-check-') as directory:
        base, adapter = Path(directory) / 'base', Path(directory) / 'adapter'
        tiny.save_pretrained(base, save_value_head=True)
        processor.save_pretrained(base)
        tiny.config._name_or_path = str(base)
        wrapped = attach_lora(tiny, 2)
        wrapped.peft_config['default'].base_model_name_or_path = str(base)
        wrapped.train()
        result = wrapped(**batch, labels=labels)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        gradients = {name: p.grad for name, p in wrapped.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        assert any('lora_' in name and g.abs().max() > 0 for name, g in gradients.items())
        assert any('value_head.' in name and g.abs().max() > 0 for name, g in gradients.items())
        assert all(any(part in name for part in ('lora_', 'value_head.', 'pooler.')) for name in gradients)
        report['tiny_cpu_backward'] = {'loss': float(result.loss.detach()), 'gradient_tensors': len(gradients)}
        # Deliberately differ from the base sidecar to detect silently ignored adapter head keys.
        with torch.no_grad():
            tiny.value_head[-1].bias.fill_(0.12345)
        wrapped.save_pretrained(adapter)
        sidecar = adapter / 'procvlm_extra'
        sidecar.mkdir()
        target = {k: v for k, v in tiny.state_dict().items() if k.startswith(('pooler.', 'value_head.'))}
        torch.save(target, sidecar / 'procvlm_head.pt')
        loaded = ProcVLMWithValueHead.from_pretrained(str(adapter), tokenizer,
                    device_map='cpu', torch_dtype=torch.float32, attn_implementation='sdpa')
        restored = loaded.get_base_model().state_dict()
        assert all(torch.equal(value, restored[key]) for key, value in target.items())
        report['adapter_and_head_reload_exact'] = True

    report['status'] = 'cpu_checks_passed_gpu_preflight_pending'
    output = Path(sys.argv[1])
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
