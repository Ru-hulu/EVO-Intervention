"""Local-only Robometer loading and lightweight progress LoRA checkpoints."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from accelerate import init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoConfig, AutoProcessor
import yaml

from robometer.models.rbm import RBM

MODEL_PATH = '/root/autodl-tmp/models/Robometer-4B'
SPECIAL_TOKENS = ['<|split_token|>', '<|reward_token|>', '<|pref_token|>',
                  '<|sim_token|>', '<|prog_token|>']


def load_processor(directory=MODEL_PATH, **kwargs):
    processor = AutoProcessor.from_pretrained(
        Path(directory) / 'processor', local_files_only=True,
        padding_side='right', do_sample_frames=False, **kwargs)
    # Match the published training code's order; never resize the checkpoint.
    for token in SPECIAL_TOKENS:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({'additional_special_tokens': [token]})
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_model(directory=MODEL_PATH, processor=None):
    directory = Path(directory)
    processor = processor or load_processor(directory)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    config.text_config.use_cache = False
    settings = SimpleNamespace(**yaml.safe_load((directory / 'config.yaml').read_text())['model'])
    settings.use_unsloth = False
    assert len(processor.tokenizer) == config.text_config.vocab_size
    with init_empty_weights():
        model = RBM(config, processor, processor.tokenizer,
                    base_model_id=settings.base_model_id, model_config=settings)
        # Published weights include this legacy head; current upstream removed it.
        # Keep and restore it frozen instead of silently discarding its parameters.
        model.similarity_head = copy.deepcopy(model.preference_head)
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(load_file(directory / shard, device='cpu'))
    model.load_state_dict(state, strict=True, assign=True)
    model.model_dtype = model.model.dtype
    model.eval()
    return model, processor


def attach_lora(model, rank=16, alpha=32, dropout=0.05):
    """Train language LoRA and progress head; freeze vision and unrelated heads."""
    model.requires_grad_(False)
    model.model.language_model = get_peft_model(model.model.language_model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj']))
    model.progress_head.float().requires_grad_(True)
    return model


def save_adapter(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.model.language_model.save_pretrained(directory, save_embedding_layers=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.progress_head.state_dict().items()},
              directory / 'progress_head.safetensors')


def load_adapter(model, directory, trainable=False):
    directory = Path(directory)
    if isinstance(model.model.language_model, PeftModel):
        # Restore in place so Trainer's optimizer keeps the same parameter objects.
        weights = load_peft_weights(str(directory), device='cpu', local_files_only=True)
        set_peft_model_state_dict(model.model.language_model, weights)
        for name, parameter in model.model.language_model.named_parameters():
            parameter.requires_grad_(trainable and 'lora_' in name)
    else:
        model.requires_grad_(False)
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model, directory, is_trainable=trainable, local_files_only=True)
    model.progress_head.float().load_state_dict(load_file(directory / 'progress_head.safetensors'))
    model.progress_head.requires_grad_(trainable)
    return model


def progress_values(logits):
    """Official 10-bin softmax expectation, in [0, 1]."""
    centers = torch.linspace(0, 1, logits.shape[-1], device=logits.device)
    return (logits.float().softmax(-1) * centers).sum(-1)


def progress_messages(task, frame_count):
    """Match upstream's progress prompt and per-frame progress-token placement."""
    prompt = (f"The task for the robot is '{task}'. Given the trajectory video, predict the task "
              'progress at each frame, how far along the robot is towards completing the task, '
              'a float between 0 and 1, where 0 is the starting state and 1 is when the task is '
              'completed. If the robot is not performing the same task, predict 0 progress.')
    content = [{'type': 'text', 'text': prompt}]
    for _ in range(frame_count):
        content.extend([{'type': 'image'}, {'type': 'text', 'text': '<|prog_token|>'}])
    return [{'role': 'user', 'content': content}]


def progress_batch(processor, examples):
    processor.tokenizer.padding_side = 'right'
    texts, images = [], []
    for task, frames in examples:
        texts.append(processor.apply_chat_template(progress_messages(task, len(frames)),
                     tokenize=False, add_generation_prompt=False, add_vision_id=True))
        images.extend(frames)
    return processor(text=texts, images=images, return_tensors='pt', padding=True)


def progress_inputs(processor, task, images):
    return progress_batch(processor, [(task, images)])


def current_logits(model, inputs):
    output, _ = model(**inputs, sample_type='progress')
    return output.progress_logits['A'][:, -1].float()


def progress_targets(pt, bins=10):
    """C51 projection matching upstream; its expectation preserves continuous PT."""
    position = pt.float() * (bins - 1)
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower
    targets = torch.zeros((*pt.shape, bins), device=pt.device, dtype=torch.float32)
    targets.scatter_add_(-1, lower.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    targets.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return targets


def progress_loss(logits, pt):
    return torch.nn.functional.cross_entropy(logits.float(), progress_targets(pt, logits.shape[-1]))
