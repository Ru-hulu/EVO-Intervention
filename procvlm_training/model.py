"""ProcVLM LoRA, FP32 progress heads, and generated-answer inference shared online/offline."""
import json
from pathlib import Path
import sys

from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
import torch
from transformers import AutoProcessor

from .data import ProgressCollator, parse_progress
from .window import ObservationWindow, sample_frame_index


def base_model(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def load_heads(model, directory):
    state = torch.load(Path(directory) / 'procvlm_extra/procvlm_head.pt', map_location='cpu', weights_only=True)
    base = base_model(model)
    for name in ('pooler', 'value_head'):
        module = getattr(base, name).float()
        module.load_state_dict({key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + '.')}, strict=True)


def attach_lora(model, rank=16, alpha=32, dropout=0.05):
    model.requires_grad_(False)
    targets = [name for name, _ in model.named_modules() if '.language_model.' in name and
               name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    if not targets:
        raise ValueError('No language attention projection modules found')
    wrapped = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                    target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    model.pooler.float().requires_grad_(True)
    model.value_head.float().requires_grad_(True)
    return wrapped


def load_model(config, *, adapter=None, trainable=False, device='cpu', dtype=torch.bfloat16):
    sys.path.insert(0, config['upstream_path'])
    from evqa.model import ProcVLMWithValueHead
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True,
        min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
    model = ProcVLMWithValueHead.from_pretrained(config['model_path'], processor.tokenizer,
        local_files_only=True, device_map=device, torch_dtype=dtype,
        attn_implementation=config['attn_implementation'], **config['value_head'])
    load_heads(model, config['model_path'])
    model.requires_grad_(False)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, local_files_only=True)
        load_heads(model, adapter)
        base_model(model).pooler.requires_grad_(trainable)
        base_model(model).value_head.requires_grad_(trainable)
    elif trainable:
        model = attach_lora(model, **config['lora'])
    model.config.use_cache = not trainable
    model.config.text_config.use_cache = not trainable
    model.eval()
    return model, processor


def save_adapter(model, processor, directory, config=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, save_embedding_layers=False)
    base = base_model(model)
    state = {f'{name}.{key}': value.detach().cpu().contiguous()
             for name in ('pooler', 'value_head') for key, value in getattr(base, name).state_dict().items()}
    (directory / 'procvlm_extra').mkdir(exist_ok=True)
    torch.save(state, directory / 'procvlm_extra/procvlm_head.pt')
    processor.save_pretrained(directory)
    if config is not None:
        (directory / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))


def restore_adapter_in_place(model, directory):
    weights = load_peft_weights(str(directory), device='cpu', local_files_only=True)
    set_peft_model_state_dict(model, weights)
    load_heads(model, directory)


def to_device(inputs, model):
    device = next(model.parameters()).device
    dtype = base_model(model).model.dtype
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() and key.startswith('pixel_values') else value.dtype)
            for key, value in inputs.items()}


@torch.inference_mode()
def predict_batch(model, collator, rows, *, max_new_tokens=4096, enable_value_head=True):
    model.eval()
    # Inference encoding reads only task/images, never PT or the remaining-action targets.
    inputs = to_device(collator.encode(rows, supervised=False), model)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    trimmed = generated[:, inputs['input_ids'].shape[1]:]
    texts = collator.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    values = [None] * len(rows)
    if enable_value_head:
        visual = {key: value for key, value in inputs.items() if key in
                  ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw')}
        # The head sees GENERATED reasoning/progress, not the teacher-forced reference answer.
        values = (base_model(model)._predict_progress_values(generated, visual_kwargs=visual).float() / 100).tolist()
    output = []
    for text, value in zip(texts, values):
        text_pt = parse_progress(text)
        prediction = value if enable_value_head and text_pt is not None else text_pt
        output.append({'predicted_pt': prediction, 'text_predicted_pt': text_pt, 'value_head_pt': value,
                       'generated_text': text, 'progress_parse_failed': text_pt is None})
    return output


class OnlineProgressPredictor:
    def __init__(self, model, processor, config):
        self.model, self.config = model, config
        self.collator = ProgressCollator(processor, config['max_length'])
        self.window = ObservationWindow(config['window_size'])
        self.raw_episode = None
        self.sample_number = 0

    def push_sample(self, episode_id, frame_index, image, task):
        """Accept one already-sampled 5-FPS observation; no labels/goal images are accepted."""
        if self.window.episode_id == episode_id and self.window.frames and frame_index <= self.window.frames[-1][0]:
            raise ValueError('Sampled frame indices must increase within an episode')
        window = self.window.push(episode_id, (frame_index, image))
        row = {'task': task, 'image': [path for _, path in window]}
        result = predict_batch(self.model, self.collator, [row], **self.config['inference'])[0]
        return {'frame_index': frame_index, 'window_frame_indices': [frame for frame, _ in window], **result}

    def push_frame(self, episode_id, frame_index, image, task, source_fps):
        """Accept sequential raw-video frames; select the same ceil-based 5-FPS grid as preparation."""
        if self.raw_episode != episode_id:
            self.raw_episode, self.sample_number = episode_id, 0
            self.window = ObservationWindow(self.config['window_size'])
        expected = sample_frame_index(self.sample_number, source_fps, self.config['sample_fps'])
        if frame_index < expected:
            return None
        if frame_index != expected:
            raise ValueError('A required sampled frame is missing')
        self.sample_number += 1
        return self.push_sample(episode_id, frame_index, image, task)
