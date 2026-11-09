"""Native Qwen3-VL GRM SFT with fixed-start, same-episode goal conditioning."""
from functools import lru_cache
from pathlib import Path
import json
import math
import re

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

VIEWS = ['cam_high', 'cam_high', 'cam_high', 'cam_left_wrist',
         'cam_right_wrist', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']
TEMPLATE = Path(__file__).with_name('prompt_template.txt').read_text()


def answer(pt):
    if not math.isfinite(pt) or not 0 <= pt <= 1:
        raise ValueError(f'Invalid PT: {pt}')
    return f'<score>{100 * pt:+.1f}%</score>'


def parse_score(text):
    match = re.fullmatch(r'\s*<score>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))%\s*</score>\s*', text)
    if not match:
        raise ValueError(f'Invalid GRM answer: {text!r}')
    value = float(match.group(1)) / 100
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError(f'Out-of-range GRM answer: {text!r}')
    return value


def messages(task, target=None):
    parts = TEMPLATE.format(task_description=task).split('<image>')
    assert len(parts) == 9
    content = []
    for i, part in enumerate(parts):
        content.append({'type': 'text', 'text': part})
        if i < 8:
            content.append({'type': 'image'})
    result = [{'role': 'user', 'content': content}]
    if target is not None:
        result.append({'role': 'assistant', 'content': target})
    return result


def load_processor(path, min_pixels=4096, max_pixels=65536):
    processor = AutoProcessor.from_pretrained(path, local_files_only=True,
                    min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = 'right'
    return processor


def load_model(path, lora=None, adapter=None):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa')
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if adapter:
        return PeftModel.from_pretrained(model, adapter, is_trainable=bool(lora), local_files_only=True)
    if lora:
        model.requires_grad_(False)
        # Match language modules explicitly; never attach adapters to the vision encoder.
        targets = [name for name, module in model.named_modules()
                   if '.language_model.' in name and isinstance(module, torch.nn.Linear)
                   and name.rsplit('.', 1)[-1] in lora['target_modules']]
        assert targets
        model = get_peft_model(model, LoraConfig(
            r=lora['rank'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
            target_modules=targets, task_type='CAUSAL_LM', bias='none'))
        assert all('lora_' in n and '.language_model.' in n
                   for n, p in model.named_parameters() if p.requires_grad)
    return model


class Dataset:
    def __init__(self, directory, split):
        directory = Path(directory)
        self.episodes = json.loads((directory / 'episodes.json').read_text())
        self.rows = [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        e = self.episodes[row['episode_id']]
        def paths(i):
            return [str(Path(e['source_dir']) / x) if x else None for x in e['native_images'][i]]
        start, end, current = paths(0), paths(len(e['native_images']) - 1), paths(row['source_sample_index'])
        return dict(row, task=e['task_prompt'], image=[start[0], end[0], *start, *current],
                    pixel_cache=e['pixel_cache'],
                    pixel_indices=[0, e['end_pixel_index'], 0, 0, 0] + [row['pixel_index']] * 3)


@lru_cache(maxsize=4)
def pixel_cache(path):
    return load_file(path)


def raw_images(row):
    result = []
    for path in row['image']:
        if path is None:
            result.append(Image.new('RGB', (64, 64), (0, 0, 0)))
        else:
            with Image.open(path) as im:
                result.append(im.convert('RGB'))
    return result


class Collator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def encode(self, rows, supervised=True, cached=True):
        processor = self.processor
        features, prompt_features, labels, pixels, grids = [], [], [], [], []
        for row in rows:
            prompt = processor.apply_chat_template(messages(row['task']), tokenize=False, add_generation_prompt=True)
            full = processor.apply_chat_template(messages(row['task'], row['answer']), tokenize=False,
                                                  add_generation_prompt=False) if supervised else prompt
            if cached:
                cache = pixel_cache(row['pixel_cache'])
                gg = torch.stack([cache['grids_' + v][i] for v, i in zip(VIEWS, row['pixel_indices'])])
                pp = torch.cat([cache['pixels_' + v][i] for v, i in zip(VIEWS, row['pixel_indices'])])
                def expand(text):
                    marker = processor.image_token
                    assert text.count(marker) == 8
                    for grid in gg:
                        n = int(grid.prod()) // processor.image_processor.merge_size ** 2
                        text = text.replace(marker, '<|placeholder|>' * n, 1)
                    return text.replace('<|placeholder|>', marker)
                encoded = processor.tokenizer(expand(full), add_special_tokens=False)
                prefix = processor.tokenizer(expand(prompt), add_special_tokens=False)
            else:
                encoded = processor(text=[full], images=raw_images(row), return_tensors='pt')
                prefix = processor(text=[prompt], images=raw_images(row), return_tensors='pt')
                pp, gg = encoded['pixel_values'], encoded['image_grid_thw']
                encoded = {k: encoded[k][0].tolist() for k in ['input_ids', 'attention_mask']}
                prefix = {k: prefix[k][0].tolist() for k in ['input_ids', 'attention_mask']}
            assert encoded['input_ids'][:len(prefix['input_ids'])] == prefix['input_ids']
            if len(encoded['input_ids']) > self.max_length:
                raise ValueError('Input too long; never truncate any of the eight images')
            features.append(encoded); prompt_features.append(prefix)
            pixels.append(pp); grids.append(gg)
            if supervised:
                labels.append([-100] * len(prefix['input_ids']) + encoded['input_ids'][len(prefix['input_ids']):])
        processor.tokenizer.padding_side = 'right' if supervised else 'left'
        batch = dict(processor.tokenizer.pad(features, padding=True, return_tensors='pt'))
        batch.update(pixel_values=torch.cat(pixels), image_grid_thw=torch.cat(grids))
        if supervised:
            n = batch['input_ids'].shape[1]
            batch['labels'] = torch.tensor([x + [-100] * (n - len(x)) for x in labels])
            processor.tokenizer.padding_side = 'left'
            prompts = processor.tokenizer.pad(prompt_features, padding=True, return_tensors='pt')
            batch.update(prompt_input_ids=prompts['input_ids'], prompt_attention_mask=prompts['attention_mask'],
                         target_pt=torch.tensor([r['pt'] for r in rows], dtype=torch.float32))
        return batch

    def __call__(self, rows):
        return self.encode(rows)


def training_inputs(batch):
    return {k: v for k, v in batch.items() if not k.startswith('prompt_') and k != 'target_pt'}


@torch.no_grad()
def generate(model, processor, batch, max_new_tokens=32):
    # Evaluation never supplies assistant targets or the full supervised input to generate().
    inputs = {k: v for k, v in batch.items() if k in ['pixel_values', 'image_grid_thw']}
    inputs['input_ids'] = batch.get('prompt_input_ids', batch['input_ids'])
    inputs['attention_mask'] = batch.get('prompt_attention_mask', batch['attention_mask'])
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    return processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
