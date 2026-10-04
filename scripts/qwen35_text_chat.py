#!/usr/bin/env python3
"""Use local Qwen3.5 weights for streaming, multi-turn text chat."""

import argparse
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TextStreamer


DEFAULT_MODEL = Path('/root/autodl-tmp/qwen35-ft/models/Qwen3.5-4B')


def read_prompt():
    text = input('\n你 > ').strip()
    if text != '/paste':
        return text
    print('粘贴多行文本，单独输入 /end 提交：')
    lines = []
    while True:
        line = input()
        if line == '/end':
            return '\n'.join(lines)
        lines.append(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--system', default='你是一个乐于助人的助手。')
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--max-input-tokens', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--thinking', action='store_true', help='开启模型思考输出；默认直接回答')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prompt', help='运行一次 Prompt 后退出')
    group.add_argument('--prompt-file', type=Path, help='从 UTF-8 文件读取一次 Prompt 后退出')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_input_tokens < 1 or not 0 <= args.temperature <= 2:
        parser.error('token 上限必须为正数；temperature 必须在 0–2 之间')
    if not torch.cuda.is_available():
        parser.error('没有可用的 CUDA GPU')
    torch.set_num_threads(4)
    print(f'正在加载 {args.model_dir} …', flush=True)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
    ).to('cuda').eval()
    print(f'模型已就绪，加载用时 {time.monotonic() - started:.1f} 秒。', flush=True)
    initial = [{'role': 'system', 'content': args.system}] if args.system else []
    messages = list(initial)

    def reply(prompt):
        candidate = messages + [{'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(
            candidate, tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking,
            return_tensors='pt', return_dict=True,
        ).to('cuda')
        input_length = inputs['input_ids'].shape[1]
        if input_length > args.max_input_tokens:
            raise ValueError(f'当前输入含 {input_length} 个 token，超过 {args.max_input_tokens}；请 /clear 或缩短文本')
        settings = {'max_new_tokens': args.max_new_tokens, 'do_sample': args.temperature > 0,
                    'use_cache': True, 'eos_token_id': tokenizer.eos_token_id,
                    'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id}
        if args.temperature > 0:
            settings.update(temperature=args.temperature, top_p=0.8, top_k=20)
        print('\nQwen > ', end='', flush=True)
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs, **settings,
                streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
            )
        tokens = output[0, input_length:]
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        elapsed = time.monotonic() - start
        print(f'[{len(tokens)} tokens · {elapsed:.2f} 秒 · {len(tokens) / elapsed:.1f} tokens/s]', flush=True)
        if len(tokens) == args.max_new_tokens and tokens[-1].item() != tokenizer.eos_token_id:
            print('[已达到输出长度上限，可提高 --max-new-tokens]', flush=True)
        messages.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}])
        return answer

    if args.prompt is not None or args.prompt_file is not None:
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding='utf-8')
        if not prompt.strip():
            parser.error('Prompt 不能为空')
        reply(prompt)
        return

    print('输入 Prompt 开始对话。/clear 清空上下文，/paste 粘贴多行，/exit 退出。', flush=True)
    print(f'思考输出：{"开启" if args.thinking else "关闭"}；最大输出：{args.max_new_tokens} tokens。', flush=True)
    while True:
        try:
            prompt = read_prompt()
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            break
        if prompt == '/exit':
            break
        if prompt == '/clear':
            messages = list(initial)
            print('已清空对话上下文。')
            continue
        if not prompt:
            continue
        try:
            reply(prompt)
        except KeyboardInterrupt:
            print('\n本次生成已中断，未加入对话历史。')
        except (ValueError, torch.cuda.OutOfMemoryError) as error:
            print(f'本次输入未完成：{error}')


if __name__ == '__main__':
    main()
