"""One worst-window GPU forward/backward; no optimizer update or checkpoint."""
import argparse
import json
from pathlib import Path
import time

import torch

from .data import ProgressCollator, ProgressDataset
from .model import attach_lora, current_logits, load_model, load_processor, progress_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = {'status': 'checking', 'optimizer_steps': 0}
    try:
        processor = load_processor(config['model_path'], min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
        model, _ = load_model(config['model_path'], processor)
        attach_lora(model, **config['lora'])
        model.to('cuda').train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        rows = ProgressDataset(config['train_data']).rows
        row = max((r for r in rows if len(set(r['image'])) == 8), key=lambda r: len(r['task']))
        batch_size = config['training']['per_device_train_batch_size']
        batch = {k: v.cuda() for k, v in ProgressCollator(processor, config['max_length'])([row] * batch_size).items()}
        target = batch.pop('labels')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = progress_loss(current_logits(model, batch), target)
        loss.backward()
        torch.cuda.synchronize()
        gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients.values()):
            raise RuntimeError('Non-finite preflight loss/gradients')
        if not all('lora_' in n or n.startswith('progress_head.') for n in gradients):
            raise RuntimeError('Unexpected trainable base parameters')
        if not any('lora_B' in n and g.abs().max() > 0 for n, g in gradients.items()):
            raise RuntimeError('LoRA gradient is zero')
        optimizer_bytes = sum(p.numel() * 8 for p in model.parameters() if p.requires_grad)
        peak = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        if free < optimizer_bytes + 2 * 1024**3:
            raise torch.OutOfMemoryError('Insufficient margin for Adam states plus 2 GiB')
        report.update(status='passed', batch_size=batch_size, loss=loss.item(),
                      sequence_length=batch['input_ids'].shape[1], sample_id=row['id'],
                      gradient_tensors=len(gradients), gradients_finite=True,
                      allocated_peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      reserved_peak_gib=peak / 1024**3, optimizer_state_estimate_gib=optimizer_bytes / 1024**3,
                      total_gpu_gib=total / 1024**3, elapsed_seconds=time.monotonic() - started)
    except torch.OutOfMemoryError as error:
        report.update(status='oom', error=str(error))
        raise
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
