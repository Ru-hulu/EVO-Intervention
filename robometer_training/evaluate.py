"""Evaluate current-frame progress with the same windows and logits as training."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import torch

from .data import ObservationWindow, ProgressCollator, ProgressDataset
from .model import current_logits, load_adapter, load_model, load_processor, progress_loss, progress_values


@torch.no_grad()
def predict(model, collator, rows):
    model.eval()
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in collator.encode(rows).items()}
    precision = torch.autocast(device.type, dtype=torch.bfloat16) if model.model.dtype == torch.bfloat16 else nullcontext()
    with precision:
        logits = current_logits(model, batch)
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite progress logits')
    return progress_values(logits), logits


class OnlinePredictor:
    """Call once per 5 FPS observation; changing episode_id resets history."""
    def __init__(self, model, collator, window_size=8):
        self.model, self.collator = model, collator
        self.window = ObservationWindow(window_size)

    def push(self, episode_id, task, image_path):
        images = self.window.push(episode_id, image_path)
        row = {'task': task, 'image': images}
        return float(predict(self.model, self.collator, [row])[0][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    processor = load_processor(config['model_path'], min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
    model, _ = load_model(config['model_path'], processor)
    if args.adapter:
        load_adapter(model, args.adapter)
    model.to(args.device).eval()
    collator = ProgressCollator(processor, config['max_length'])
    rows = ProgressDataset(config['validation_data']).rows
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    errors, loss_sum = [], 0.0
    started = time.monotonic()
    with (args.output_dir / 'predictions.jsonl').open('w') as stream:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            values, logits = predict(model, collator, batch)
            target = torch.tensor([r['pt'] for r in batch], device=logits.device)
            loss_sum += progress_loss(logits, target).item() * len(batch)
            for row, value in zip(batch, values.tolist(), strict=True):
                errors.append(100 * (value - row['pt']))
                record = {k: row[k] for k in ('id', 'episode_id', 'frame_index', 'task', 'history_frame_indices')}
                record.update(target_pt=row['pt'], predicted_pt=value)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
    report = {'samples': len(rows), 'loss': loss_sum / len(rows),
              'mae_pp': sum(abs(e) for e in errors) / len(rows),
              'rmse_pp': math.sqrt(sum(e*e for e in errors) / len(rows)),
              'conditioning': 'causal eight-frame main-view window, 5 FPS',
              'adapter': str(args.adapter) if args.adapter else None,
              'elapsed_seconds': time.monotonic() - started}
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
