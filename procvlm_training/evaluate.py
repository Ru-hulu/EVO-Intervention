"""Teacher-forced loss on eligible rows; generated progress metrics on ALL validation rows."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from .data import ProgressCollator, ProgressDataset
from .model import load_model, predict_batch, to_device


def progress_metrics(rows, key):
    errors = [(row[key] - row['target_pt']) * 100 for row in rows if row[key] is not None and math.isfinite(row[key])]
    return {'valid_predictions': len(errors), 'invalid_predictions': len(rows) - len(errors),
            'mae_pp': sum(abs(x) for x in errors) / len(errors) if errors else None,
            'rmse_pp': math.sqrt(sum(x*x for x in errors) / len(errors)) if errors else None}


@torch.inference_mode()
def evaluate(config, checkpoint, output, batch_size=1):
    output.mkdir(parents=True, exist_ok=False)
    model, processor = load_model(config, adapter=checkpoint, device='cuda:0')
    model.eval()
    collator = ProgressCollator(processor, config['max_length'])
    rows = ProgressDataset(config['validation_data']).rows
    loss_rows = [row for row in rows if row['supervision_eligible']]
    weighted_loss = 0.0
    loss_batch_size = config['training']['per_device_eval_batch_size']
    for start in range(0, len(loss_rows), loss_batch_size):
        group = loss_rows[start:start + loss_batch_size]
        result = model(**to_device(collator(group), model))
        weighted_loss += float(result.loss) * len(group)
        del result
    predictions = []
    with (output / 'predictions.jsonl').open('x') as stream:
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            torch.cuda.synchronize()
            started = time.perf_counter()
            results = predict_batch(model, collator, group, **config['inference'])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for row, result in zip(group, results):
                record = {'episode_id': row['episode_id'], 'frame_index': row['frame_index'],
                    'target_pt': row['pt'], 'window_frame_indices': row['window_frame_indices'],
                    'included_in_teacher_forced_loss': row['supervision_eligible'], **result,
                    'batch_size': len(group), 'batch_elapsed_seconds': elapsed}
                predictions.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
    metrics = {'samples': len(rows), 'teacher_forced_loss_samples': len(loss_rows),
        'loss': weighted_loss / len(loss_rows), 'loss_kind': 'teacher-forced language CE plus weighted progress L1',
        **progress_metrics(predictions, 'predicted_pt'), 'text_progress': progress_metrics(predictions, 'text_predicted_pt'),
        'value_head_enabled': config['inference']['enable_value_head'],
        'conditioning': 'causal main-view current + seven previous 5-FPS samples; reference answers are never used for prediction',
        'checkpoint': str(checkpoint), 'training_started': False}
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    config = json.loads((args.config or args.checkpoint / 'run_config.json').read_text())
    print(json.dumps(evaluate(config, args.checkpoint, args.output_dir, args.batch_size), ensure_ascii=False, indent=2))
