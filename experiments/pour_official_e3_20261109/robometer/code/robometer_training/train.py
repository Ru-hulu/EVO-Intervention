"""Single- or multi-GPU progress LoRA training; runs only when explicitly invoked."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from functools import partial

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from .data import ProgressCollator, ProgressDataset
from .periodic import SaveBeforeValidation, ValidationProgress
from .model import (attach_lora, current_logits, load_adapter, load_model, load_processor,
                    progress_loss, progress_values, save_adapter)


def compute_metrics(prediction):
    progress = progress_values(torch.from_numpy(prediction.predictions)).numpy()
    errors = 100 * (progress - prediction.label_ids)
    return {'mae_pp': float(abs(errors).mean()), 'rmse_pp': float((errors ** 2).mean() ** 0.5)}


def compute_suite_metrics(prediction, suites):
    metrics=compute_metrics(prediction)
    predicted=progress_values(torch.from_numpy(prediction.predictions)).numpy()
    labels=prediction.label_ids
    assert len(predicted)==len(labels)==len(suites), 'Validation sample count mismatch'
    errors=100*(predicted-labels)
    metrics['samples']=len(suites)
    suite_array=np.asarray(suites)
    for suite in ('Count','Order','State'):
        selected=errors[suite_array==suite]
        if len(selected):
            metrics[suite+'_mae_pp']=float(np.abs(selected).mean())
            metrics[suite+'_samples']=len(selected)
    return metrics


class RunMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if any(not math.isfinite(logs[k]) for k in ('loss', 'grad_norm', 'eval_loss') if k in logs):
            raise RuntimeError('Non-finite loss or gradient')
        if not state.is_world_process_zero:
            return
        stats = {'step': state.global_step, 'total_steps': state.max_steps,
                 'timestamp': time.time(), 'epoch': state.epoch, **logs,
                 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0}
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(stats) + '\n')
        if shutil.disk_usage(output).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB free; stopping this run')

    def on_evaluate(self, args, state, control, **kwargs):
        self.last_validation_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.global_step >= state.max_steps
                and getattr(self, 'last_validation_step', None) != state.global_step):
            control.should_save = True
            control.should_evaluate = True
        return control


class ProgressTrainer(SaveBeforeValidation, Trainer):
    def __init__(self, *args, run_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = run_config
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        logits = current_logits(model, {k: v for k, v in inputs.items() if k != 'labels'})
        loss = progress_loss(logits, labels)
        return (loss, {'loss': loss, 'logits': logits}) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output = Path(output_dir or self.args.output_dir)
        save_adapter(self.model, output)
        # Trainer separately saves optimizer/scheduler/RNG and the sampling cursor.
        torch.save(self.args, output / 'training_args.bin')
        if self.run_config is not None:
            (output / 'run_config.json').write_text(json.dumps(self.run_config, indent=2))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_adapter(model or self.model, resume_from_checkpoint, trainable=True)


def make_trainer(config, model, processor, *, output_dir=None, use_cpu=False):
    options = dict(gradient_checkpointing=True, dataloader_num_workers=0)
    options.update(config['training'])
    workers = options['dataloader_num_workers']
    options.setdefault('dataloader_persistent_workers', bool(workers))
    options.setdefault('ddp_find_unused_parameters', False)
    arguments = TrainingArguments(
        output_dir=output_dir or config['output_dir'], use_cpu=use_cpu,
        bf16=not use_cpu or model.model.dtype == torch.bfloat16,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy=config.get('eval_strategy', 'steps'), eval_steps=config['eval_steps'],
        save_strategy='steps', save_steps=config.get('save_steps',config['eval_steps']),
        save_total_limit=config.get('save_total_limit',2),
        metric_for_best_model='mae_pp', greater_is_better=False, load_best_model_at_end=False,
        logging_nan_inf_filter=False, prediction_loss_only=False,
        remove_unused_columns=False, label_names=['labels'], report_to='none',
        dataloader_pin_memory=not use_cpu, seed=config['seed'],
        **options)
    validation=ProgressDataset(config['validation_data'])
    suites=[row.get('suite',row['episode_id'].split('/')[0]) for row in validation.rows]
    return ProgressTrainer(
        model=model, args=arguments, run_config=config,
        train_dataset=ProgressDataset(config['train_data']),
        eval_dataset=validation,
        data_collator=ProgressCollator(processor, config['max_length']),
        processing_class=processor, compute_metrics=partial(compute_suite_metrics,suites=suites),
        callbacks=[RunMonitor(),ValidationProgress()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output_dir:
        config['output_dir'] = str(args.output_dir)
    output = Path(config['output_dir'])
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError('Output directory is nonempty; choose a new directory or explicitly resume')
    if args.resume_from_checkpoint:
        saved = json.loads((args.resume_from_checkpoint / 'run_config.json').read_text())
        for key in ('model_path', 'lora', 'sample_fps', 'window_size', 'min_pixels', 'max_pixels',
                    'train_data', 'validation_data', 'seed', 'training'):
            if config[key] != saved[key]:
                raise ValueError(f'Resume configuration differs: {key}')
    if not torch.cuda.is_available():
        raise RuntimeError('The training entry requires a CUDA GPU')
    if config['training'].get('dataloader_num_workers', 0):
        torch.multiprocessing.set_start_method('spawn', force=True)
    set_seed(config['seed'])
    processor = load_processor(config['model_path'], min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
    model, _ = load_model(config['model_path'], processor)
    attach_lora(model, **config['lora'])
    trainer = make_trainer(config, model, processor)
    output.mkdir(parents=True, exist_ok=True)
    if trainer.is_world_process_zero():
        (output / 'run_config.json').write_text(json.dumps(config, indent=2))
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_state()


if __name__ == '__main__':
    main()
