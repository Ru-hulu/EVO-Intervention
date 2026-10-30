"""Explicit LoRA training entry. Importing/preparing/checking never starts training."""
import argparse
import json
import math
from pathlib import Path
import shutil

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from .data import ProgressCollator, ProgressDataset
from .model import base_model, load_model, restore_adapter_in_place, save_adapter


class RunMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        values = dict(logs or {})
        if any(not math.isfinite(values[key]) for key in ('loss', 'eval_loss', 'grad_norm') if key in values):
            raise RuntimeError('Non-finite loss or gradient')
        if model is not None:
            base = base_model(model)
            values.update(last_batch_lm_loss=getattr(base, '_last_lm_loss', None),
                          last_batch_value_loss=getattr(base, '_last_value_loss', None))
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': state.global_step, 'epoch': state.epoch, **values,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0}) + '\n')
        if shutil.disk_usage(output).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB free')

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step >= state.max_steps
        return control


class ProgressTrainer(Trainer):
    def __init__(self, *args, run_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = run_config
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output = model(**inputs)
        return (output.loss, {'loss': output.loss}) if return_outputs else output.loss

    def _save(self, output_dir=None, state_dict=None):
        directory = Path(output_dir or self.args.output_dir)
        save_adapter(self.model, self.processing_class, directory, self.run_config)
        # Trainer saves optimizer/scheduler/RNG/cursor beside these inference weights.
        torch.save(self.args, directory / 'training_args.bin')

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        restore_adapter_in_place(model or self.model, resume_from_checkpoint)


def make_trainer(config, model, processor, *, output_dir=None, use_cpu=False):
    args = TrainingArguments(output_dir=output_dir or config['output_dir'], use_cpu=use_cpu,
        bf16=not use_cpu, gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
        prediction_loss_only=True, logging_nan_inf_filter=False, remove_unused_columns=False,
        label_names=['labels'], report_to='none', seed=config['seed'],
        dataloader_num_workers=0 if use_cpu else config['dataloader_num_workers'],
        dataloader_persistent_workers=not use_cpu and config['dataloader_num_workers'] > 0,
        dataloader_prefetch_factor=2 if not use_cpu and config['dataloader_num_workers'] > 0 else None,
        dataloader_pin_memory=not use_cpu, **config['training'])
    return ProgressTrainer(model=model, args=args, run_config=config,
        train_dataset=ProgressDataset(config['train_data'], supervised=True),
        eval_dataset=ProgressDataset(config['validation_data'], supervised=True),
        data_collator=ProgressCollator(processor, config['max_length']), processing_class=processor,
        callbacks=[RunMonitor()])


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
        raise ValueError('Output directory is nonempty')
    if args.resume_from_checkpoint:
        saved = json.loads((args.resume_from_checkpoint / 'run_config.json').read_text())
        if {k:v for k,v in saved.items() if k != 'output_dir'} != {k:v for k,v in config.items() if k != 'output_dir'}:
            raise ValueError('Resume configuration differs')
    if not torch.cuda.is_available():
        raise RuntimeError('Formal training requires CUDA')
    set_seed(config['seed'])
    torch.set_num_threads(8)
    model, processor = load_model(config, trainable=True)
    trainer = make_trainer(config, model, processor)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_state()


if __name__ == '__main__':
    main()
