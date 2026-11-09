"""Explicitly invoked native-answer SFT; no progress head or automatic training launch."""
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from grm import Dataset, Collator, load_model, load_processor, training_inputs, generate, parse_score
from metrics import summarize
from periodic import SaveBeforeValidation, ValidationProgress


class FinishEpoch(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        self.validated_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps and getattr(self, 'validated_step', -1) != state.global_step:
            control.should_save = control.should_evaluate = True
        return control


class ForwardTrainer(SaveBeforeValidation, Trainer):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = config
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**training_inputs(inputs), use_cache=False)
        if not torch.isfinite(outputs.loss):
            raise RuntimeError('Non-finite SFT loss')
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        native = self.accelerator.unwrap_model(model)
        answers = generate(native, self.processing_class, inputs, self.run_config['max_new_tokens'])
        # Invalid generations are explicit evaluation failures, never silently converted to zero.
        values = [parse_score(text) for text in answers]
        return None, torch.tensor(values, device=inputs['target_pt'].device), inputs['target_pt']

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        Path(output_dir or self.args.output_dir, 'run_config.json').write_text(json.dumps(self.run_config, indent=2))


def compute_metrics(prediction, metadata):
    records = [dict(episode_id=row['episode_id'], predicted_pt=float(p), target_pt=float(t))
               for row, p, t in zip(metadata, prediction.predictions, prediction.label_ids, strict=True)]
    result = summarize(records)
    metrics = {'mae_pp':result['overall']['mae_pp'], 'samples':len(records)}
    if result['overall']['pearson_episode_mean'] is not None:
        metrics['pearson_episode_mean'] = result['overall']['pearson_episode_mean']
    metrics['undefined_pearson_episodes'] = len(result['overall']['undefined_pearson_episodes'])
    for suite, values in result['by_suite'].items():
        metrics[suite+'_mae_pp'] = values['mae_pp']
        if values['pearson_episode_mean'] is not None:
            metrics[suite+'_pearson_episode_mean'] = values['pearson_episode_mean']
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output_dir: config['output_dir'] = str(args.output_dir)
    output = Path(config['output_dir'])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Refuse to overwrite a nonempty run directory')
    if args.resume:
        previous = json.loads((args.resume/'run_config.json').read_text())
        for key in ['model_path','data_dir','lora','min_pixels','max_pixels','max_length','training','seed','input_contract']:
            assert config[key] == previous[key], f'Resume configuration mismatch: {key}'
    set_seed(config['seed'])
    processor = load_processor(config['model_path'], config['min_pixels'], config['max_pixels'])
    model = load_model(config['model_path'], lora=config['lora'])
    train, val = Dataset(config['data_dir'], 'train'), Dataset(config['data_dir'], 'validation')
    options = dict(config['training'])
    world = int(os.environ.get('WORLD_SIZE', '1'))
    updates = math.ceil(math.ceil(len(train)/(options['per_device_train_batch_size']*world))/options['gradient_accumulation_steps'])
    half_epoch = math.ceil(updates/2)
    arguments = TrainingArguments(output_dir=str(output), bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='steps', eval_steps=half_epoch, save_strategy='steps', save_steps=half_epoch,
        save_total_limit=2, metric_for_best_model='mae_pp', greater_is_better=False,
        load_best_model_at_end=False, remove_unused_columns=False, label_names=['target_pt'],
        report_to='none', seed=config['seed'], ddp_find_unused_parameters=False,
        logging_nan_inf_filter=False, dataloader_pin_memory=True, **options)
    trainer = ForwardTrainer(model=model, args=arguments, config=config, processing_class=processor,
        train_dataset=train, eval_dataset=val, data_collator=Collator(processor, config['max_length']),
        compute_metrics=partial(compute_metrics, metadata=val.rows), callbacks=[FinishEpoch(), ValidationProgress()])
    output.mkdir(parents=True, exist_ok=True)
    if trainer.is_world_process_zero():
        (output/'run_config.json').write_text(json.dumps(config, indent=2))
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_state()


if __name__ == '__main__':
    main()
