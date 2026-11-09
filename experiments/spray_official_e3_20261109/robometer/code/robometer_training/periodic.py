"""Save a complete resumable checkpoint before any scheduled validation."""
import json
import os
from pathlib import Path
import time
from transformers import TrainerCallback

def atomic_json(path, value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)

def atomic_link(directory, name, checkpoint):
    directory=Path(directory)
    temp=directory/(name+'.tmp')
    if temp.is_symlink(): temp.unlink()
    temp.symlink_to(Path(checkpoint).name, target_is_directory=True)
    temp.replace(directory/name)

class SaveBeforeValidation:
    def _event(self, name, **details):
        if not self.is_world_process_zero(): return
        output=Path(self.args.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        with (output/'lifecycle.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=name,time=time.time(),
                step=self.state.global_step,epoch=self.state.epoch,**details))+'\n')

    def _save_checkpoint(self, model, trial):
        self._event('checkpoint_save_started')
        started=time.perf_counter()
        super()._save_checkpoint(model,trial)
        # All ranks must finish their RNG-state files before validation starts.
        self.accelerator.wait_for_everyone()
        checkpoint=Path(self._get_output_dir(trial=trial))/f'checkpoint-{self.state.global_step}'
        if self.is_world_process_zero():
            required=['adapter_config.json','adapter_model.safetensors','progress_head.safetensors',
                      'optimizer.pt','scheduler.pt','trainer_state.json','run_config.json']
            required += ([f'rng_state_{r}.pth' for r in range(self.args.world_size)]
                         if self.args.world_size>1 else ['rng_state.pth'])
            missing=[name for name in required if not (checkpoint/name).is_file()]
            if missing: raise RuntimeError(f'Incomplete checkpoint: {missing}')
            receipt=dict(step=self.state.global_step,epoch=self.state.epoch,
                         checkpoint=str(checkpoint),complete=True,
                         files=required,saved_before_validation=True,time=time.time())
            atomic_json(checkpoint/'complete.json',receipt)
            atomic_json(Path(self.args.output_dir)/'last.json',receipt)
            atomic_link(self.args.output_dir,'last',checkpoint)
        self.accelerator.wait_for_everyone()
        self._event('checkpoint_saved',checkpoint=str(checkpoint),seconds=time.perf_counter()-started)

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        self._save_checkpoint(self.model,trial)
        self.control=self.callback_handler.on_save(self.args,self.state,self.control)
        self.control.should_save=False
        self._event('validation_started')
        started=time.perf_counter()
        try:
            metrics=super()._evaluate(trial,ignore_keys_for_eval,skip_scheduler=skip_scheduler)
        except BaseException as exc:
            self._event('validation_failed',error=repr(exc))
            raise
        self._event('validation_finished',seconds=time.perf_counter()-started,metrics=metrics)
        return metrics

    def _determine_best_metric(self, metrics, trial):
        improved=super()._determine_best_metric(metrics,trial)
        checkpoint=Path(self._get_output_dir(trial=trial))/f'checkpoint-{self.state.global_step}'
        if improved:
            self.state.best_model_checkpoint=str(checkpoint)
        if self.is_world_process_zero():
            assert (checkpoint/'complete.json').exists()
            atomic_json(checkpoint/'validation.json',metrics)
            state_tmp=checkpoint/'trainer_state.json.tmp'
            self.state.save_to_json(str(state_tmp))
            state_tmp.replace(checkpoint/'trainer_state.json')
            if improved:
                atomic_json(Path(self.args.output_dir)/'best.json',
                    dict(step=self.state.global_step,epoch=self.state.epoch,
                         checkpoint=str(checkpoint),mae_pp=metrics['eval_mae_pp']))
                atomic_link(self.args.output_dir,'best',checkpoint)
        return improved

class ValidationProgress(TrainerCallback):
    def __init__(self):
        self.step=None
        self.batches=0

    def on_prediction_step(self,args,state,control,**kwargs):
        if self.step != state.global_step:
            self.step=state.global_step
            self.batches=0
        self.batches+=1
        if self.batches==1 or self.batches%50==0:
            atomic_json(Path(args.output_dir)/f'validation-rank{args.process_index}.json',
                dict(phase='validating',step=state.global_step,batches=self.batches,time=time.time()))

    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        atomic_json(Path(args.output_dir)/f'validation-rank{args.process_index}.json',
            dict(phase='completed',step=state.global_step,batches=self.batches,time=time.time()))
