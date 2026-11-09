import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from robometer_training.periodic import SaveBeforeValidation
from robometer_training.train import RunMonitor

class FakeState:
    global_step=5
    epoch=.5
    def save_to_json(self,path):Path(path).write_text('{}')
class FakeBase:
    def _get_output_dir(self,trial=None):return self.args.output_dir
    def is_world_process_zero(self):return True
    def _save_checkpoint(self,model,trial):
        root=Path(self.args.output_dir)/'checkpoint-5'
        root.mkdir()
        for name in ['adapter_config.json','adapter_model.safetensors','progress_head.safetensors',
                     'optimizer.pt','scheduler.pt','trainer_state.json','run_config.json','rng_state.pth']:
            (root/name).write_text('{}')
    def _evaluate(self,trial,ignore_keys_for_eval,skip_scheduler=False):
        assert (Path(self.args.output_dir)/'checkpoint-5/complete.json').exists()
        raise RuntimeError('injected validation failure')
class FakeTrainer(SaveBeforeValidation,FakeBase):pass

class SavingChecks(unittest.TestCase):
    def test_validation_failure_preserves_recovery_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            t=FakeTrainer();t.args=SimpleNamespace(output_dir=temp,world_size=1)
            t.state=FakeState();t.model=None;t.control=SimpleNamespace(should_save=True)
            t.accelerator=SimpleNamespace(wait_for_everyone=lambda:None)
            t.callback_handler=SimpleNamespace(on_save=lambda *args:args[-1])
            with self.assertRaisesRegex(RuntimeError,'injected validation failure'):
                t._evaluate(None,None)
            self.assertTrue((Path(temp)/'last/complete.json').exists())
            events=[json.loads(s)['event'] for s in (Path(temp)/'lifecycle.jsonl').read_text().splitlines()]
            self.assertEqual(events,['checkpoint_save_started','checkpoint_saved','validation_started','validation_failed'])
    def test_final_step_gets_validation_when_not_multiple_of_interval(self):
        c=SimpleNamespace(should_save=False,should_evaluate=False)
        RunMonitor().on_epoch_end(None,SimpleNamespace(global_step=9967,max_steps=9967),c)
        self.assertTrue(c.should_save and c.should_evaluate)

    def test_final_step_is_not_validated_twice(self):
        monitor=RunMonitor()
        state=SimpleNamespace(global_step=40,max_steps=40)
        c=SimpleNamespace(should_save=False,should_evaluate=False)
        monitor.on_evaluate(None,state,c)
        monitor.on_epoch_end(None,state,c)
        self.assertFalse(c.should_save or c.should_evaluate)

if __name__=='__main__':unittest.main(verbosity=2)
