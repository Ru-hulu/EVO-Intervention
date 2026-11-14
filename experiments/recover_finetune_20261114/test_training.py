import unittest
from .data import grm_answer
from .common import ORDER


class TrainingContract(unittest.TestCase):
    def test_native_grm_relative_score_from_initial(self):
        for pt,answer in [(0,'<score>0%</score>'),(.125,'<score>+13%</score>'),(1,'<score>+100%</score>')]:
            self.assertEqual(grm_answer(dict(before_pt=0,after_pt=pt,pt=pt)),answer)

    def test_reject_different_reference_semantics(self):
        for row in [dict(before_pt=.1,after_pt=.5,pt=.5),dict(before_pt=0,after_pt=.7,pt=.5),
                    dict(before_pt=0,after_pt=float('nan'),pt=float('nan'))]:
            with self.assertRaises(ValueError):grm_answer(row)

    def test_two_workspace_ownership(self):
        self.assertEqual(ORDER,['robometer-4b','robobrain25-4b','dopamine-3b'])

    def test_trainer_logs_can_contain_epoch(self):
        try:
            from .train import Monitor
        except ImportError:
            self.skipTest('Training dependencies are tested in the remote model environment')
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from types import SimpleNamespace
        from .common import read
        with TemporaryDirectory() as directory:
            monitor=Monitor({},Path(directory))
            state=SimpleNamespace(global_step=1,max_steps=3,epoch=.25,is_world_process_zero=True)
            monitor.on_log(None,state,None,logs={'epoch':.25,'loss':1.,'grad_norm':.1})
            saved=read(Path(directory)/'progress.json')
            self.assertEqual(saved['step'],1)
            self.assertEqual(saved['epoch'],.25)
            with self.assertRaises(AssertionError):monitor.on_log(None,state,None,logs={'loss':float('nan')})


if __name__=='__main__':unittest.main()
