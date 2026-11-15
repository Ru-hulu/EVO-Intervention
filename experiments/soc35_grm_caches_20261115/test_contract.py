import copy
import unittest
from .common import check_selection, slots, SUITES
from experiments.recover_finetune_20261114.data import grm_answer


class Contract(unittest.TestCase):
    def fixture(self):
        episodes=[];manifests={}
        for suite in SUITES:
            tasks=[]
            for i in range(15):
                task=f'T{i:02d}';train=[f'e{j}' for j in range(3 if i<5 else 2)]
                tasks.append(dict(task_id=task,train=train,val=['validation']))
                episodes.extend(dict(episode_id=f'{suite}/{task}/{e}',suite=suite,task_id=task,sample_id=e,samples=10) for e in train)
            manifests[suite]=dict(tasks=tasks)
        return episodes,manifests

    def test_full_coverage(self):
        episodes,m=self.fixture();result=check_selection(episodes,m)
        self.assertEqual([v['episodes'] for v in result.values()],[35]*3)

    def test_reject_validation_leak(self):
        e,m=self.fixture();e[0]['sample_id']='validation'
        with self.assertRaises(AssertionError):check_selection(e,m)

    def test_reject_uncovered_task(self):
        e,m=self.fixture();m['State']['tasks'].append(dict(task_id='missing',train=['a','b'],val=[]))
        with self.assertRaises(AssertionError):check_selection(e,m)

    def test_reject_duplicate_episode(self):
        e,m=self.fixture();e[0]=copy.deepcopy(e[1])
        with self.assertRaises(AssertionError):check_selection(e,m)

    def test_grm_order_and_endpoint(self):
        self.assertEqual(slots(3,20),[(0,0),(19,0),(0,0),(0,1),(0,2),(3,0),(3,1),(3,2)])
        self.assertEqual(len(slots(0,1)),8)
        with self.assertRaises(ValueError):slots(20,20)

    def test_integer_score_retains_continuous_pt(self):
        row=dict(before_pt=0,after_pt=.125,pt=.125)
        self.assertEqual(grm_answer(row),'<score>+13%</score>');self.assertEqual(row['pt'],.125)
        with self.assertRaises(ValueError):grm_answer(dict(before_pt=.1,after_pt=.5,pt=.5))


if __name__=='__main__':unittest.main()
