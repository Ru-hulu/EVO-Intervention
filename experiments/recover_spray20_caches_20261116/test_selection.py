import unittest
from .run import choose


class Selection(unittest.TestCase):
    def test_deterministic_train_only(self):
        task = dict(task_id='R03',train=[f'ep-{i:02d}' for i in range(25)],val=['heldout'])
        selected = choose(task)
        self.assertEqual(len(selected),20)
        self.assertEqual(selected,choose(dict(task,train=task['train'][::-1])))
        self.assertTrue(set(selected) <= set(task['train']))
        self.assertNotIn('heldout',selected)

    def test_reject_invalid_or_insufficient_splits(self):
        task = dict(task_id='R03',train=list(range(25)),val=[30])
        for changed in [dict(task,train=list(range(19))),dict(task,val=[0]),
                        dict(task,train=list(range(24))+[0]),dict(task,task_id='R05')]:
            with self.assertRaises(ValueError): choose(changed)


if __name__ == '__main__': unittest.main()
