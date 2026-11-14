import unittest
from .prepare_run import select


class SelectionTests(unittest.TestCase):
    def test_balanced_unique_reproducible_selection(self):
        groups={f'C{i:02d}':[f'episode-{j}' for j in range(40+i)] for i in range(1,16)}
        chosen=select(groups)
        self.assertEqual(sum(map(len,chosen.values())),35)
        self.assertEqual(sorted(map(len,chosen.values())),[2]*10+[3]*5)
        self.assertEqual(chosen,select({k:list(reversed(v)) for k,v in reversed(list(groups.items()))}))
        for task,rows in chosen.items():
            self.assertEqual(len(rows),len(set(rows)))
            self.assertTrue(set(rows)<=set(groups[task]))

    def test_impossible_task_coverage_is_rejected(self):
        with self.assertRaises(ValueError):select({'C01':['one']})
        with self.assertRaises(ValueError):select({f'C{i}':['a','b'] for i in range(18)})
        with self.assertRaises(ValueError):select({f'C{i}':['a','b','b'] for i in range(15)})


if __name__=='__main__':unittest.main()
