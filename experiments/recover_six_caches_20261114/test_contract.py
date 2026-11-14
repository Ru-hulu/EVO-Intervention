import unittest
from .common import select,slots,MODELS


class Contract(unittest.TestCase):
    def test_disjoint_seeded_selection(self):
        tasks=[dict(task_id=f'R{i}',train=list(range(12)),val=[20,21]) for i in range(5)]
        self.assertEqual(select(tasks),select(tasks[::-1]))
        self.assertTrue(all(len(x)==2 and len(set(x))==2 for x in select(tasks).values()))
        tasks[0]['val']=[0]
        with self.assertRaises(ValueError):select(tasks)

    def test_native_slots(self):
        for i in range(30):
            for model in MODELS:
                s=slots(model,i,30)
                self.assertTrue(all(0<=j<30 and v in [0,1,2] for j,v in s))
                if model not in ['robobrain25-4b','dopamine-3b']:
                    self.assertTrue(all(j<=i for j,v in s))
            self.assertEqual(slots('procvlm-2b',i,30),[(max(0,j),0) for j in range(i-7,i+1)])
            self.assertEqual(slots('robometer-4b',i,30)[-3:],[(i,v) for v in range(3)])
            self.assertEqual(slots('vlac-2b',i,30),[(max(0,i-1),0),(i,0)])
        self.assertEqual(slots('robometer-4b',2,30)[:7],[(0,0),(1,0)]+[(2,0)]*5)
        self.assertEqual(slots('robobrain25-4b',0,30),[(0,0),(29,0),(0,0),(0,1),(0,2),(0,0),(0,1),(0,2)])


if __name__=='__main__':unittest.main()
