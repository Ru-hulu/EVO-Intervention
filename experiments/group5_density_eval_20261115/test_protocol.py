import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .common import choose,initialize_queue,claim,finish,db
from .worker import equivalent


class ProtocolTests(unittest.TestCase):
    def test_select_smallest_near_best_and_reject_unsafe(self):
        rows=[dict(replicas_per_gpu=n,observations_per_second=v,passed=ok)
              for n,v,ok in ((1,10,True),(2,15,True),(3,15.3,True),(4,20,False))]
        self.assertEqual(choose(rows)['replicas_per_gpu'],2)
        with self.assertRaises(AssertionError):choose([dict(passed=False)])

    def test_claims_are_unique_and_allow_only_owner_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'queue.sqlite';items=[(str(i),2,10+i) for i in range(24)]
            initialize_queue(p,items)
            with ThreadPoolExecutor(max_workers=8) as pool:
                claimed=list(pool.map(lambda i:claim(p,str(i)),range(24)))
            self.assertEqual(len({r['key'] for r in claimed}),24)
            self.assertIsNone(claim(p,'extra'))
            r=claimed[0]
            with self.assertRaises(AssertionError):finish(p,'wrong',r['key'],2)
            finish(p,'0',r['key'],2,'pending');self.assertEqual(claim(p,'again')['key'],r['key'])
            with self.assertRaises(AssertionError):initialize_queue(p,items)

    def test_equivalence_never_ignores_invalid_mask_or_memory(self):
        def v(p=.2,a=1):return dict(prediction=p,input_audit=dict(history=a))
        self.assertLess(equivalent([v()],[v(.201)],.002),.002)
        for x in (v(None),v(.21),v(.2,2)):
            with self.assertRaises(AssertionError):equivalent([v()],[x],.002)


if __name__=='__main__':unittest.main()
