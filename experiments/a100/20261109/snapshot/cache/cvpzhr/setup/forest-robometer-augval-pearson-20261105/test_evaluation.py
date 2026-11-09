import unittest
from common import *
from controller import training_ready
from aggregate import validate
class Tests(unittest.TestCase):
    def test_pearson_reference_cases(self):
        self.assertAlmostEqual(pearson([0,1,2,3],[0,1,2,3])[0],1)
        self.assertAlmostEqual(pearson([3,2,1,0],[0,1,2,3])[0],-1)
        self.assertAlmostEqual(pearson([1,-1,-1,1],[-1,-1,1,1])[0],0)
        self.assertAlmostEqual(pearson([.25,.375,.5,.625,.75],[0,.25,.5,.75,1])[0],1)
    def test_constants_are_explicitly_undefined(self):
        self.assertEqual(pearson([.5,.5,.5],[0,.5,1]),(None,'constant_prediction'))
        self.assertEqual(pearson([0,.5,1],[.5,.5,.5]),(None,'constant_target'))
    def test_episode_average_not_pooled_or_length_weighted(self):
        r=[]
        for eid,x,y in [('A',[0,1],[0,1]),('B',[1,.5,0],[0,.5,1])]:
            r += [dict(episode_id=eid,suite='Count',task='Count/test',kind='original',source_sample_index=i,
                       anchor_sample_index=i,predicted_pt=a,target_pt=b) for i,(a,b) in enumerate(zip(x,y))]
        metrics=summarize(r)['overall']
        self.assertAlmostEqual(metrics['pearson_episode_mean'],0)
        self.assertEqual(metrics['valid_pearson_episodes'],2)
        self.assertAlmostEqual(metrics['mae_pp'],40)
    def test_training_gate_waits_for_final_validation(self):
        self.assertFalse(training_ready(dict(phase='validating',step=9967,epoch=1)))
        self.assertFalse(training_ready(dict(phase='training',step=4984,epoch=.5)))
        self.assertTrue(training_ready(dict(phase='completed',step=9967,epoch=1)))
        with self.assertRaises(RuntimeError):training_ready(dict(phase='failed',step=9967,epoch=1))
    def test_shared_timeline_windows_and_labels(self):
        p=plan();n=0
        for e in p['episodes']:
            meta=rows(e['timeline']);rb=rows(e['robometer_rows']);source=rows(Path(e['source_dir'])/'samples.jsonl')
            self.assertEqual(len(meta),len(rb))
            for i,(a,b) in enumerate(zip(meta,rb)):
                self.assertEqual(a['pt'],b['pt'])
                self.assertEqual(a['pt'],source[a['anchor_sample_index']]['pt'])
                self.assertEqual(a['images'],source[a['source_sample_index']]['images'])
                obs=[max(0,j) for j in range(i-7,i)]+[i,i,i]
                self.assertEqual(b['observation_indices'],obs)
                self.assertEqual(b['window_indices'],[meta[j]['source_sample_index'] for j in obs])
                self.assertEqual(b['input_views'],['cam_high']*8+['cam_left_wrist','cam_right_wrist'])
                n+=1
        self.assertEqual(n,78554)
    def test_pair_alignment_rejects_duplicate_prediction(self):
        p=plan()
        from worker import record
        rr=[record(r,r['pt']) for e in p['episodes'] for r in rows(e['timeline'])]
        checked=validate(rr,p)
        self.assertEqual(len(checked),78554)
        checked[-1]=dict(checked[0])
        with self.assertRaises(AssertionError):validate(checked,p)
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT/'cpu-tests.json',dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
          failures=len(result.failures),errors=len(result.errors),completed_at=now()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
