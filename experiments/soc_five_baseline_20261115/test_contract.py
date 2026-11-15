import unittest
from unittest.mock import patch
from .common import partition,slots,validate,digest,MODELS
from .controller import forest_released
from .report import summarize


class Contract(unittest.TestCase):
    def test_replica_partitions_are_disjoint_complete_and_balanced(self):
        rows=[dict(key=str(i),frame_indices=list(range(20+i*7))) for i in range(22)]
        for n in (4,8):
            p=partition(rows,n);flat=[k for x in p for k in x]
            self.assertEqual(len(flat),len(set(flat)));self.assertEqual(set(flat),{r['key'] for r in rows})
            loads=[sum(len(rows[int(k)]['frame_indices']) for k in x) for x in p]
            self.assertLessEqual(max(loads)-min(loads),max(len(r['frame_indices']) for r in rows))

    def test_native_image_windows(self):
        self.assertEqual(slots('robobrain25-4b',7,20),[(0,0),(19,0),(0,0),(0,1),(0,2),(7,0),(7,1),(7,2)])
        self.assertEqual(slots('dopamine-3b',7,20),slots('robobrain25-4b',7,20))
        self.assertEqual(slots('procvlm-2b',2,20),[(0,0)]*6+[(1,0),(2,0)])
        self.assertEqual(len(slots('robometer-4b',19,20)),10)
        self.assertTrue(all(i<=7 for i,v in slots('robometer-4b',7,20)))

    def test_forest_pair_requires_completed_and_all_processes_exited(self):
        with patch('experiments.soc_five_baseline_20261115.controller.alive',side_effect=lambda p:bool(p and p.get('live'))):
            self.assertFalse(forest_released(dict(status='running',controller=None),[]))
            self.assertFalse(forest_released(dict(status='completed',controller=dict(live=True)),[]))
            self.assertFalse(forest_released(dict(status='completed',controller=None),[dict(process=dict(live=True))]))
            self.assertTrue(forest_released(dict(status='completed',controller=None),[]))

    def test_incomplete_suites_do_not_publish_mean(self):
        r=dict(samples=10,invalid_predictions=0,pearson=.3,spearman=.2,VS=.1,ER=.5,C=.2)
        x=summarize([r],2);self.assertIsNone(x['metrics']['pearson']['mean'])
        bad=dict(r,pearson=None,spearman=None,invalid_predictions=1)
        x=summarize([r,bad],2)
        self.assertEqual(x['metrics']['pearson'],dict(mean=.3,defined=1,undefined=1))
        self.assertEqual(x['invalid_frames'],1)

    def test_negative_grm_raw_and_checkpoint_binding(self):
        row=dict(frame_indices=[0],timestamps=[0],target_pt=[0],video_hashes=['v'],progress_sha256='pt',suite='State')
        expected=dict(model='dopamine-3b',suite='State',checkpoint_sha256='ck')
        audit=dict(expected,strict_final_state=True,epoch=1,future_endpoint_reference=True)
        r=dict(identity=expected,predictions=[-.25],raw_responses=['<score>-25%</score>'],
            frame_indices=[0],timestamps=[0],target_pt=[0],source_video_sha256=['v'],progress_sha256='pt',
            initialization=audit,initialization_sha256=digest(audit),input_audits=[dict(sample_index=0,frame_index=0,
                slots=slots('dopamine-3b',0,1))],status='completed')
        self.assertEqual(validate(r,row,expected,True),1)
        with self.assertRaises(AssertionError):validate(r,row,dict(expected,checkpoint_sha256='other'),True)


if __name__=='__main__':unittest.main()
