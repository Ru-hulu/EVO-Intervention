import unittest
from prepare_cache import check_source, make_records

class TimelineChecks(unittest.TestCase):
    def fixture(self):
        original = [dict(frame_index=i*5, pt=i/10, images={'cam_high': f'cam_high/{i}.png'})
                    for i in range(5)]
        # Original 0,1,2,3; pause at 3; reverse to 2,1; return through 2,3; continue 4.
        sequence = [(0,'original',0),(1,'original',1),(2,'original',2),(3,'original',3),
                    (3,'stationary',3),(2,'reverse',3),(1,'reverse',3),
                    (2,'forward',3),(3,'forward',3),(4,'original',4)]
        rows = [dict(sample_index=i, frame_index=i*5,
                     source_frame_index=original[j]['frame_index'], source_sample_index=j,
                     anchor_sample_index=a, kind=kind, pt=original[a]['pt'],
                     images=original[j]['images'])
                for i,(j,kind,a) in enumerate(sequence)]
        summary = dict(episode_id='Count/example/episode_1', num_samples=len(rows), task_prompt='task',
                       source=dict(split='train', source_id='Count/example/episode_1',
                                   suite='Count', task='Count/example', fps=25))
        return summary, rows, original

    def test_replays_are_independent_supervised_samples(self):
        s,r,o=self.fixture()
        check_source(s,r,o,'train')
        records=make_records(s,r,[str(i) for i in range(len(r))],'/pixels','train')
        self.assertEqual(len({x['id'] for x in records}),len(r))
        self.assertEqual([x['pt'] for x in records[4:9]],[0.3]*5)
        self.assertEqual(records[8]['history_source_frame_indices'],[5,10,15,15,10,5,10,15])
        self.assertEqual(records[8]['window_indices'],list(range(1,9)))
        self.assertEqual(records[0]['window_indices'],[0]*8)

    def test_rejects_replay_frame_pt_instead_of_anchor_pt(self):
        s,r,o=self.fixture()
        r[5]['pt']=o[2]['pt']
        with self.assertRaises(AssertionError):
            check_source(s,r,o,'train')

    def test_fractional_timeline_has_unique_integer_sample_ids(self):
        s,r,o=self.fixture()
        r[-1]['frame_index']=r[-2]['frame_index']+2.5
        check_source(s,r,o,'train')
        records=make_records(s,r,[str(i) for i in range(len(r))],'/pixels','train')
        self.assertEqual(records[-1]['id'],'Count/example/episode_1/00000009')
        self.assertEqual(records[-1]['frame_index'],42.5)

if __name__=='__main__':
    unittest.main(verbosity=2)
