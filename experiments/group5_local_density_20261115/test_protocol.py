import tempfile
import unittest
from pathlib import Path
from .protocol import *
from .common import freeze


def fixture(n):
    row=dict(key='State/S12/example',suite='State',task_id='S12',frames=n,progress_sha256='truth')
    old={}
    for rate in RATES:
        frames=[];k=0
        while (k*30+rate-1)//rate<n:
            frames.append((k*30+rate-1)//rate);k+=1
        if frames[-1]!=n-1:frames.append(n-1)
        common=list(range(0,n,5));lookup={f:i for i,f in enumerate(frames)}
        old[rate]=dict(source_frame_indices=frames,target_pt=[f/(n-1) for f in frames],timestamps=[f/30 for f in frames],
            common_scoring_indices=[lookup[f] for f in common],common_scoring_native_frames=common,
            common_scoring_timestamps=[f/30 for f in common])
    return row,old


class Protocol(unittest.TestCase):
    def test_stable_unique_pairs(self):
        order=segment_order('example');self.assertEqual(order,segment_order('example'))
        self.assertEqual(sorted(order),list(range(10)));self.assertNotEqual(order,segment_order('another'))
        self.assertEqual(sorted(segment_rates(order,5)),[6,6,12,12,18,18,24,24,30,30])

    def test_equal_boundaries_no_loss_or_overlap(self):
        for n in (10,19,487,1000):
            bins=[[f for f in range(n) if segment_of(f,n)==j] for j in range(10)]
            self.assertEqual([f for group in bins for f in group],list(range(n)))
            self.assertLessEqual(max(map(len,bins))-min(map(len,bins)),1)

    def test_baseline_exact_all_fields(self):
        for n in (19,487,1001):
            row,old=fixture(n);entry=make_entry(row,old,1,segment_order(row['key']))
            for k in ('source_frame_indices','target_pt','timestamps','common_scoring_indices'):
                self.assertEqual(entry[k],old[6][k])

    def test_cumulative_new_segments_only(self):
        row,old=fixture(487);order=segment_order(row['key']);previous=make_entry(row,old,1,order)
        for level in range(2,6):
            current=make_entry(row,old,level,order)
            self.assertTrue(set(previous['source_frame_indices'])<set(current['source_frame_indices']))
            self.assertEqual({i for i in range(10) if previous['segment_rates'][i]!=current['segment_rates'][i]},set(order[(level-2)*2:(level-1)*2]))
            self.assertEqual([current['source_frame_indices'][i] for i in current['common_scoring_indices']],list(range(0,487,5)))
            self.assertEqual(current['target_pt'],[f/486 for f in current['source_frame_indices']])
            previous=current

    def test_history_uses_observation_index(self):
        self.assertEqual(slots(0,'procvlm-2b'),[[0,0]]*8)
        self.assertEqual(slots(12,'procvlm-2b'),[[i,0] for i in range(5,13)])
        for i in (0,2,7,100):
            self.assertEqual(len(slots(i,'robometer-4b')),10)
            self.assertTrue(all(j<=i for j,v in slots(i,'robometer-4b')))
            self.assertEqual(slots(i,'forest'),[[i,v] for v in range(3)])

    def test_no_target_in_inference(self):
        row,old=fixture(487);row.update(image_root='/images',task_prompt='Task',videos=['main','left',None])
        e=make_entry(row,old,3,segment_order(row['key']));inf=inference_episode(row,e)
        self.assertEqual(set(inf),{'key','task','positions','images'})
        self.assertTrue(all(images[2] is None for images in inf['images']))

    def test_frozen_cache_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'frozen.json';first=freeze(p,dict(seed=42));self.assertEqual(first,freeze(p,dict(seed=42)))
            with self.assertRaises(AssertionError):freeze(p,dict(seed=43))


if __name__=='__main__':unittest.main()
