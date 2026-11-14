import copy
import unittest

from .r2_subset import draw, validate_saved, MODEL
from .eval_common import trajectory_metrics
from .group1_report import summarize


class SubsetProtocolTests(unittest.TestCase):
    def test_selection_is_order_independent_and_has_no_duplicate(self):
        keys = [f'Count/C01/episode_{i:03d}' for i in range(177)]
        expected = [6, 26, 28, 35, 57, 62, 70, 139, 163, 173]
        self.assertEqual(draw(keys), [keys[i] for i in expected])
        self.assertEqual(draw(keys[::-1]), draw(keys))
        with self.assertRaises(AssertionError):
            draw(keys + [keys[0]])

    def test_reused_result_requires_same_frame_map_truth_and_raw_coverage(self):
        row = dict(key='State/S01/episode', suite='State', task_id='S01',
            target_pt=[0., .5, 1.], frame_indices=[0, 6, 12], timestamps=[0., .2, .4],
            progress_sha256='label', video_hashes=['video', None, None])
        record = dict(copy.deepcopy(row), model=MODEL, selection_sha256='selection',
            source_video_sha256=row['video_hashes'], predictions=[0., .4, .9], raw_responses=['0','40','90'],
            metrics=trajectory_metrics([0., .4, .9],row['target_pt'],row['timestamps']))
        validate_saved(record, row, 'selection')
        for key, replacement in [('frame_indices',[0,5,12]),('target_pt',[0.,.6,1.]),('raw_responses',['0','40'])]:
            bad=copy.deepcopy(record);bad[key]=replacement
            with self.assertRaises(AssertionError):validate_saved(bad,row,'selection')

    def test_incomplete_and_undefined_are_not_reported_as_complete_means(self):
        good=trajectory_metrics([0.,.4,1.],[0.,.5,1.],[0.,.2,.4])
        constant=trajectory_metrics([.5,.5,.5],[0.,.5,1.],[0.,.2,.4])
        incomplete=summarize([good],10,30)
        self.assertIsNone(incomplete['defined_trajectory_means']['pearson'])
        complete=summarize([good]*9+[constant],10,30)
        self.assertEqual(complete['defined_episode_counts']['pearson'],9)
        self.assertEqual(complete['undefined_episode_counts']['pearson'],1)
        self.assertAlmostEqual(complete['defined_trajectory_means']['pearson'],good['pearson'])


if __name__=='__main__':unittest.main()
