import unittest
from pathlib import Path
from .calculate import native_functions,choose,score


class Metric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native=native_functions(Path(__file__).parents[1]/'prm_five_groups_20261114/metric_reference/fpl.py')

    def test_window_largest_not_nearest(self):
        d=[(5,10,1.),(40,45,.7),(49,51,.2)]
        chosen,candidates=choose(d,50,100,self.native)
        self.assertEqual(chosen,(40,45,.7));self.assertEqual(len(candidates),2)

    def test_fallback_to_global(self):
        self.assertEqual(choose([(5,10,1.),(90,95,.7)],50,100,self.native),((5,10,1.),[]))

    def test_inclusive_boundary_and_onset_only(self):
        chosen,candidates=choose([(30,35,.8),(29,50,1.),(70,75,.3)],50,100,self.native)
        self.assertEqual(chosen,(30,35,.8));self.assertEqual(len(candidates),2)

    def test_native_tie_break(self):
        self.assertEqual(choose([(40,46,.5),(60,64,.5),(50,54,.5)],50,100,self.native)[0],(50,54,.5))

    def test_no_prediction_descent_penalty(self):
        r=score([0,.1,.2,.3,.4],[0,1,.5,.8,1],[0,25,50,75,100],101,self.native)
        self.assertEqual(r['value'],.75);self.assertEqual(r['selection_mode'],'no_descent_penalty')

    def test_no_ground_truth_descent(self):
        r=score([0,1,0],[0,.5,1],[0,50,100],101,self.native)
        self.assertIsNone(r['value']);self.assertEqual(r['reason'],'no_ground_truth_descent')

    def test_invalid_predictions_remain_undefined(self):
        r=score([0,None,0],[0,1,.5],[0,50,100],101,self.native)
        self.assertIsNone(r['value']);self.assertEqual(r['reason'],'invalid_predictions')

    def test_exact_onset_match(self):
        r=score([0,1,.5,1],[0,.8,.3,.9],[0,40,60,100],101,self.native)
        self.assertEqual(r['value'],0);self.assertEqual(r['selection_mode'],'within_window')

    def test_nearest_ignores_larger_distant_drop(self):
        d=[(5,10,1.),(40,45,.7),(49,51,.2)]
        self.assertEqual(choose(d,50,100,self.native,'nearest')[0],(49,51,.2))

    def test_nearest_empty_window_still_uses_global_largest(self):
        d=[(5,10,1.),(75,80,.2)]
        self.assertEqual(choose(d,50,100,self.native,'nearest'),((5,10,1.),[]))

    def test_nearest_equal_distance_tie(self):
        d=[(45,48,.3),(55,59,.6),(55,58,.6)]
        self.assertEqual(choose(d,50,100,self.native,'nearest')[0],(55,58,.6))

    def test_nearest_boundary_included(self):
        self.assertEqual(choose([(30,35,.2)],50,100,self.native,'nearest')[0],(30,35,.2))


if __name__=='__main__':unittest.main()
