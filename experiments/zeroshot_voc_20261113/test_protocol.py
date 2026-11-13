"""Semantic checks for temporal leakage, metric target, ties and invalid outputs."""
import unittest
from common import correlations, uniform_indices
from worker import images_for, parse


class ProtocolTests(unittest.TestCase):
    def test_uniform_is_causal_and_right_padded(self):
        from runtime.robometer_training.uniform import prefix_indices
        self.assertEqual(uniform_indices(2), [0,1,2,2,2,2,2,2])
        for i in range(50):
            ids=uniform_indices(i)
            self.assertEqual(ids,prefix_indices(i))
            self.assertEqual(len(ids),8)
            self.assertEqual(ids[-1],i)
            self.assertTrue(all(0<=j<=i for j in ids))
            self.assertEqual(ids,sorted(ids))

    def test_actual_frames_and_goal(self):
        e={'frames':[{'image':[f'm{i}',f'l{i}',f'r{i}']} for i in range(20)]}
        self.assertEqual(images_for('grm',e,4),['m0','m19','m0','l0','r0','m4','l4','r4'])
        self.assertEqual(images_for('proc',e,2),['m0']*6+['m1','m2'])
        self.assertEqual(images_for('robometer',e,2),['m0','m1','m2','m2','m2','m2','m2','m2','l2','r2'])
        self.assertEqual(images_for('vlac',e,4),['m3','m4'])
        self.assertEqual(images_for('r2',e,4),['m4'])

    def test_correlation_uses_pt_not_time_and_preserves_ties(self):
        target=[0,.5,.5,.2,1]
        p,s,reason=correlations(target,target)
        self.assertAlmostEqual(p,1);self.assertAlmostEqual(s,1);self.assertIsNone(reason)
        self.assertLess(correlations(list(range(5)),target)[1],1)
        self.assertEqual(correlations([.5]*5,target)[2],'constant_prediction')
        self.assertEqual(correlations([0,None,.5,.2,1],target)[2],'invalid_predictions')

    def test_no_imputation_of_malformed_responses(self):
        self.assertEqual(parse('<score>+25%</score>','score',-100,100),.25)
        self.assertEqual(parse('<progress>25.5</progress>','progress',0,100),.255)
        self.assertEqual(parse('<think>3 steps</think><answer>45</answer>','answer',0,100),.45)
        self.assertEqual(parse('-20','vlac',-100,100),-.2)
        self.assertIsNone(parse('probably about 50','progress',0,100))
        self.assertIsNone(parse('<answer>101</answer>','answer',0,100))


if __name__=='__main__':unittest.main()
