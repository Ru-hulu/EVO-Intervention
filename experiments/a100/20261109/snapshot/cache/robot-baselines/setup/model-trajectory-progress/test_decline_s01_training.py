import unittest

from decline_s01_training import decline


class DeclineTest(unittest.TestCase):
    def test_failure_uses_visual_mass_and_native_zero_frame(self):
        old = [0., .2, .5, .5, .5, .5, 0., 0.]
        new, amount = decline(old, [0., 1., 1., 1., 0., 3., 1., 1.], 3, 6, failure=6)
        self.assertEqual(new[:3], old[:3])
        self.assertAlmostEqual(new[3], .475)
        self.assertEqual(new[3], new[4])
        self.assertAlmostEqual(new[5], .4)
        self.assertEqual(new[6:], [0., 0.])
        self.assertEqual(amount, .1)

    def test_recovery_is_gradual_and_rejoins_original(self):
        old = [0., .3, .5, .5, .5, .55, .6, .7, 1.]
        new, _ = decline(old, [1.] * 9, 3, 5, restored=6)
        self.assertAlmostEqual(new[4], .4)
        self.assertAlmostEqual(new[5], .5)
        self.assertEqual(new[6:], old[6:])

    def test_low_progress_cannot_be_zeroed_before_native_failure(self):
        new, amount = decline([0., .04, .04, .04, 0.], [1.] * 5, 2, 4, failure=4)
        self.assertEqual(amount, .02)
        self.assertGreater(new[3], 0)

    def test_no_visual_evidence_cannot_fall_back_to_clock(self):
        with self.assertRaises(AssertionError):
            decline([0., .5, .5, .5, 0.], [0.] * 5, 2, 4, failure=4)


if __name__ == '__main__':
    unittest.main()
