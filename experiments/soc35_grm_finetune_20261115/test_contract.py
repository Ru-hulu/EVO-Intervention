import unittest
from .common import configure_batch, choose_trial


class Contract(unittest.TestCase):
    def test_effective_batch_preserved(self):
        for micro in (2,4,8):
            r=configure_batch(dict(effective_batch=16,learning_rate=1e-4,epochs=1),micro,True)
            self.assertEqual(2*r['microbatch']*r['accumulation'],16)
            self.assertEqual(r['learning_rate'],1e-4);self.assertEqual(r['epochs'],1)
        with self.assertRaises(AssertionError):configure_batch(dict(effective_batch=16),3,False)

    def test_fastest_safe_choice(self):
        slow=dict(status='passed',samples_per_second=5,peak_reserved_gib=40)
        fast=dict(status='passed',samples_per_second=9,peak_reserved_gib=55)
        risky=dict(status='passed',samples_per_second=20,peak_reserved_gib=70)
        self.assertEqual(choose_trial([slow,fast,risky]),fast)
        with self.assertRaises(AssertionError):choose_trial([risky,dict(status='failed')])


if __name__=='__main__':unittest.main()
