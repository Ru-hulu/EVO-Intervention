import unittest
from .common import configure, ORDERS


class Contract(unittest.TestCase):
    def test_proven_batches_preserve_effective_batch(self):
        for name, micro, effective in [('robometer-4b',8,16), ('procvlm-2b',1,16),
                ('robobrain25-4b',4,16), ('dopamine-3b',4,16), ('forest',4,8)]:
            base = dict(name=name, effective_batch=effective, epochs=1, seed=42,
                learning_rate=1e-5 if name == 'procvlm-2b' else 1e-4,
                native=dict(model={}, history_capacity=1008))
            cfg = configure(base, micro, False)
            self.assertEqual(2*cfg['microbatch']*cfg['accumulation'], effective)
            self.assertEqual(cfg['learning_rate'], base['learning_rate'])
            self.assertEqual(cfg['epochs'], 1)
            self.assertNotIn('microbatch', base)
            if name == 'forest': self.assertEqual(cfg['native']['history_capacity'], 1008)

    def test_invalid_batch_rejected(self):
        with self.assertRaises(AssertionError): configure(dict(name='forest', effective_batch=8), 3, False)

    def test_pairs_cover_five_models_once(self):
        models = [m for order in ORDERS.values() for m in order]
        self.assertEqual(len(models), 5); self.assertEqual(len(set(models)), 5)
        self.assertEqual(ORDERS[0][0], 'robometer-4b'); self.assertEqual(ORDERS[1][0], 'procvlm-2b')


if __name__ == '__main__': unittest.main()
