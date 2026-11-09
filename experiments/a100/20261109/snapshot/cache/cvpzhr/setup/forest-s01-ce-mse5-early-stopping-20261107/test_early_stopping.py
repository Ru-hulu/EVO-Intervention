import json
import unittest
from early_stopping import validate_settings, update_early_stopping


class EarlyStoppingTests(unittest.TestCase):
    def config(self, **kwargs):
        cfg = dict(validation_sample_files=['val.jsonl'], **kwargs)
        validate_settings(cfg)
        return cfg

    def test_plateau_and_resume(self):
        cfg = self.config()
        self.assertFalse(update_early_stopping(cfg, 2.0, 10))
        self.assertFalse(update_early_stopping(cfg, 2.0, 20))
        cfg = json.loads(json.dumps(cfg))  # persisted checkpoint config
        self.assertFalse(update_early_stopping(cfg, 2.1, 30))
        self.assertTrue(update_early_stopping(cfg, 2.2, 40))
        self.assertEqual(cfg['early_stopping_state']['stop_step'], 40)
        self.assertTrue(update_early_stopping(cfg, 1.0, 50))

    def test_improvement_resets_patience_and_ranks_agree(self):
        ranks = [self.config() for _ in range(4)]
        for step, value in enumerate([2., 2.1, 1.8, 1.9, 1.9, 1.9]):
            decisions = [update_early_stopping(c, value, step) for c in ranks]
            self.assertEqual(decisions, [step == 5] * 4)
            self.assertTrue(all(c == ranks[0] for c in ranks))

    def test_min_delta_accumulates_against_reference(self):
        cfg = self.config(early_stopping_min_delta_pp=.1)
        for step, value in enumerate([2., 1.96, 1.92, 1.85]):
            self.assertFalse(update_early_stopping(cfg, value, step))
        self.assertEqual(cfg['early_stopping_state']['bad_validations'], 0)

    def test_disabled_and_invalid(self):
        cfg = dict(early_stopping_patience=0)
        validate_settings(cfg)
        self.assertFalse(update_early_stopping(cfg, 2., 1))
        self.assertNotIn('early_stopping_state', cfg)
        for bad in [-1, 1.5, True]:
            with self.assertRaises(ValueError): self.config(early_stopping_patience=bad)
        for bad in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): self.config(early_stopping_min_delta_pp=bad)
        with self.assertRaises(ValueError): validate_settings({})
        with self.assertRaises(ValueError): update_early_stopping(self.config(), float('nan'), 1)

    def test_completed_run_would_not_stop_prematurely(self):
        cfg = self.config()
        for step, value in enumerate([2.1656, 2.0642, 1.6279, 1.3767, 1.5467, 1.4547]):
            self.assertFalse(update_early_stopping(cfg, value, step))
        self.assertEqual(cfg['early_stopping_state']['bad_validations'], 2)


if __name__ == '__main__': unittest.main()
