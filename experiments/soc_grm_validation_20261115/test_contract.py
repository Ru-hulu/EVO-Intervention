import tempfile
import unittest
from pathlib import Path
from .common import native_input, check_checkpoint_metadata, freeze, resolve, sha


class Contract(unittest.TestCase):
    def test_eight_images_and_no_label_leakage(self):
        row = dict(frame_indices=[0,6,12],image_root='/images',videos=['a','b','c'],
                   task_prompt='Move the object.',target_pt=[0,.5,1])
        value = native_input(row,1,'prompt.json')
        self.assertEqual(value['images'],[f'/images/{f:08d}-{v}.png' for f,v in
                         [(0,0),(12,0),(0,0),(0,1),(0,2),(6,0),(6,1),(6,2)]])
        self.assertEqual(set(value),{'task','prompt_path','images'})
        row['target_pt'] = [1,0,0]
        self.assertEqual(value,native_input(row,1,'prompt.json'))

    def test_missing_camera_is_not_filled_with_another_view(self):
        row = dict(frame_indices=[0,6],image_root='/images',videos=['a','b',None],task_prompt='x')
        value = native_input(row,0,'p')['images']
        self.assertIsNone(value[4]); self.assertIsNone(value[7]); self.assertEqual(len(value),8)

    def metadata(self):
        proof = dict(status='passed',pilot=False,epoch=1,expected_episodes=35,suite='State',
            selection_sha256='s',train_input_sha256='d',optimizer_scheduler_two_rank_rng=True,
            replicas_equal=True,fresh_official_initialization=True,expected_samples=6381,steps=399,
            checkpoint='/ck/checkpoint-399')
        cfg = dict(suite='State',name='robobrain25-4b',epochs=1,episodes=35,training_fps=5,
            selection_sha256='s',train_sha256='d',future_endpoint_reference=True,samples=6381)
        return proof,cfg

    def test_wrong_suite_pilot_or_partial_epoch_rejected(self):
        p,c = self.metadata(); check_checkpoint_metadata('robobrain25-4b','State',p,c,'s')
        for key,value in [('pilot',True),('suite','Order'),('epoch',.9),('expected_episodes',10)]:
            bad = dict(p); bad[key] = value
            with self.assertRaises(AssertionError): check_checkpoint_metadata('robobrain25-4b','State',bad,c,'s')
        with self.assertRaises(AssertionError): check_checkpoint_metadata('dopamine-3b','State',p,c,'s')

    def test_backup_requires_exact_original_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); current = root/'current'; backup = root/'backup'
            current.mkdir(); backup.mkdir(); (current/'pt').write_text('new'); (backup/'pt').write_text('original')
            self.assertEqual(resolve(current/'pt',sha(backup/'pt'),current,backup),backup/'pt')
            with self.assertRaises(AssertionError): resolve(current/'pt','wrong',current,backup)

    def test_frozen_manifest_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'manifest.json'; first = freeze(p,{'checkpoint':'State'})
            self.assertEqual(first,freeze(p,{'checkpoint':'State'}))
            with self.assertRaises(AssertionError): freeze(p,{'checkpoint':'Order'})


if __name__ == '__main__': unittest.main()
