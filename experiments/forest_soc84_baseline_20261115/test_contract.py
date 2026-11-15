import unittest
from .run import shards,model_input,validate,digest
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics


class Contract(unittest.TestCase):
    def test_shards_cover_once_and_balance_observation_work(self):
        rows=[dict(key=str(i),frame_indices=list(range(n))) for i,n in enumerate((100,80,30,20,10))]
        parts=shards(rows)
        self.assertEqual(set(parts[0])|set(parts[1]),{r['key'] for r in rows})
        self.assertFalse(set(parts[0])&set(parts[1]))
        self.assertEqual([sum(len(rows[int(k)]['frame_indices']) for k in p) for p in parts],[120,120])

    def test_baseline_has_no_future_or_ground_truth_in_memory_input(self):
        row=dict(key='State/S04/a',task_prompt='task',image_root='/images',videos=['main','left',None],
                 frame_indices=[0,4,7,10],target_pt=[0,.3,.7,1],suite='State')
        inp=model_input(row)
        self.assertEqual(set(inp),{'key','task','images','positions'})
        self.assertEqual(inp['positions'],[0,4,7,10])
        self.assertEqual(inp['images'][1],['/images/00000004-0.png','/images/00000004-1.png',None])
        row['target_pt']=[1]*4
        self.assertEqual(inp,model_input(row))

    def test_invalid_predictions_remain_undefined(self):
        score=trajectory_metrics([0,.1,None,.3,.4,.5,.6],[i/6 for i in range(7)],[i/5 for i in range(7)])
        self.assertEqual(score['invalid_predictions'],1)
        for name in ('pearson','spearman','VS'):self.assertIsNone(score[name])

    def test_prefix_rejects_wrong_checkpoint_and_changed_frames(self):
        row=dict(suite='State',frame_indices=[0],timestamps=[0],target_pt=[0])
        ident=dict(checkpoint_sha256='a');audit=dict(strict_final_state=True,memory_enabled=True,
            future_endpoint_reference=False,checkpoint_sha256='a',suite='State')
        r=dict(identity=ident,predictions=[0.2],raw_responses=[None],input_audits=[dict(playback_step=0,
            playback_position=0,input_images=3,input_playback_slots=[[0,0],[0,1],[0,2]])],
            frame_indices=[0],timestamps=[0],target_pt=[0],initialization=audit,initialization_sha256=digest(audit),status='completed')
        self.assertEqual(validate(r,row,ident,True),1)
        with self.assertRaises(AssertionError):validate(r,row,dict(checkpoint_sha256='other'),True)
        r['frame_indices']=[1]
        with self.assertRaises(AssertionError):validate(r,row,ident,True)


if __name__=='__main__':unittest.main()
