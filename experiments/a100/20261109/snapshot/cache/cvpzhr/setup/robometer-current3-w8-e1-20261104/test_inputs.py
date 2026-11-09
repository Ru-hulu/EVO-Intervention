import unittest
from prepare_cache import make_records, check_source, VIEWS
from robometer_training.model import progress_messages
from robometer_training.data import ObservationWindow

class InputChecks(unittest.TestCase):
    def fixture(self):
        originals=[dict(frame_index=i*5,pt=i/10,images={v:f'{v}/{i}.png' for v in VIEWS}) for i in range(5)]
        timeline=[(0,'original',0),(1,'original',1),(2,'original',2),(3,'original',3),
                  (3,'stationary',3),(2,'reverse',3),(1,'reverse',3),
                  (2,'forward',3),(3,'forward',3),(4,'original',4)]
        rows=[dict(sample_index=i,frame_index=i*5,source_frame_index=originals[j]['frame_index'],
              source_sample_index=j,anchor_sample_index=a,kind=kind,pt=originals[a]['pt'],images=originals[j]['images'])
              for i,(j,kind,a) in enumerate(timeline)]
        summary=dict(episode_id='Count/example/ep',num_samples=len(rows),task_prompt='task',
                     source=dict(split='train',source_id='Count/example/ep',suite='Count',task='Count/example',fps=25))
        images=[r['images'] for r in rows]
        return summary,rows,originals,images

    def test_current_three_images_are_one_progress_observation(self):
        content=progress_messages('task',10)[0]['content']
        images=0;progress_after=[]
        for item in content:
            if item['type']=='image':images+=1
            if item.get('text')=='<|prog_token|>':progress_after.append(images)
        self.assertEqual(images,10)
        self.assertEqual(progress_after,[1,2,3,4,5,6,7,10])
        with self.assertRaises(ValueError):progress_messages('task',8)

    def test_causal_history_and_synchronized_views_match_online_window(self):
        s,rows,originals,images=self.fixture()
        check_source(s,rows,originals,'train')
        records=make_records(s,rows,images,'/cache/pixels','train')
        window=ObservationWindow(8)
        for i,r in enumerate(records):
            expected=window.push(s['episode_id'],images[i]['cam_high'])[:-1]+[images[i][v] for v in VIEWS]
            self.assertEqual(r['image'],expected)
            self.assertEqual(r['window_indices'][-3:],[i,i,i])
            self.assertEqual(r['input_views'],['cam_high']*8+['cam_left_wrist','cam_right_wrist'])
        self.assertEqual(records[0]['image'],[images[0]['cam_high']]*8+[images[0]['cam_left_wrist'],images[0]['cam_right_wrist']])
        self.assertEqual(window.push('another_episode','new')[0],'new')

    def test_every_replayed_timestep_retains_anchor_pt(self):
        s,rows,originals,images=self.fixture()
        result=make_records(s,rows,images,'/cache/pixels','train')
        self.assertEqual(len({r['id'] for r in result}),len(rows))
        self.assertEqual([r['pt'] for r in result[4:9]],[.3]*5)
        rows[5]['pt']=originals[2]['pt']
        with self.assertRaises(AssertionError):check_source(s,rows,originals,'train')

    def test_fractional_augmented_positions_keep_unique_ids(self):
        s,rows,originals,images=self.fixture()
        rows[-1]['frame_index']=42.5
        check_source(s,rows,originals,'train')
        result=make_records(s,rows,images,'/cache/pixels','train')
        self.assertEqual(result[-1]['id'],'Count/example/ep/00000009')
        self.assertEqual(result[-1]['frame_index'],42.5)
    def test_absent_camera_keeps_a_current_view_slot_and_same_pt(self):
        s,rows,originals,images=self.fixture()
        for row in originals:
            row['images']['cam_right_wrist']=None
        images=[{**row['images'],'cam_right_wrist':'/images/_missing_rgb64.png'} for row in rows]
        check_source(s,rows,originals,'train')
        result=make_records(s,rows,images,'/cache/pixels','train')
        self.assertTrue(all(r['image'][-1]=='/images/_missing_rgb64.png' for r in result))
        self.assertTrue(all(r['missing_current_views']==['cam_right_wrist'] for r in result))
        self.assertEqual([r['pt'] for r in result],[r['pt'] for r in rows])

if __name__=='__main__':unittest.main(verbosity=2)
