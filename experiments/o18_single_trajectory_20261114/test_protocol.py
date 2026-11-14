import unittest
from experiments.o18_single_trajectory_20261114.run import select_source, entry_for
from experiments.prm_five_groups_20261114.stream_runtime import mapped_episode
from experiments.prm_five_groups_20261114.eval_common import sampling_indices

class IllustrationProtocol(unittest.TestCase):
    def test_requires_explicit_validation_membership(self):
        s='evomind_O18_episode_000028';m={'tasks':[{'task_id':'O18','train':['train'], 'val':[s]}]}
        rows=[dict(sample_id=s,task_id='O18',total_frames=1724,fps=30)]
        self.assertEqual(select_source(m,rows)[1],rows[0])
        m['tasks'][0]['train'].append(s)
        with self.assertRaises(AssertionError):select_source(m,rows)
    def test_exact_sampling_and_no_truth_in_model_request(self):
        indices=sampling_indices(1724,'30');self.assertEqual(len(indices),289)
        self.assertEqual(indices[:3],[0,6,12]);self.assertEqual(indices[-2:],[1722,1723])
        row=dict(key='Order/O18/test',suite='Order',task_id='O18',progress_sha256='sha',input_issues=[],
            source_rate='30',target_pt=[i/288 for i in range(289)],frame_indices=indices,
            timestamps=[i/30 for i in indices],frames=1724,task_prompt='Inspect bottles',videos=['main','left','right'])
        entry=entry_for(row);episode=mapped_episode(row,entry,'/images','grm')
        self.assertEqual(set(episode),{'task','_memory_enabled','frames'})
        self.assertNotIn('target_pt',str(episode));self.assertEqual(len(episode['frames']),289)
        entry['target_pt'][0]=.3
        with self.assertRaises(AssertionError):mapped_episode(row,entry,'/images','grm')

if __name__=='__main__':unittest.main()
