import unittest
from .protocol import endpoint_score,parse_progress,current_sample,current_robometer_messages
from .report import summary

class ContractTests(unittest.TestCase):
    def test_tail_uses_sample_count_and_preserves_invalid(self):
        result=endpoint_score([0.]*99+[.2,.6])
        self.assertEqual(result['endpoint_indices'],[98,99,100])
        self.assertAlmostEqual(result['endpoint_score'],.8/3)
        self.assertIsNone(endpoint_score([0.]*49+[None])['endpoint_score'])
        self.assertEqual(endpoint_score([None]*49+[.8])['endpoint_score'],.8)

    def test_prediction_scale_and_malformed_output(self):
        self.assertEqual(parse_progress('<progress>25.00%</progress>'),.25)
        for bad in ['25','<progress>125%</progress>','<progress>NaN%</progress>','explanation <progress>10%</progress>']:
            self.assertIsNone(parse_progress(bad))

    def test_request_ignores_labels_and_history(self):
        row=dict(task_prompt='test',frame_indices=[0,5],image_root='/x',videos=['a','b',None],
            target_pt=[0,1],future_endpoint='/secret',history_tokens=[123])
        self.assertEqual(current_sample(row,1,'procvlm-2b'),dict(task='test',images=['/x/00000005-0.png']))
        self.assertEqual(current_sample(row,1,'forest')['images'],['/x/00000005-0.png','/x/00000005-1.png',None])
        content=[dict(type='text',text='task'),dict(type='text',text='Historical observations')]
        content += [dict(type='image'),dict(type='text',text='<|prog_token|>')]*7
        for view in ['main','left','right']:content += [dict(type='text',text='Current observation, '+view),dict(type='image')]
        content += [dict(type='text',text='<|prog_token|>')]
        result=current_robometer_messages([dict(role='user',content=content)])[0]['content']
        self.assertEqual(sum(x['type']=='image' for x in result),3)
        self.assertEqual(sum(x.get('text')=='<|prog_token|>' for x in result),1)

    def test_incomplete_coverage_not_reported_as_full_mean(self):
        row=dict(pearson=.5,spearman=.4,VS=.3,ER=.6,C=.5,SA=.8,invalid_predictions=0)
        self.assertIsNone(summary([row],2)['metrics']['SA']['mean'])
        self.assertEqual(summary([row],1)['metrics']['SA']['mean'],.8)

if __name__=='__main__':unittest.main()
