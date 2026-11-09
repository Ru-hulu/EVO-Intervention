import json,tempfile,unittest
from pathlib import Path
import finalize
class PublishTest(unittest.TestCase):
 def test_preserves_outcomes_and_archives_old_labels(self):
  with tempfile.TemporaryDirectory() as tmp:
   finalize.ROOT=Path(tmp);finalize.OUT=Path(tmp)/'output';finalize.DATASETS=['test']
   root=Path(tmp)/'viewer-datasets/test';root.mkdir(parents=True)
   row={'sample_id':'x','task_success':False,'fps':30,'total_frames':3,'annotation_path':'x/annotation.json'}
   original=json.dumps(row)+'\n';(root/'metadata.jsonl').write_text(original)
   finalize.write(root/'x/annotation.json',{'stages':[],'progress':{'dense_progress':[0,0,0]}})
   finalize.write(root/'progress/x.json',{'dense_progress':[0,0,0]})
   finalize.write(finalize.OUT/'integration-status.json',{'pending':0,'issues':[]})
   p={'policy':'test-integral','dense_progress':[0,.2,.5]}
   finalize.write(finalize.OUT/'test/progress/x.json',p)
   finalize.write(finalize.OUT/'test/boundaries/x.json',{'events':[{'frame':0,'done':[]},{'frame':2,'done':['a']}]})
   finalize.publish()
   new=json.loads((root/'metadata.jsonl').read_text())
   self.assertFalse(new['task_success']);self.assertFalse(new['training_eligible'])
   self.assertEqual(json.loads((root/'progress/x.json').read_text()),p)
   self.assertEqual((finalize.OUT/'test/original/metadata.jsonl').read_text(),original)
   self.assertEqual(json.loads((finalize.OUT/'test/original/x.progress.json').read_text())['dense_progress'],[0,0,0])
if __name__=='__main__':unittest.main()
