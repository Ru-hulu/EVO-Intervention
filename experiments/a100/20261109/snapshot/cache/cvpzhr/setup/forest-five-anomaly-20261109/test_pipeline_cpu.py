from pathlib import Path
import sys,json,tempfile,unittest,hashlib
from unittest.mock import patch
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S/'code'))
import torch
import evaluate_final as ev
from mem_lite.train_stage3 import save_checkpoint,checkpoint_schedule
class Checks(unittest.TestCase):
 def test_metrics(self):
  rows=[]
  for eid,task,y,p in [('a','task1',[0,.5,1],[0,.5,1]),('b','task1',[0,1],[1,0]),('c','task2',[.5,.5],[.2,.2])]:
   for i,(v,z) in enumerate(zip(y,p)):rows.append(dict(episode_id=eid,task=task,frame_index=i,target_pt=v,predicted_pt=z))
  result=ev.summarize(rows);self.assertAlmostEqual(result['overall']['pearson'],0)
  self.assertEqual(result['overall']['pearson_undefined_episodes'],1)
  self.assertAlmostEqual(result['overall']['mae'],2.6/7)
  self.assertIsNone(result['by_task']['task2']['pearson'])
 def test_balanced_coverage(self):
  records=[dict(episode_id=str(i),augmented_samples=100-i) for i in range(33)]
  split=ev.groups(records);self.assertEqual(sorted(x['episode_id'] for group in split for x in group),sorted(x['episode_id'] for x in records))
 def test_archive_all_epochs(self):
  model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters());cfg={'keep_only_latest':False,'save_every_steps':None,'save_every_epochs':1}
  with tempfile.TemporaryDirectory() as td,patch('torch.cuda.get_rng_state',return_value=torch.get_rng_state()),patch('mem_lite.train_stage3.trainable_state',return_value={'test':model.state_dict()}):
   for epoch in range(1,4):
    self.assertEqual(checkpoint_schedule(cfg,epoch*10,epoch,True),(True,True))
    save_checkpoint(model,optimizer,cfg,Path(td),epoch,0,epoch*10,True)
   self.assertEqual(sorted(p.name for p in Path(td).glob('*.pt')),['epoch-01.pt','epoch-02.pt','epoch-03.pt','last.pt'])
   self.assertEqual(torch.load(Path(td)/'epoch-03.pt',weights_only=True)['epoch'],3)
 def test_frozen_cache_code(self):
  plan=json.loads(Path('/cache/cvpzhr/datasets/forest-five-anomaly-5fps-static-jitter-seed42-20261109/plan.json').read_text())
  for rel,d in plan['code_sha256'].items():self.assertEqual(hashlib.sha256((S/rel).read_bytes()).hexdigest(),d,rel)
if __name__=='__main__':unittest.main()
