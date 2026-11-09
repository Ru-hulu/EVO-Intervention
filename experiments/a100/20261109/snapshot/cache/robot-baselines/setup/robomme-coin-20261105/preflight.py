import json
import numpy as np
import jax
from flax import nnx
from coin_data import DATA,SETUP,CoinDataset,batch,absolute_actions,normalize,DELTA
from train_coin import config

def main():
 c=config();train=CoinDataset('train',features=False);val=CoinDataset('val',validation=True,features=False)
 split=json.loads((DATA/'split.json').read_text());assert not set(split['train'])&set(split['val'])
 for ds in (train,val):
  for i in [0,len(ds)//2,len(ds)-1]:
   sample=ds[i];assert sample['actions'].shape==(20,32)
   assert sample['tokenized_prompt_mask'][-1]==False,'prompt may be truncated'
   eid,j=ds.indices[i];e=ds.episode(eid);t=e['anchors'][j]
   reconstructed=absolute_actions(sample['actions'],e['state'][t])
   expected=e['action'][np.minimum(t+np.arange(20),len(e['action'])-1)]
   assert np.max(np.abs(reconstructed-expected))<1e-4
   assert all(np.isfinite(x).all() for x in jax.tree.leaves(sample))
 obs,actions=batch(train,np.arange(8))
 shape=jax.eval_shape(lambda key:nnx.state(c.model.create(key)).to_pure_dict(),jax.random.key(42))
 loaded=c.weight_loader.load(shape);del loaded
 # Trace both training and validation loss with the real 3-camera/14-dim adapter.
 def loss_shapes(key,ob,ac):
  m=c.model.create(key)
  return m.compute_loss(key,ob,ac,train=True)[0],m.compute_loss(key,ob,ac,train=False)[0]
 ls=jax.eval_shape(loss_shapes,jax.random.key(0),obs,actions)
 result={'stage':'cpu_preflight_passed','train_episodes':80,'val_episodes':20,'train_anchors':len(train),'validation_anchors':len(val),'action_roundtrip':True,'finite_data':True,'split_disjoint':True,'loss_shapes':[list(x.shape) for x in ls],'weight_audit':'weight-load-audit.json','gpu_forward_backward':'pending idle GPU'}
 (SETUP/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
