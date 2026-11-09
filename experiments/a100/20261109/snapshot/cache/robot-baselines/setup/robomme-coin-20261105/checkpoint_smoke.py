"""Reload the selected checkpoint and run held-out dual-arm inference, without robot execution."""
import json
import jax,numpy as np
from coin_data import DATA,RUN,SETUP,CoinDataset,batch,absolute_actions
from train_coin import config
from openpi.models.model import restore_params
from openpi.shared.nnx_utils import module_jit

def main():
 best=json.loads((RUN/'best.json').read_text());c=config()
 model=c.model.load(restore_params(best['checkpoint']+'/params'))
 ds=CoinDataset('val',validation=True);obs,action=batch(ds,[0])
 predict=module_jit(model.sample_actions)
 pred=np.asarray(predict(jax.random.key(7),obs))
 eid,j=ds.indices[0];ep=ds.episode(eid);rawstate=ep['state'][ep['anchors'][j]]
 actions=absolute_actions(pred[0],rawstate)
 assert actions.shape==(20,14) and np.isfinite(actions).all()
 np.save(RUN/'reload-inference-actions.npy',actions)
 (SETUP/'checkpoint-smoke.json').write_text(json.dumps({'checkpoint':best['checkpoint'],'episode':eid,'output_shape':list(actions.shape),'finite':True,'robot_executed':False},indent=2))
if __name__=='__main__':main()
