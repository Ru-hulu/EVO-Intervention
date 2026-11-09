"""Single-GPU LoRA fine-tuning with deterministic episode-held-out validation."""
import dataclasses,functools,json,logging,sys,time,math
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import nnx,traverse_util
import orbax.checkpoint as ocp
from coin_data import DATA,RUN,BASE,REPO,SETUP,CoinDataset,batch,absolute_actions
sys.path.insert(0,str(REPO))
from scripts.train import init_train_state,train_step
from mme_vla_suite.training import config as cfg
from openpi.training import optimizer,sharding,checkpoints
from openpi.models import model as base_model
from openpi.shared import normalize

@dataclasses.dataclass(frozen=True)
class StrictLoader:
 params_path:str=str(BASE/'params')
 def load(self,reference):
  loaded=traverse_util.flatten_dict(base_model.restore_params(self.params_path,restore_type=np.ndarray),sep='/')
  ref=traverse_util.flatten_dict(reference,sep='/');missing=[];out={}
  for k,v in ref.items():
   if k not in loaded:
    assert 'lora' in k,('unexpected missing pretrained tensor',k)
    out[k]=v;missing.append(k)
   else:
    assert loaded[k].shape==v.shape,(k,loaded[k].shape,v.shape)
    out[k]=loaded[k].astype(v.dtype)
  extra=sorted(set(loaded)-set(ref)); assert not extra,extra
  SETUP.mkdir(exist_ok=True)
  (SETUP/'weight-load-audit.json').write_text(json.dumps({'checkpoint':self.params_path,'loaded_tensors':len(loaded),'new_lora_tensors':missing,'unexpected_missing':[],'unexpected_unused':[]},indent=2))
  return traverse_util.unflatten_dict(out,sep='/')

def config():
 base=cfg.get_config('mme_vla_suite')
 model=dataclasses.replace(base.model,history_config='perceptual-framesamp-modul.yaml',paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',max_token_len=128,discrete_state_input=True)
 return dataclasses.replace(base,exp_name='state-coin-20261105',model=model,freeze_filter=model.get_freeze_filter(),weight_loader=StrictLoader(),batch_size=8,fsdp_devices=1,num_train_steps=10000,ema_decay=None,num_workers=0,wandb_enabled=False,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=200,peak_lr=2e-5,decay_steps=10000,decay_lr=2e-6),optimizer=optimizer.AdamW(clip_gradient_norm=1.0,weight_decay=1e-5))

def write_status(**values):
 values['time']=time.strftime('%Y-%m-%dT%H:%M:%S%z');p=SETUP/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(values,indent=2));tmp.replace(p)

def main():
 logging.basicConfig(level=logging.INFO)
 assert jax.default_backend()=='gpu' and jax.device_count()==1
 assert (DATA/'features-complete.json').exists()
 c=config();RUN.mkdir(parents=True,exist_ok=True)
 if (RUN/'metrics.jsonl').exists():raise RuntimeError('Existing run: refusing to overwrite; inspect before resuming')
 (RUN/'history_config.txt').write_text(c.model.history_config)
 description={'base_checkpoint':str(BASE),'model':dataclasses.asdict(c.model),'train':{'batch_size':8,'max_steps':10000,'warmup':200,'lr':2e-5,'min_lr':2e-6,'grad_clip':1.,'weight_decay':1e-5,'seed':42,'eval_every':500,'early_stop_patience':5,'min_steps_before_early_stop':2000,'selection_metric':'val_action_normalized_mae_14d','trainable':'language/action LoRA + memory/modulation + action projections','frozen':'vision and base language/action transformer','ema':False},'data':json.loads((DATA/'dataset.json').read_text()),'split':json.loads((DATA/'split.json').read_text()),'validation':'32 temporal anchors per held-out episode for flow loss; 4 per episode for fixed-noise 10-step action rollout MAE; no train augmentation in validation','deployment':'Use coin_data adapter; 14-dim outputs, delta joints and absolute grippers, native 30 Hz actions.'}
 (RUN/'run_config.json').write_text(json.dumps(description,ensure_ascii=False,indent=2))
 jax.config.update('jax_compilation_cache_dir',str(ROOT_CACHE))
 mesh=sharding.make_mesh(1); replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 write_status(stage='initializing',gpu=jax.devices()[0].device_kind)
 state,state_sharding=init_train_state(c,jax.random.key(42),mesh,resume=False);jax.block_until_ready(state)
 ds=CoinDataset('train');val=CoinDataset('val',validation=True)
 trainable=sum(x.size for x in jax.tree.leaves(state.params.filter(c.trainable_filter)))
 (RUN/'parameter-counts.json').write_text(json.dumps({'trainable':trainable,'total':sum(x.size for x in jax.tree.leaves(state.params))}))
 step_fn=jax.jit(functools.partial(train_step,c),donate_argnums=(1,))
 @jax.jit
 def eval_fn(params,obs,actions,key):
  m=nnx.merge(state.model_def,params);m.eval();loss,_=m.compute_loss(key,obs,actions,train=False);return jnp.mean(loss)
 @jax.jit
 def predict_fn(params,obs,key):
  m=nnx.merge(state.model_def,params);m.eval();return m.sample_actions(key,obs,num_steps=10)
 val_batches=[np.arange(i,i+8) for i in range(0,len(val),8)]
 # Four evenly-spaced rollout anchors in every held-out episode.
 rollout_ids=[]
 for eid in c_ids:
  ids=[i for i,(e,_) in enumerate(val.indices) if e==eid]
  rollout_ids.extend(np.asarray(ids)[np.linspace(0,len(ids)-1,4,dtype=int)].tolist())
 rollout_batches=[rollout_ids[i:i+8] for i in range(0,len(rollout_ids),8)]
 stats=normalize.load(DATA)
 def save_assets(path):
  normalize.save(path/'piper_coin',stats)
  (path/'run_config.json').write_text((RUN/'run_config.json').read_text())
 manager=ocp.CheckpointManager(RUN/'checkpoints',item_handlers={'params':ocp.PyTreeCheckpointHandler(),'train_state':ocp.PyTreeCheckpointHandler(),'assets':checkpoints.CallbackHandler()},options=ocp.CheckpointManagerOptions(max_to_keep=2,best_fn=lambda x:x['val_action_normalized_mae_14d'],best_mode='min',async_options=ocp.AsyncOptions(timeout_secs=7200)))
 rng=np.random.default_rng(42);order=rng.permutation(len(ds));cursor=0;best=float('inf');bad=0;start=time.time()
 def evaluate(step):
  losses=[];errors=[];native=[]
  for i,indices in enumerate(val_batches):
   obs,act=batch(val,indices)
   losses.append(float(eval_fn(state.params,obs,act,jax.random.fold_in(jax.random.key(123),i))))
  for i,indices in enumerate(rollout_batches):
   obs,act=batch(val,indices)
   pred=np.asarray(predict_fn(state.params,obs,jax.random.fold_in(jax.random.key(456),i)))
   err=np.abs(pred[...,:14]-act[...,:14]);errors.append(err.mean((0,1)))
   st=stats['actions'];native.append((err*(np.asarray(st.q99)-np.asarray(st.q01)+1e-6)/2).mean((0,1)))
  per_joint=np.mean(errors,0);assert np.isfinite(per_joint).all() and np.isfinite(losses).all()
  return {'step':step,'val_flow_loss_32d':float(np.mean(losses)),'val_action_normalized_mae_14d':float(per_joint.mean()),'val_per_dimension_normalized_mae':per_joint.tolist(),'val_per_dimension_native_mae':np.mean(native,0).tolist(),'validation_episodes':20,'flow_samples':len(val),'rollout_samples':len(rollout_ids)}
 def record(x):
  x['elapsed_seconds']=time.time()-start
  with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
 baseline=evaluate(0);record(baseline)
 for step in range(1,c.num_train_steps+1):
  if cursor+8>len(order):order=rng.permutation(len(ds));cursor=0
  b=batch(ds,order[cursor:cursor+8]);cursor+=8
  with sharding.set_mesh(mesh):state,info,_=step_fn(jax.random.key(42),state,b)
  if step==1 or step%20==0:
   out={k:float(v) for k,v in jax.device_get(info).items()};assert all(math.isfinite(x) for x in out.values()),out
   record({'step':step,**out,'lr':float(c.lr_schedule.create()(step))});write_status(stage='training',step=step,max_steps=c.num_train_steps,best_val_mae=best if math.isfinite(best) else None,metrics=out)
  if step==20 or step%500==0 or step==c.num_train_steps:
   result=evaluate(step);record(result);metric=result['val_action_normalized_mae_14d']
   manager.save(step,items={'params':{'params':state.params},'train_state':state,'assets':save_assets},metrics=result);manager.wait_until_finished()
   if metric<best:
    best=metric;bad=0;(RUN/'best.json').write_text(json.dumps({'step':step,'metric':best,'checkpoint':str(RUN/'checkpoints'/str(step))},indent=2))
   elif step>=2000:bad+=1
   write_status(stage='training',step=step,max_steps=c.num_train_steps,best_val_mae=best,latest_validation=result)
   if bad>=5:break
 manager.wait_until_finished();manager.close()
 write_status(stage='completed',step=step,best=json.loads((RUN/'best.json').read_text()),elapsed_seconds=time.time()-start)
ROOT_CACHE=Path('/cache/robot-baselines/.cache/jax-robomme-coin')
c_ids=json.loads((DATA/'split.json').read_text())['val'] if (DATA/'split.json').exists() else []
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  write_status(stage='failed',error=traceback.format_exc());raise
