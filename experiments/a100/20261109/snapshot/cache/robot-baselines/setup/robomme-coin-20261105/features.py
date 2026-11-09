"""Cache the frozen checkpoint's environment-camera tokens; no fitting on validation."""
import json,time,gc
import numpy as np
import jax,jax.numpy as jnp
from flax import nnx
from flax.nnx import bridge
from openpi.models import model,siglip
from mme_vla_suite.shared.data_utils import pool_tokens_to_size
from mme_vla_suite.shared.posemb_3d import PosEmb3D
from coin_data import DATA,BASE

def main():
 assert jax.default_backend()=='gpu'
 meta=json.loads((DATA/'dataset.json').read_text()); start=time.time()
 params=model.restore_params(BASE/'params',restore_type=np.ndarray)['PaliGemma']['img']
 img=bridge.ToNNX(siglip.Module(num_classes=2048,variant='So400m/14',pool_type='none',scan=True,dtype_mm='bfloat16'))
 img.lazy_init(jnp.ones((1,224,224,3)),train=False,rngs=nnx.Rngs(42))
 graph,state=nnx.split(img); state.replace_by_pure_dict(jax.tree.map(jnp.asarray,params)); img=nnx.merge(graph,state); del params;gc.collect()
 @jax.jit
 def encode(x):
  tokens,_=img(x,train=False)
  return pool_tokens_to_size(tokens[:,None],16)[:,0].astype(jnp.float16)
 max_frames=max(e['frames'] for e in meta['episodes'])
 np.save(DATA/'positions.npy',np.asarray(PosEmb3D(768)(jnp.arange(max_frames),4)))
 for ep in meta['episodes']:
  p=DATA/f"episode_{ep['episode']:03d}"; dst=p/'features.npy'
  if dst.exists():continue
  images=np.load(p/'images.npy',mmap_mode='r'); n=len(images)
  out=np.lib.format.open_memmap(p/'features.tmp.npy',mode='w+',dtype=np.float16,shape=(n,16,2048))
  for i in range(0,n,32):
   x=np.asarray(images[i:i+32,0],dtype=np.float32)/127.5-1; count=len(x)
   x=np.pad(x,((0,32-count),(0,0),(0,0),(0,0)),mode='edge')
   out[i:i+count]=np.asarray(encode(x))[:count]
  assert np.isfinite(out).all();out.flush();del out;(p/'features.tmp.npy').rename(dst)
  print(json.dumps({'episode':ep['episode'],'frames':n,'elapsed':time.time()-start}),flush=True)
 (DATA/'features-complete.json').write_text(json.dumps({'episodes':100,'elapsed':time.time()-start,'checkpoint':str(BASE),'history_fps':10}))
if __name__=='__main__':main()
