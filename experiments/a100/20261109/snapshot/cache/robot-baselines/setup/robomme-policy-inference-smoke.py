"""Load the official checkpoint and infer with synthetic observations."""
import json
import os
from pathlib import Path
import time
import jax
import numpy as np
from mme_vla_suite.training import config
from mme_vla_suite.policies import policy_config

base=Path('/cache/robot-baselines')
checkpoint=base/'models/RoboMME/perceptual-framesamp-modul/79999'
print('JAX_DEVICES',jax.devices(),flush=True)
assert jax.default_backend()=='gpu'
t0=time.monotonic()
policy=policy_config.create_trained_policy(config.get_config('mme_vla_suite'),checkpoint,seed=7)
load_sec=time.monotonic()-t0
print('POLICY_LOADED',load_sec,flush=True)
assert policy.config.representation_type=='perceptual'
assert policy.config.perceptual_memory.type=='frame_sampling'
assert policy.config.integration_type=='modulation'
rng=np.random.default_rng(7)
state=np.asarray(policy.state_norm_stats.mean,dtype=np.float32)
image=rng.integers(0,256,(224,224,3),dtype=np.uint8)
wrist=rng.integers(0,256,(224,224,3),dtype=np.uint8)
obs={'observation/image':image,'observation/wrist_image':wrist,
     'observation/state':state,'prompt':'Pick up the red cube.'}
policy.add_buffer({'images':image[None,None], 'state':state[None]})
print('HISTORY_BUFFER_READY',flush=True)
results=[]
for i in range(2):
    if i:
        policy.add_buffer({'images':image[None,None], 'state':state[None]})
    start=time.monotonic()
    output=policy.infer(obs)
    actions=np.asarray(output['actions'])
    assert actions.shape==(20,8),actions.shape
    assert np.isfinite(actions).all()
    results.append({'pass':i+1,'elapsed_sec':time.monotonic()-start,'shape':list(actions.shape),
                    'min':float(actions.min()),'max':float(actions.max()),'finite':True})
    print('INFERENCE_OK',json.dumps(results[-1]),flush=True)
report={'checkpoint':str(checkpoint),'jax_version':jax.__version__,'devices':[str(d) for d in jax.devices()],
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'synthetic_observations':True,'subtask_annotations_provided':False,
        'history_frames':len(policy.mem_buffer._history_feats),'load_sec':load_sec,'inferences':results}
out=base/'eval-results/robomme-framesamp-inference';out.mkdir(exist_ok=True)
np.save(out/'actions.npy',actions)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('INFERENCE_SMOKE_PASSED',str(out),flush=True)
