import numpy as np,torch
from types import SimpleNamespace
from multiprocessing.connection import Listener
from robomme_sim.eval_success import build_policy
args=SimpleNamespace(pretrained_checkpoint='/cache/robot-baselines/models/SimpleMemVLA/RoboMME',compute_dtype='bfloat16',attn_implementation='sdpa',pipelined=False,num_denoising_steps=10,eval_temperature=1.0,max_subtask_tokens=64)
torch.manual_seed(0);np.random.seed(0)
from transformers.initialization import no_init_weights
with no_init_weights():
 batched,factory,normalizer=build_policy(args)
buffer=factory()
print('POLICY READY',flush=True)
with Listener(('127.0.0.1',8016),authkey=b'local-robomme-simplemem') as listener:
 with listener.accept() as conn:
  while True:
   try:r=conn.recv()
   except EOFError:break
   try:
    if r.get('reset'):
     buffer.reset();reply={'reset_finished':True}
    elif r.get('add_buffer'):
     for main,wrist in zip(r['images'],r['wrists']):buffer.observe({'observation.images.front':main,'observation.images.wrist':wrist})
     reply={'add_buffer_finished':True}
    else:
     state=None
     if normalizer is not None:state=normalizer.normalize(torch.from_numpy(np.asarray(r['observation/state'],dtype=np.float32))).unsqueeze(0).to(batched.device)
     with torch.inference_mode():actions,subtask=batched.generate_batch([buffer._prepare_inputs(r['prompt'])],[state])[0]
     reply={'actions':np.asarray(actions),'subtask':subtask}
    conn.send(reply)
   except Exception as e:
    conn.send({'error':repr(e)});raise
